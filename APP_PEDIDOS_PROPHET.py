import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.dates as mdates
from datetime import datetime

# Cargar el archivo Excel
st.title('Predicción de Entrada de Pedidos')

uploaded_file = st.file_uploader("Sube el archivo Excel", type=["xlsx"])
st.info("Este dataset se filtra para líneas de pedido cerradas y solo órdenes de producción.")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Filtrar solo Órdenes de Producción cerradas
    #df_filtered = df[(df['ser.ser_descritivo'].str.strip() == 'Orden de Producción') & (df['Estado_OP'].str.strip() == '(F)Cerrada')]
    df_filtered = df[(df['ser.ser_descritivo'].str.strip() == 'Orden de Producción')]
    # Filtrar por Cliente y Comercial
    cliente_options = ['Todos'] + list(df_filtered['Cliente'].unique())
    selected_cliente = st.selectbox('Selecciona un Cliente', options=cliente_options)

    if selected_cliente != 'Todos':
        df_filtered = df_filtered[df_filtered['Cliente'] == selected_cliente]

    comercial_options = ['Todos'] + list(df_filtered['Comercial'].unique())
    selected_comercial = st.selectbox('Selecciona un Comercial', options=comercial_options, key='comercial')

    if selected_comercial != 'Todos':
        df_filtered = df_filtered[df_filtered['Comercial'] == selected_comercial]

    # Convertir 'Fecha_Pedido' a formato datetime
    df_filtered.loc[:, 'Fecha_Pedido'] = pd.to_datetime(df_filtered['Fecha_Pedido'], format='%d/%m/%Y', errors='coerce')

    # Revisar la calidad de la columna 'Fecha_Pedido'
    num_missing_dates = df_filtered['Fecha_Pedido'].isna().sum()
    total_rows = len(df_filtered)
    missing_percentage = (num_missing_dates / total_rows) * 100
    st.write(f"Total de filas: {total_rows}")
    st.write(f"Total de fechas faltantes: {num_missing_dates}")
    st.write(f"Porcentaje de fechas faltantes: {missing_percentage:.2f}%")

    # Filtrar filas con fechas válidas
    df_filtered = df_filtered.dropna(subset=['Fecha_Pedido'])

    # Convertir min_date y max_date a objetos datetime para el slider
    min_date = df_filtered['Fecha_Pedido'].min().to_pydatetime()
    max_date = df_filtered['Fecha_Pedido'].max().to_pydatetime()

    # Filtro para seleccionar el rango de fechas para la predicción
    selected_range = st.slider("Selecciona el rango de fechas para la predicción", min_value=min_date.date(), max_value=max_date.date(), value=(min_date.date(), max_date.date()))
    df_filtered = df_filtered[(df_filtered['Fecha_Pedido'] >= pd.to_datetime(selected_range[0])) & (df_filtered['Fecha_Pedido'] <= pd.to_datetime(selected_range[1]))]

    # Seleccionar tipo de predicción
    prediction_type = st.selectbox("Selecciona el tipo de predicción", ["Volumen (unidades)", "Valor en euros"])

    # Elegir la columna a proyectar según el tipo de predicción
    if prediction_type == "Volumen (unidades)":
        if 'encln.encln_qtd' in df_filtered.columns:
            df_filtered = df_filtered.rename(columns={'encln.encln_qtd': 'y'})
        else:
            st.error("La columna 'encln.encln_qtd' no se encuentra en el DataFrame.")
            st.stop()
    elif prediction_type == "Valor en euros":
        if 'Valor_Linea_Pedido' in df_filtered.columns:
            df_filtered = df_filtered.rename(columns={'Valor_Linea_Pedido': 'y'})
        else:
            st.error("La columna 'Valor_Linea_Pedido' no se encuentra en el DataFrame.")
            st.stop()

    # Seleccionar la frecuencia de predicción
    frequency = st.selectbox("Selecciona la frecuencia de la predicción", ["Diaria", "Mensual"], index=1)

    if frequency == "Mensual":
        # Agrupar datos por mes y año
        df_filtered['Mes_Año'] = df_filtered['Fecha_Pedido'].dt.to_period('M').dt.to_timestamp()
        df_grouped = df_filtered.groupby('Mes_Año').agg({'y': 'sum'}).reset_index()
        df_grouped['ds'] = df_grouped['Mes_Año']
    else:
        # Usar datos diarios
        df_filtered['ds'] = df_filtered['Fecha_Pedido']
        df_grouped = df_filtered[['ds', 'y']]

    # Preparar el DataFrame para Prophet
    df_prophet = df_grouped.dropna()

    # Preparar el DataFrame para Prophet
    df_prophet = df_grouped.dropna()

    # Verificar que el DataFrame tiene las columnas necesarias
    if 'ds' not in df_prophet.columns or 'y' not in df_prophet.columns:
        st.error("El DataFrame debe tener las columnas 'ds' y 'y' para proceder con Prophet.")
        st.stop()

    # Crear el modelo Prophet
    model = Prophet()
    model.fit(df_prophet)

    # Hacer una predicción de los próximos 12 meses
    future = model.make_future_dataframe(periods=13, freq='M' if frequency == "Mensual" else 'D')
    forecast = model.predict(future)

    # Mostrar los resultados
    # Calcular el total mensual y anual
    total_mensual = forecast[['ds', 'yhat']].set_index('ds').resample('M').sum()
    total_mensual_real = df_prophet.set_index('ds').resample('M').sum(numeric_only=True)
    total_mensual['y_real'] = total_mensual_real['y']
    
    total_anual = total_mensual.resample('A').sum(numeric_only=True)
    
    st.write("### Predicción de Entrada de Pedidos")
    st.write("#### Totales Mensuales para el Próximo Año")
    total_mensual_display = total_mensual.copy()
    total_mensual_display.index = total_mensual_display.index.strftime('%B-%Y')
    total_mensual.index = total_mensual.index.strftime('%B-%Y')
    if prediction_type == "Valor en euros":
        st.dataframe(total_mensual_display.style.format({'yhat': "€{:.2f}", 'y_real': "€{:.2f}"}))
    else:
        st.dataframe(total_mensual_display.style.format({'yhat': "{:.0f}k", 'y_real': "{:.0f}k"}))
    
    st.write("#### Total Anual para el Próximo Año")
    total_anual_display = total_anual.copy()
    total_anual_display.index = total_anual_display.index.strftime('%Y')
    total_anual.index = total_anual.index.strftime('%Y')
    if prediction_type == "Valor en euros":
        st.dataframe(total_anual_display.style.format({'yhat': "€{:.2f}", 'y_real': "€{:.2f}"}))
    else:
        st.dataframe(total_anual_display.style.format({'yhat': "{:.0f}k", 'y_real': "{:.0f}k"}))
    
    
        st.write("### Predicción de Entrada de Pedidos")
    forecast_comparado = forecast.set_index('ds').join(df_prophet.set_index('ds'), rsuffix='_real')
    forecast_comparado = forecast_comparado[['y', 'yhat', 'yhat_lower', 'yhat_upper']]
    st.write("### Comparación de Predicción vs Datos Reales")
    st.write(forecast_comparado.tail())

    # Gráfico de la predicción
    fig1, ax1 = plt.subplots(figsize=(10, 6))  # Aumentar el tamaño del gráfico
    ax1.plot(df_prophet['ds'], df_prophet['y'], 'k-', label='Datos Reales')  # Cambiar puntos a línea para datos reales
    ax1.plot(forecast['ds'], forecast['yhat'], 'b-', label='Predicción')
    ax1.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='blue', alpha=0.2, label='Intervalo de Confianza')
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel(prediction_type)
    ax1.legend()
    ax1.grid(True)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # Mostrar etiquetas cada 3 meses
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    fig1.autofmt_xdate(rotation=45)  # Rotar etiquetas del eje X para mejorar visibilidad
    st.pyplot(fig1)

    # Gráfico de los componentes de la predicción (tendencia, estacionalidad)
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)
else:
    st.warning("Por favor, sube un archivo Excel para continuar.")
