import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Título de la app
st.title('Predicción de Consumo de Papel')

# Subir el archivo Excel
uploaded_file = st.file_uploader("Sube el archivo Excel", type=["xlsx"])

if uploaded_file is not None:
    # Cargar el archivo Excel
    df = pd.read_excel(uploaded_file, sheet_name='00GG_CONTROL_CONSUMOS')

    # Convertir 'Fecha' a formato datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

    # Mostrar las primeras filas del archivo subido
    st.write("Vista previa del archivo cargado:")
    st.dataframe(df.head())

    # Agrupar por familia, subfamilia, certificación, gramaje, marca, y fecha
    grouped_consumption = df.groupby(['Papeles.Familia', 'Papeles.SubFamilia', 'CERTIFICADO', 'Papeles.Gramaje', 'MARCA', 'Fecha']).agg({
        'KG Consumidos': 'sum'
    }).reset_index()

    # Seleccionar subfamilia
    subfamilias = grouped_consumption['Papeles.SubFamilia'].unique()
    subfamilia_seleccionada = st.selectbox("Selecciona la subfamilia", subfamilias)

    # Filtrar por la subfamilia seleccionada
    subfamilia_filtered = grouped_consumption[grouped_consumption['Papeles.SubFamilia'] == subfamilia_seleccionada]

    # Agrupar los datos por semana
    subfamilia_filtered['Fecha'] = pd.to_datetime(subfamilia_filtered['Fecha'])
    subfamilia_filtered = subfamilia_filtered.resample('W', on='Fecha').sum().reset_index()

    # Preparar los datos para Prophet
    df_prophet_subfamilia = subfamilia_filtered[['Fecha', 'KG Consumidos']].rename(columns={'Fecha': 'ds', 'KG Consumidos': 'y'})

    # Instanciar y ajustar el modelo Prophet
    model_prophet_subfamilia = Prophet()
    model_prophet_subfamilia.fit(df_prophet_subfamilia)

    # Crear un marco de datos para los próximos 12 meses (semanales)
    future_subfamilia = model_prophet_subfamilia.make_future_dataframe(periods=52, freq='W')

    # Realizar la predicción
    forecast_prophet_subfamilia = model_prophet_subfamilia.predict(future_subfamilia)

    # Graficar los resultados
    fig, ax = plt.subplots(figsize=(10, 6))

    # Graficar los datos reales hasta el último punto
    ax.plot(df_prophet_subfamilia   ['ds'], df_prophet_subfamilia['y'], label='Datos Reales', color='blue')

    # Graficar las predicciones desde el último punto en rojo
    ax.plot(forecast_prophet_subfamilia['ds'], forecast_prophet_subfamilia['yhat'], label='Predicción', color='red')

    # Agregar bandas de incertidumbre a la predicción (opcional)
    ax.fill_between(forecast_prophet_subfamilia['ds'], forecast_prophet_subfamilia['yhat_lower'], forecast_prophet_subfamilia['yhat_upper'], color='pink', alpha=0.3)

    plt.title(f'Predicción de Consumo (KG) para la Subfamilia "{subfamilia_seleccionada}" - Datos Semanales')
    plt.xlabel('Fecha')
    plt.ylabel('KG Consumidos')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Mostrar el gráfico en la app de Streamlit
    st.pyplot(fig)
