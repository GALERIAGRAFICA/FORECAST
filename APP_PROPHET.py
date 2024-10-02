import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from PIL import Image

# LOGO GG
# Cargar el logo
logo = Image.open("Logo.png")

# Mostrar el logo en la app
st.image(logo, width=150, use_column_width=True)

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

    # Primer nivel de filtrado: Seleccionar subfamilia
    subfamilias = grouped_consumption['Papeles.SubFamilia'].unique()
    subfamilia_seleccionada = st.selectbox("Selecciona la subfamilia", subfamilias)

    # Filtrar por la subfamilia seleccionada
    subfamilia_filtered = grouped_consumption[grouped_consumption['Papeles.SubFamilia'] == subfamilia_seleccionada]

    # Segundo nivel de filtrado: Seleccionar si se quiere filtrar por gramaje
    gramajes_unicos = subfamilia_filtered['Papeles.Gramaje'].unique()
    gramaje_seleccionado = st.selectbox("Selecciona el gramaje (opcional)", ['Ninguno'] + list(gramajes_unicos))

    if gramaje_seleccionado != 'Ninguno':
        # Filtrar por gramaje seleccionado
        gramaje_filtered = subfamilia_filtered[subfamilia_filtered['Papeles.Gramaje'] == gramaje_seleccionado]
    else:
        gramaje_filtered = subfamilia_filtered

    # Tercer nivel de filtrado: Seleccionar si se quiere filtrar por marca o certificado
    tipo_analisis = st.selectbox("Selecciona análisis adicional (opcional)", ["Ninguno", "Marca", "Certificado"])

    if tipo_analisis == "Marca":
        # Filtrar por marca
        marcas_unicas = gramaje_filtered['MARCA'].unique()
        marca_seleccionada = st.selectbox("Selecciona la marca", marcas_unicas)

        # Filtrar por la marca seleccionada
        analisis_filtered = gramaje_filtered[gramaje_filtered['MARCA'] == marca_seleccionada]

    elif tipo_analisis == "Certificado":
        # Filtrar por certificado
        certificados_unicos = gramaje_filtered['CERTIFICADO'].unique()
        certificado_seleccionado = st.selectbox("Selecciona el certificado", certificados_unicos)

        # Filtrar por el certificado seleccionado
        analisis_filtered = gramaje_filtered[gramaje_filtered['CERTIFICADO'] == certificado_seleccionado]

    else:
        # Si no se selecciona análisis adicional, usamos el filtrado hasta el nivel anterior
        analisis_filtered = gramaje_filtered

    # Agrupar los datos por semana y fecha
    analisis_filtered = analisis_filtered.resample('W', on='Fecha').sum().reset_index()

    # Preparar los datos para Prophet
    df_prophet = analisis_filtered[['Fecha', 'KG Consumidos']].rename(columns={'Fecha': 'ds', 'KG Consumidos': 'y'})

    # Verificar si hay suficientes datos para Prophet
    if len(df_prophet) >= 2:
        # Instanciar y ajustar el modelo Prophet
        model_prophet = Prophet()
        model_prophet.fit(df_prophet)

        # Crear un marco de datos para los próximos 12 meses (semanales)
        future = model_prophet.make_future_dataframe(periods=52, freq='W')

        # Realizar la predicción
        forecast = model_prophet.predict(future)

        # Graficar los resultados
        fig, ax = plt.subplots(figsize=(10, 6))

        # Graficar los datos reales hasta el último punto
        ax.plot(df_prophet['ds'], df_prophet['y'], label='Datos Reales', color='blue')

        # Graficar las predicciones desde el último punto en rojo
        ax.plot(forecast['ds'], forecast['yhat'], label='Predicción', color='red')

        # Agregar bandas de incertidumbre a la predicción (opcional)
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='pink', alpha=0.3)

        # Mostrar el gráfico en la app de Streamlit
        plt.title(f'Predicción de Consumo (KG) - Subfamilia: {subfamilia_seleccionada}')
        if gramaje_seleccionado != 'Ninguno':
            plt.title(f'Predicción de Consumo (KG) - Subfamilia: {subfamilia_seleccionada}, Gramaje: {gramaje_seleccionado}')
        if tipo_analisis == "Marca":
            plt.title(f'Predicción de Consumo (KG) - Subfamilia: {subfamilia_seleccionada}, Marca: {marca_seleccionada}')
        elif tipo_analisis == "Certificado":
            plt.title(f'Predicción de Consumo (KG) - Subfamilia: {subfamilia_seleccionada}, Certificado: {certificado_seleccionado}')

        plt.xlabel('Fecha')
        plt.ylabel('KG Consumidos')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write(f"No hay suficientes datos para realizar la predicción para la subfamilia {subfamilia_seleccionada}.")
