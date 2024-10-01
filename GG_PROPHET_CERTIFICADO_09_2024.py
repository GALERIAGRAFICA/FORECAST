import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Cargar el archivo Excel desde la ruta
file_path = r"C:\Users\rcaceres\OneDrive - Hinojosa Packaging Group\00GG_CONSUMOS_X_PAPEL.xlsx"
df = pd.read_excel(file_path, sheet_name='00GG_CONTROL_CONSUMOS')

# Convertir 'Fecha' a formato datetime
df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

# Agrupar por familia, subfamilia, certificación, gramaje, marca, y fecha
grouped_consumption = df.groupby(['Papeles.Familia', 'Papeles.SubFamilia', 'CERTIFICADO', 'Papeles.Gramaje', 'MARCA', 'Fecha']).agg({
    'KG Consumidos': 'sum'
}).reset_index()

# Filtrar por la subfamilia seleccionada
subfamilia_filtered = grouped_consumption[grouped_consumption['Papeles.SubFamilia'] == 'Folding Blanco (GC1)']

# Obtener los certificados únicos
certificados_unicos = subfamilia_filtered['CERTIFICADO'].unique()
print(certificados_unicos)

# Iterar sobre cada certificado y aplicar Prophet
for certificado in certificados_unicos:
    print(f"Predicción para el certificado: {certificado}")
    
    # Filtrar los datos por el certificado actual
    certificado_filtered = subfamilia_filtered[subfamilia_filtered['CERTIFICADO'] == certificado]
    
    # --- CAMBIO: Agrupar los datos por semana o mes ---
    # Si prefieres semanal, usa 'W'; para mensual, usa 'M'
    certificado_filtered['Fecha'] = pd.to_datetime(certificado_filtered['Fecha'])
    certificado_filtered = certificado_filtered.resample('W', on='Fecha').sum().reset_index()  # 'W' para semanas, 'M' para meses
    
    # Preparar los datos para Prophet
    df_prophet_certificado = certificado_filtered[['Fecha', 'KG Consumidos']].rename(columns={'Fecha': 'ds', 'KG Consumidos': 'y'})
    
    # Verificar si hay al menos 2 filas válidas para continuar
    if len(df_prophet_certificado) < 2:
        print(f"Certificado {certificado} tiene menos de 2 filas de datos válidos. Saltando...")
        continue
    
    # Instanciar y ajustar el modelo Prophet
    model_prophet_certificado = Prophet()
    model_prophet_certificado.fit(df_prophet_certificado)
    
    # Crear un marco de datos para los próximos 12 meses (semanales o mensuales)
    future_certificado = model_prophet_certificado.make_future_dataframe(periods=52, freq='W')  # Cambia 'W' a 'M' para agrupar por meses
    
    # Realizar la predicción
    forecast_prophet_certificado = model_prophet_certificado.predict(future_certificado)
    
    # Graficar los datos reales en azul
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_prophet_certificado['ds'], df_prophet_certificado['y'], label='Datos Reales', color='blue')
    
    # Graficar las predicciones en rojo
    ax.plot(forecast_prophet_certificado['ds'], forecast_prophet_certificado['yhat'], label='Predicción', color='red')
    
    # Agregar bandas de incertidumbre en color rosado
    ax.fill_between(forecast_prophet_certificado['ds'], forecast_prophet_certificado['yhat_lower'], forecast_prophet_certificado['yhat_upper'], color='pink', alpha=0.3)
    
    # Títulos y etiquetas
    plt.title(f'Predicción de Consumo (KG) por CERTIFICADO: {certificado} - Agrupado por Semana')
    plt.xlabel('Fecha')
    plt.ylabel('KG Consumidos')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
