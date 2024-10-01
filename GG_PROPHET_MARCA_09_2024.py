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

# Obtener las marcas únicas
marcas_unicas = subfamilia_filtered['MARCA'].unique()

# Iterar sobre cada marca y aplicar Prophet
for marca in marcas_unicas:
    print(f"Predicción para la marca: {marca}")
    
    # Filtrar los datos por la marca actual
    marca_filtered = subfamilia_filtered[subfamilia_filtered['MARCA'] == marca]
    
    # --- CAMBIO: Agrupar los datos por semana o mes ---
    # Si prefieres semanal, usa 'W'; para mensual, usa 'M'
    marca_filtered['Fecha'] = pd.to_datetime(marca_filtered['Fecha'])
    marca_filtered = marca_filtered.resample('W', on='Fecha').sum().reset_index()  # 'W' para semanas, 'M' para meses
    
    # Preparar los datos para Prophet
    df_prophet_marca = marca_filtered[['Fecha', 'KG Consumidos']].rename(columns={'Fecha': 'ds', 'KG Consumidos': 'y'})
    
    # Verificar si hay al menos 2 filas válidas para continuar
    if len(df_prophet_marca) < 2:
        print(f"Marca {marca} tiene menos de 2 filas de datos válidos. Saltando...")
        continue
    
    # Instanciar y ajustar el modelo Prophet
    model_prophet_marca = Prophet()
    model_prophet_marca.fit(df_prophet_marca)
    
    # Crear un marco de datos para los próximos 12 meses (semanales o mensuales)
    future_marca = model_prophet_marca.make_future_dataframe(periods=52, freq='W')  # Cambia 'W' a 'M' para agrupar por meses
    
    # Realizar la predicción
    forecast_prophet_marca = model_prophet_marca.predict(future_marca)
    
    # Graficar los datos reales en azul
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_prophet_marca['ds'], df_prophet_marca['y'], label='Datos Reales', color='blue')
    
    # Graficar las predicciones en rojo
    ax.plot(forecast_prophet_marca['ds'], forecast_prophet_marca['yhat'], label='Predicción', color='red')
    
    # Agregar bandas de incertidumbre en color rosado
    ax.fill_between(forecast_prophet_marca['ds'], forecast_prophet_marca['yhat_lower'], forecast_prophet_marca['yhat_upper'], color='pink', alpha=0.3)
    
    # Títulos y etiquetas
    plt.title(f'Predicción de Consumo (KG) por MARCA: {marca} - Agrupado por Semana')
    plt.xlabel('Fecha')
    plt.ylabel('KG Consumidos')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
