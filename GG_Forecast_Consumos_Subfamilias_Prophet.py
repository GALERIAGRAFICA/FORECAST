import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Cargar el archivo desde la ruta local
file_path = r"C:\Users\rcaceres\OneDrive - Hinojosa Packaging Group\00GG_CONSUMOS_X_PAPEL.xlsx"

# Leer el archivo Excel
df = pd.read_excel(file_path, sheet_name='00GG_CONTROL_CONSUMOS')

# Convertir 'Fecha' a formato datetime
df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

# Agrupar los consumos por fecha
df_consumption = df[['Fecha', 'KG Consumidos']].groupby('Fecha').sum().reset_index()

# Preparar los datos para Prophet
df_prophet = df_consumption.rename(columns={'Fecha': 'ds', 'KG Consumidos': 'y'})

# Instanciar el modelo Prophet
model_prophet = Prophet()

# Ajustar el modelo
model_prophet.fit(df_prophet)

# Crear un marco de datos para los próximos 12 meses (365 días)
future = model_prophet.make_future_dataframe(periods=365)

# Hacer la predicción
forecast_prophet = model_prophet.predict(future)

# Graficar la predicción
fig = model_prophet.plot(forecast_prophet)
plt.title('Predicción de Consumo de Papel (KG) usando Prophet para los Próximos 12 Meses')
plt.xlabel('Fecha')
plt.ylabel('KG Consumidos')
plt.grid(True)
plt.tight_layout()
plt.show()
