import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


#cargamos el excel 

df = pd.read_excel(r'C:\Users\rcaceres\OneDrive - Hinojosa Packaging Group\PEDIDOS_PROPHET.xlsx')

#ordenes de produccion cerradas y que solo sean ordenes de produccion
df_filtered = df[(df['ser.ser_descritivo']=='Orden de Producción') & (df['Estado_OP']=='(F)Cerrada')]

#Convertimos la fecha
df_filtered['Fecha_Pedido']=pd.to_datetime(df_filtered['Fecha_Pedido'],format='%d/%m/%Y', errors= 'coerce')

prediction_type = input("Selecciona el tipo de predicción volumen(unidades)/valor en euros:")

#se escoge la columna segun lo que se hya determinado
if prediction_type == "Volumen(unidades)":
    df_filtered = df_filtered.rename(columns={'encln.encln_qtd':'y'})
elif prediction_type == "Valor en euros":
    df_filtered = df_filtered.rename(columns={'Valor_Linea_Pedido':'y'})
    
# Preparar el DataFrame para Prophet
df_prophet = df_filtered[['Fecha_Pedido', 'y']].rename(columns={'Fecha_Pedido': 'ds'})
df_prophet = df_prophet.dropna()

# Verificar que el DataFrame tiene las columnas necesarias
if 'ds' not in df_prophet.columns or 'y' not in df_prophet.columns:
    raise ValueError("El DataFrame debe tener las columnas 'ds' y 'y' para proceder con Prophet.")
# Crear el modelo Prophet
model = Prophet()
model.fit(df_prophet)

# Hacer una predicción de los próximos 12 meses
future = model.make_future_dataframe(periods=12, freq='ME')
forecast = model.predict(future)

#Mostrar
# Mostrar la predicción de forma correcta
print("### Predicción de entrada de pedidos")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])


#vamos a plotear

fig1 = model.plot(forecast)
plt.xlabel("Fecha")
plt.ylabel(prediction_type)
plt.show()

# Gráfico de los componentes de la predicción (tendencia, estacionalidad)
fig2 = model.plot_components(forecast)
plt.show()