import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt



# Load the data
sheet_name = "00GG_CONTROL_CONSUMOS"
consumos_control_data = pd.read_excel(r"C:\Users\rcaceres\OneDrive - Hinojosa Packaging Group\Consumos.xlsx", sheet_name=sheet_name)

# Group by month and sum up the "KG Consumidos"
consumos_control_data['Fecha'] = pd.to_datetime(consumos_control_data['Fecha'])
monthly_consumption = consumos_control_data.groupby(consumos_control_data["Fecha"].dt.to_period("M"))["KG Consumidos"].sum().reset_index()
monthly_consumption['Fecha'] = monthly_consumption['Fecha'].dt.to_timestamp()


# Prepare data for Prophet
prophet_data = monthly_consumption.rename(columns={"Fecha": "ds", "KG Consumidos": "y"})

# Train the Prophet model
model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
model.fit(prophet_data)

# Make future dataframe for next 12 months prediction
future = model.make_future_dataframe(periods=12, freq='M')

# Predict with Prophet
forecast = model.predict(future)

# Plot the predictions
fig = model.plot(forecast)
plt.show()

predicted_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
print(predicted_values)

predicted_values.to_csv('predicted_values.csv', index=False)
