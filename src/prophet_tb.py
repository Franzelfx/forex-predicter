import pandas as pd
from prophet import Prophet
from matplotlib import pyplot as plt

# Load data
df = pd.read_csv("src/testbench/pairs/USDJPY_15.csv")
df = df.rename(columns={"t": "ds", "c": "y"})
df = df[["ds", "y"]]
print(df.head(5))
plt.plot(df['y'])
plt.show()

# Create model
m = Prophet()
m.fit(df)

# Make predictions
future = m.make_future_dataframe(periods=1000)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(1000)

# Plot the data and the yhat values
plt.plot(forecast['ds'], forecast['yhat'])
plt.show()
