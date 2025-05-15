import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

df = pd.read_csv("data/energy.csv")

df['date'] = pd.to_datetime(df['date'])
df['day_of_year'] = df['date'].dt.dayofyear

X = df['day_of_year'].values.reshape(-1, 1) / 365.0 
y = df['consumption_kwh'].values

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=500, verbose=0)

days = np.linspace(1, 365, 365).reshape(-1, 1) / 365.0
predictions = model.predict(days)

plt.figure(figsize=(12, 6))
plt.scatter(df['day_of_year'], y, label='Реальні дані', color='black')
plt.plot(np.linspace(1, 365, 365), predictions, label='Нейронна мережа', color='blue')
plt.title("Прогноз споживання електроенергії протягом року")
plt.xlabel("День року")
plt.ylabel("Споживання (kWh)")
plt.grid(True)
plt.legend()
plt.show()
