import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt

data = pd.read_csv('coin_Bitcoin.csv')

# Scaling data
actual_price = data['Close'].values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(actual_price)

# sequence of 30 will predict next value
pred_days=30

# creating x_train and y_train
x_train, y_train = [], []

for x in range(pred_days, len(scaled_data)):
  x_train.append(scaled_data[x-pred_days:x, 0])
  y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

x_test = x_train[2400:2863]
y_test = y_train[2400:2863].reshape(-1,1)
x_train = x_train[0:2400]
y_train = y_train[0:2400].reshape(-1,1)

# creating model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(optimizer=opt, loss='mean_squared_error')
history = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_test, y_test))

# predicting and plotting data
prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

plt.plot(actual_price[2400:2863], color='red', label='Actual Prices')
plt.plot(prediction_prices, color='yellow', label='Prediction Prices')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend(loc="upper left")
plt.show()