import mlflow
import numpy as np
import tensorflow as tf
from time import gmtime, strftime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from file_handler import FileHandler
from sklearn.preprocessing import StandardScaler
from config import Config
import matplotlib.pyplot as plt


class TimeSeriesModel:

  def __init__(self, data, WINDOW_SIZE, EPOCHS, model_name):
    self.EPOCHS = EPOCHS
    self.WINDOW_SIZE = WINDOW_SIZE
    self.BATCH_SIZE = len(data) - WINDOW_SIZE * 2
    self.data = self.get_scaled_data(data)
    self.model_name = model_name

  def train_model(self):
    XTrain = self.data.DataScaled.values[0:self.BATCH_SIZE].astype('float32')
    XValid = self.data.DataScaled.values[self.BATCH_SIZE:].astype('float32')

    DatasetTrain = self.windowed_dataset(XTrain)
    DatasetVal = self.windowed_dataset(XValid)

    model = self.build_model()
    mlflow.set_experiment('Tensorflow')
    mlflow.tensorflow.autolog()
    history = model.fit(DatasetTrain, epochs=self.EPOCHS, validation_data=DatasetVal, verbose=1)
    return model, history

  def get_scaled_data(self, data):
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(data)
    data['DataScaled'] = scaled_array
    return data

  def build_model(self):
    model = Sequential()
    model.add(LSTM(32, input_shape=[None, 1], return_sequences=True))
    model.add(LSTM(16, input_shape=[None, 1]))
    model.add(Dense(1))
    model.compile(loss="huber_loss", optimizer='adam')
    model.summary()
    return model

  def windowed_dataset(self, series):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(self.WINDOW_SIZE + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(self.WINDOW_SIZE + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(self.BATCH_SIZE).prefetch(1)
    return dataset

  def model_forecast(self, model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(len(series)).prefetch(1)
    forecast = model.predict(ds)
    return forecast

  def plot_history(self, history):
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['loss'], label="loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.legend()
    # time = strftime("%Y-%m-%d-%H:%M", gmtime())
    name = Config.IMAGE_PATH / str(self.model_name + ".png")
    plt.savefig(name, dpi=120)
    plt.close()

  def save_model(self, model):
    time = strftime("%Y-%m-%d-%H:%M", gmtime())
    name = Config.MODELS_PATH / str(self.model_name + '-' + time)
    model.save(name)


WINDOW_SIZE = 48
EPOCHS = 100


def model(data, model_name):
  model = TimeSeriesModel(data, WINDOW_SIZE, EPOCHS, model_name)
  return model


file_handler = FileHandler()
train_df = file_handler.read_csv("../pages/train.csv")

sales_data = train_df.groupby("Date").agg({"Sales": "mean"})
new_model = model(sales_data, "TimeSeriesModel-Sales")
sales_model, history = new_model.train_model()
new_model.save_model(sales_model)
new_model.plot_history(history)

customers_data = train_df.groupby("Date").agg({"Customers": "mean"})
new_model = model(customers_data, "TimeSeriesModel-Customers")
customers_model, history = new_model.train_model()
new_model.save_model(customers_model)
new_model.plot_history(history)
