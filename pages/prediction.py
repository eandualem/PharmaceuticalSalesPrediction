import streamlit as st
# import tensorflow as tf
import pandas as pd
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


@st.cache
def loadTrain():
  df = pd.read_csv("./pages/train.csv")  # for performance
  return df


@st.cache
def loadFeatures():
  df = pd.read_csv("./features/train_features.csv")  # for performance
  return df


# def model_forecast(model, series, window_size):
#   ds = tf.data.Dataset.from_tensor_slices(series)
#   ds = ds.window(window_size, shift=1, drop_remainder=True)
#   ds = ds.flat_map(lambda w: w.batch(window_size))
#   ds = ds.batch(len(series)).prefetch(1)
#   forecast = model.predict(ds)
#   return forecast


# def forcast(df, model):
#   sales_data = df.groupby("Date").agg({"Sales": "mean"})[:200].copy()
#   scaler = StandardScaler()
#   scaled_array = scaler.fit_transform(sales_data)
#   sales_data['DataScaled'] = scaled_array
#   WINDOW_SIZE = 48
#   BATCH_SIZE = len(sales_data) - (WINDOW_SIZE * 2)

#   start = BATCH_SIZE - WINDOW_SIZE
#   Forecast = model_forecast(model, sales_data.DataScaled.values[:, np.newaxis], WINDOW_SIZE)
#   Results = Forecast[start:-1]
#   Results1 = scaler.inverse_transform(Results.reshape(-1, 1))

#   prediction_df = sales_data[start: start + len(Results1)]
#   prediction_df["Forecast"] = Results1
#   st.write(prediction_df[["Forecast", "Sales"]])

#   fig = plt.figure(figsize=(30, 8))
#   plt.title("LSTM Model Forecast Future sales")
#   sns.lineplot(x=prediction_df.index, y=prediction_df["Forecast"])
#   sns.lineplot(x=prediction_df.index, y=prediction_df["Sales"])
#   plt.xticks(rotation=90)
#   st.pyplot(fig)
  


def app():
  st.title('Sales Forcasting')
  st.header('Here is a sample forcast from the training data')
  # model = tf.keras.models.load_model("./models/TimeSeriesModel-Customers-2021-08-01-12:43.h5")
  # df = loadTrain().copy()
  # forcast(df, model)
  # st.header('Forcast on the test data')
