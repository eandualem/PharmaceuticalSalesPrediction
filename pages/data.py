import streamlit as st
import pandas as pd


@st.cache
def loadStore():
  df = pd.read_csv("./pages/store.csv", nrows=2000)  # for performance
  return df


@st.cache
def loadTrain():
  df = pd.read_csv("./pages/train.csv", nrows=2000)  # for performance
  return df


@st.cache
def loadFeatures():
  df = pd.read_csv("./features/train_features.csv", nrows=2000)  # for performance
  return df


def app():  
  st.title('Data description')

  st.header('Train Data')
  train_df = loadTrain().copy()
  st.write(train_df)
  st.write('Train data contains historical data including Sales and Customers')


  st.header('Store Data')
  store_df = loadStore().copy()
  st.write(store_df)
  st.write('Train data contains supplemental information about the stores')

  
  st.header('Features')
  feature_df = loadFeatures().copy() 
  st.write(feature_df)
  st.write('Features created using the joint columns of test and train.')
