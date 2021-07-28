import pandas as pd
from config import Config
from sklearn.preprocessing import LabelEncoder

'''
This is a simple script for creating features for train and test data
I have extracted 6 features for training the model
'''

Config.FEATURES_PATH.mkdir(parents=True, exist_ok=True)
train_df = pd.read_csv(str(Config.TRAIN_PATH))
test_df = pd.read_csv(str(Config.TEST_PATH))
store_df = pd.read_csv(str(Config.STORE_PATH))


def extract_features(df):

  # convert date column to datetime
  df['Date'] = pd.to_datetime(df.Date)

  # Feature creation
  df['Year'] = df.Date.dt.year
  df['Month'] = df.Date.dt.month
  df['Day'] = df.Date.dt.day
  df['DayOfWeek'] = df.Date.dt.dayofweek
  df['WeekOfYear'] = df.Date.dt.weekofyear

  df = df.set_index('Date')
  df = df.sort_index()

  return df


def merge(df, store):
  df_merge = pd.merge(df, store, on='Store')
  return df_merge


def extract_sales(df):
  return df[["Sales"]]

def extract_customers(df):
  return df[["Sales"]]


train_features = extract_features(merge(train_df, store_df))
test_features = extract_features(merge(test_df, store_df))

train_sales = extract_sales(train_df)
train_customers = extract_customers(train_df)

train_features.to_csv(str(Config.FEATURES_PATH / "train_features.csv"), index=None)
test_features.to_csv(str(Config.FEATURES_PATH / "test_features.csv"), index=None)

train_sales.to_csv(str(Config.FEATURES_PATH / "train_customers.csv"), index=None)
train_customers.to_csv(str(Config.FEATURES_PATH / "train_sales.csv"), index=None)
