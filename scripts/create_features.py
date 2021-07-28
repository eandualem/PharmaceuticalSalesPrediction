import pandas as pd
from config import Config
from sklearn.preprocessing import LabelEncoder

'''
This is a simple script for creating features for train and test data
I have extracted 6 features for training the model
'''

Config.FEATURES_PATH.mkdir(parents=True, exist_ok=True)
train_df = pd.read_csv(str(Config.DATASET_PATH / "train.csv"))
test_df = pd.read_csv(str(Config.DATASET_PATH / "test.csv"))


def extract_features(df):

  # date
  df["date"] = pd.to_datetime(df.date).dt.date
  df["date_of_week"] = pd.to_datetime(df.date).dt.dayofweek
  df["date_of_month"] = pd.to_datetime(df.date).dt.day

  # change categorical variables to numerical value
  lb = LabelEncoder()
  df['experiment'] = lb.fit_transform(df['experiment'])
  df['device_make'] = lb.fit_transform(df['device_make'])
  df['platform_os'] = lb.fit_transform(df['platform_os'])
  df['browser'] = lb.fit_transform(df['browser'])

  return df[["experiment", "hour", "date_of_week", "date_of_month", 'device_make', 'platform_os', 'browser']]


def extract_labels(df):
  df['aware'] = 0
  df.loc[df['yes'] == 1, 'aware'] = 1
  df.loc[df['yes'] == 0, 'aware'] = 0
  return df[["aware"]]


train_features = extract_features(train_df)
test_features = extract_features(test_df)

train_labels = extract_labels(train_df)
test_labels = extract_labels(test_df)

train_features.to_csv(str(Config.FEATURES_PATH / "train_features.csv"), index=None)
test_features.to_csv(str(Config.FEATURES_PATH / "test_features.csv"), index=None)

train_labels.to_csv(str(Config.FEATURES_PATH / "train_labels.csv"), index=None)
test_labels.to_csv(str(Config.FEATURES_PATH / "test_labels.csv"), index=None)
