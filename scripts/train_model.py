import mlflow
import pandas as pd
import xgboost.xgb as XGBRegressor
import lightgbm.gbm as LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def Scaler():
  scaler = StandardScaler()
  return scaler


def rf_model():
  model = RandomForestRegressor(n_jobs=-1, n_estimators=15)
  return model


def xgb_model():
  model = XGBRegressor()
  return model


def gbm_model():
  model = LGBMRegressor()
  return model


def model_pipeline(X, y):
  my_pipe = Pipeline(
    [('Scaling', Scaler()),
     ('Random Forest', rf_model())])
  return my_pipe.fit(X, y)


train = pd.read_csv('../features/train_features.csv')
test = pd.read_csv('../features/test_features.csv')
train = train[train['Open'] != 0]

X = train.drop(['Sales', 'Customers', "Store"], axis=1)
y = train.Sales

mlflow.set_experiment('Random Forest')
mlflow.sklearn.autolog()
model_pipeline(X, y)
