import pandas as pd
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import tree, ensemble
import xgboost as xgb
import lightgbm as gbm
import mlflow


if __name__ == '__main__':
  train = pd.read_csv('../features/train_features.csv')
  test = pd.read_csv('../features/test_features.csv')
  train = train[train['Open'] != 0]

  def Scaler():
    scaler = preprocessing.StandardScaler()

    return scaler

  def rf_model():
    rf = ensemble.RandomForestRegressor(n_jobs=-1, n_estimators=15)

    return rf

  def xgb_model():
    model_xgb = xgb.XGBRegressor()

    return model_xgb

  def gbm_model():
    model_gbm = gbm.LGBMRegressor()

    return model_gbm

  # build pipeline for scaling and building model

  def model_pipe_rf(X, y):
    my_pipe = pipeline.Pipeline(
        [('Scaling', Scaler()),
         ('Random Forest', rf_model())
         ])
    return my_pipe.fit(X, y)

  X = train.drop(['Sales', 'Customers', "Store"], axis=1)
  y = train.Sales

  # run pipeline
  mlflow.set_experiment('Random Forest')
  mlflow.sklearn.autolog()
  model_pipe_rf(X, y)
