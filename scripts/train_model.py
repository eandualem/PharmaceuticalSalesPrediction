import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TrainModel():
  '''
  Class for training a model using sklearn pipeline
  '''

  def __init__(self, model, name):
    self.model = model
    self.name = name

  def scaler(self):
    scaler = StandardScaler()
    return scaler

  def pipeline(self, X, y):
    train_pipe = Pipeline(
      [('Scaling', self.scaler()),
       ('Random Forest', self.model())])
    return train_pipe.fit(X, y)

  def train(self, X, y):
    mlflow.set_experiment('Random Forest')
    mlflow.sklearn.autolog()
    self.pipeline(X, y)

  def train_sales(self):
    X = pd.read_csv('../features/train_features.csv')
    y = pd.read_csv('../features/train_sales.csv')
    self.train(X, y)

  def train_customers(self):
    X = pd.read_csv('../features/train_features.csv')
    y = pd.read_csv('../features/train_customers.csv')
    self.train(X, y)
