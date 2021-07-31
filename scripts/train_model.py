import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from file_handler import FileHandler


class TrainModel():
  '''
  Class for training a model using sklearn pipeline
  '''

  def __init__(self, model, name):
    self.model = model
    self.name = name
    self.file_handler = FileHandler()

  def scaler(self):
    scaler = StandardScaler()
    return scaler

  def pipeline(self, X, y):
    train_pipe = Pipeline(
      [('Scaling', self.scaler()),
       (self.name, self.model())])

    pipe = train_pipe.fit(X, y)
    self.file_handler.save_model(pipe.steps[1][1], str(self.name + "-sales"))
    return

  def train(self, X, y, model_type):
    mlflow.set_experiment(self.name)
    mlflow.sklearn.autolog()

    train_pipe = Pipeline(
      [('Scaling', self.scaler()),
       (self.name, self.model())])

    pipe = train_pipe.fit(X, y)
    self.file_handler.save_model(pipe.steps[1][1], str(self.name + "-" + model_type))
    return

  def train_sales(self):
    X = pd.read_csv('../features/train_features.csv')
    y = pd.read_csv('../features/train_sales.csv')
    self.train(X, y, 'sales')

  def train_customers(self):
    X = pd.read_csv('../features/train_features.csv')
    y = pd.read_csv('../features/train_customers.csv')
    self.train(X, y, 'customers')
