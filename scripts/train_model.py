import mlflow
import pandas as pd
from config import Config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from file_handler import FileHandler
from evaluate_model import EvaluateModel


Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
Config.METRICS_FILE_PATH.mkdir(parents=True, exist_ok=True)


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

  def train(self, X, y, X_val, y_val, model_type):
    mlflow.set_experiment(self.name)
    mlflow.sklearn.autolog()

    train_pipe = Pipeline(
      [('Scaling', self.scaler()),
       (self.name, self.model())])

    pipe = train_pipe.fit(X, y.to_numpy().ravel())
    y_pred = pipe.predict(X_val)
    em = EvaluateModel(y_val, y_pred)
    metrics = em.get_metrics()
    self.file_handler.save_metrics(metrics, str(self.name + "-" + model_type))
    self.file_handler.save_model(pipe.steps[1][1], str(self.name + "-" + model_type))
    return

  def train_sales(self):
    X = pd.read_csv('../features/train_features.csv')
    y = pd.read_csv('../features/train_sales.csv')
    X_val = pd.read_csv('../features/test_features.csv')
    y_val = pd.read_csv('../features/test_sales.csv')
    self.train(X, y, X_val, y_val, 'sales')

  def train_customers(self):
    X = pd.read_csv('../features/train_features.csv')
    y = pd.read_csv('../features/train_customers.csv')
    X_val = pd.read_csv('../features/test_features.csv')
    y_val = pd.read_csv('../features/test_customers.csv')
    self.train(X, y, X_val, y_val, 'customers')
