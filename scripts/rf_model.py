from config import Config
from train_model import TrainModel
from evaluate_model import EvaluateModel
from file_handler import FileHandler
from sklearn.ensemble import RandomForestRegressor

file_handler = FileHandler()
Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)


def model():
  model = RandomForestRegressor(n_jobs=-1, n_estimators=15)
  return model


train_model = TrainModel(model, "RandomForestRegressor")

sales_model = train_model.train_sales()
customers_model = train_model.train_customers()
