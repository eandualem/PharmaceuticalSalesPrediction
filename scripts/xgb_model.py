from config import Config
from train_model import TrainModel
from xgboost import XGBRegressor
from evaluate_model import EvaluateModel
from file_handler import FileHandler

file_handler = FileHandler()
Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)


def model():
  model = XGBRegressor(params={'max_depth': 10, 'eta': 0.1, 'gamma': 2})
  return model


train_model = TrainModel(model, "XGBRegressor")

sales_model = train_model.train_sales()
file_handler.save_model(sales_model, "XGBRegressor-sales")

customers_model = train_model.train_customers()
file_handler.save_model(customers_model, "XGBRegressor-customers")
