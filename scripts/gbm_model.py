from config import Config
from train_model import TrainModel
from evaluate_model import EvaluateModel
from file_handler import FileHandler
from lightgbm import LGBMRegressor

file_handler = FileHandler()
Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)


def model():
  model = LGBMRegressor()
  return model


train_model = TrainModel(model, "LGBMRegressor")
sales_model = train_model.train_sales()
customers_model = train_model.train_customers()
