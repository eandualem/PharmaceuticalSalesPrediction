from train_model import TrainModel
from lightgbm import LGBMRegressor


def model():
  model = LGBMRegressor()
  return model


train_model = TrainModel(model, "LGBMRegressor")
sales_model = train_model.train_sales()
customers_model = train_model.train_customers()
