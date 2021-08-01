from train_model import TrainModel
from xgboost import XGBRegressor


def model():
  model = XGBRegressor(nthread=-1,
                       max_depth=10,
                       eta=0.02,
                       objective='reg:squarederror',
                       colsample_bytree=0.7,
                       subsample=0.7)
  return model


train_model = TrainModel(model, "XGBRegressor")
sales_model = train_model.train_sales()
customers_model = train_model.train_customers()
