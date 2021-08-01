from train_model import TrainModel
from sklearn.ensemble import RandomForestRegressor


def model():
  model = RandomForestRegressor(n_estimators=200, 
                                verbose=True, 
                                max_depth=10, 
                                min_samples_split=2,
                                min_samples_leaf=1)
  return model


train_model = TrainModel(model, "RandomForestRegressor")

sales_model = train_model.train_sales()
customers_model = train_model.train_customers()
