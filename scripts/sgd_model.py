from train_model import TrainModel
from sklearn.linear_model import SGDRegressor


def model():
  model = SGDRegressor(eta0=0.1, fit_intercept=False, shuffle=False, learning_rate='adaptive', random_state=21,)
  return model


train_model = TrainModel(model, "SGDRegressor")

sales_model = train_model.train_sales()
customers_model = train_model.train_customers()
