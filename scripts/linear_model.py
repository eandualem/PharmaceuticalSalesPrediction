from train_model import TrainModel
from sklearn.linear_model import LinearRegression


def model():
  model = LinearRegression()
  return model


train_model = TrainModel(model, "LinearRegression")
sales_model = train_model.train_sales()
customers_model = train_model.train_customers()
