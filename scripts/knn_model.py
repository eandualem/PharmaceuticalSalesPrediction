from train_model import TrainModel
from sklearn.neighbors import KNeighborsRegressor


def model():
  model = KNeighborsRegressor(n_neighbors=2, weights='distance', p=1)
  return model


train_model = TrainModel(model, "KNeighborsRegressor")
sales_model = train_model.train_sales()
customers_model = train_model.train_customers()
