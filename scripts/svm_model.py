from train_model import TrainModel
from sklearn.svm import SVR


def model():
  model = SVR(kernel='linear', degree=3, gamma='auto', coef0=0.0, tol=0.001,
              C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
  return model


train_model = TrainModel(model, "SVR")
sales_model = train_model.train_sales()
customers_model = train_model.train_customers()
