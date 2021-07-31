from sklearn.ensemble import RandomForestRegressor


def rf_model(self):
  model = RandomForestRegressor(n_jobs=-1, n_estimators=15)
  return model
