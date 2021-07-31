import pickle
import pandas as pd
from config import Config
from train_model import TrainModel
from evaluate_model import EvaluateModel
from file_handler import FileHandler
from sklearn.ensemble import RandomForestRegressor

file_handler = FileHandler()
Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

def rf_model(self):
  model = RandomForestRegressor(n_jobs=-1, n_estimators=15)
  return model
