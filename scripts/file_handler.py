import pickle
# import dvc.api
import pandas as pd
from config import Config
from log import get_logger
from time import gmtime, strftime

my_logger = get_logger("DfHelper")


class FileHandler():

  def __init__(self):
    pass

  # def get_dvc_file(self, file_path, tag):
  #   try:
  #     data_url = dvc.api.get_url(path=str(file_path), repo=str(Config.REPO), rev=tag)
  #     df = pd.read_csv(data_url, sep=',')
  #     my_logger.info("Load data from dvc")
  #     return df

  #   except Exception:
  #     my_logger.exception("Error loading file from dvc")

  def save_csv(self, df, csv_path, index=False):
    try:
      df.to_csv(csv_path, index=index)
      my_logger.info("file saved as csv")

    except Exception:
      my_logger.exception("save failed")

  def read_csv(self, csv_path, missing_values=[]):
    try:
      df = pd.read_csv(csv_path, na_values=missing_values)
      my_logger.debug("file read as csv")
      return df
    except FileNotFoundError:
      my_logger.exception("file not found")

  def save_model(self, model, model_name):
    try:
      time = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
      name = Config.MODELS_PATH / str(model_name + "-" + time + ".pkl")
      pickle.dump(model, open(str(name), "wb"))
      my_logger.info("file saved as csv")

    except Exception:
      my_logger.exception("save failed")

  def read_model(self, model_name):
    try:
      name = Config.MODELS_PATH / str(model_name + ".pkl")
      model = pickle.load(open(name, "rb"))
      my_logger.debug("model read as pkl")
      return model
    except FileNotFoundError:
      my_logger.exception("model not found")
