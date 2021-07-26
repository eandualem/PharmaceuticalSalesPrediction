import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join('../')))
from log import get_logger

my_logger = get_logger("DfHelper")


class DvcHelper():

  def __init__(self):
    pass

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
