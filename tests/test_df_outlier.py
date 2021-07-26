import os
import sys
import unittest
import pandas as pd


sys.path.append(os.path.abspath(os.path.join('../scripts')))
from df_outlier import DfOutlier


class TestDfOutlier(unittest.TestCase):

  def setUp(self) -> pd.DataFrame:
    self.cleaner = DfOutlier()


if __name__ == '__main__':
  unittest.main()
