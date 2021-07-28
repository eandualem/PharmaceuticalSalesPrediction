from pathlib import Path


class Config:
  RANDOM_SEED = 42
  ASSETS_PATH = Path("../")
  REPO = "/Users/ea/Projects/PharmaceuticalSalesPrediction"
  STORE_PATH = "../data/clean_store.csv"
  TRAIN_PATH = "../data/clean_train.csv"
  TEST_PATH = "../data/clean_test.csv"
  DATASET_PATH = ASSETS_PATH / "data"
  FEATURES_PATH = ASSETS_PATH / "features"
  MODELS_PATH = ASSETS_PATH / "models"
  METRICS_FILE_PATH = ASSETS_PATH / "metrics"
