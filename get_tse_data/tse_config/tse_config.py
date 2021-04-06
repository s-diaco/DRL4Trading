import pathlib

# import finrl

import pandas as pd
import datetime
import os

# pd.options.display.max_rows = 10
# pd.options.display.max_columns = 10


# PACKAGE_ROOT = pathlib.Path(finrl.__file__).resolve().parent
# PACKAGE_ROOT = pathlib.Path().resolve().parent

# DATASET_DIR = PACKAGE_ROOT / "data"

# data
# TRAINING_DATA_FILE = "data/ETF_SPY_2009_2020.csv"
# TURBULENCE_DATA = "data/dow30_turbulence_index.csv"
# TESTING_DATA_FILE = "test.csv"

# now = datetime.datetime.now()
# TRAINED_MODEL_DIR = f"trained_models/{now}"
DATA_SAVE_DIR = f"datasets"
TRAINED_MODEL_DIR = f"trained_models"
TENSORBOARD_LOG_DIR = f"tensorboard_log"
RESULTS_DIR = f"results"
# os.makedirs(TRAINED_MODEL_DIR)


## time_fmt = '%Y-%m-%d'
START_DATE = "2000-01-03"
END_DATE = "2021-01-07"

START_TRADE_DATE = "2019-01-01"

IN_DIR = 'tickers_data'
CSV_DIR = 'csv_data'
EXP_FILE_NAME = 'combined_csv.csv'
TSEI = 'TSEI_20081206_20210405.csv'
