import configparser
import os
import pathlib as pathlb
import time

import numpy as np
import pandas as pd

# preprocessor
import preprocessing.preprocessors as preproc
import logging


logging.basicConfig(level=logging.INFO)


def run_model() -> None:
    """Train the model."""
    # read and preprocess data
    config = configparser.ConfigParser()
    config.read('config.ini')
    preprocessed_file = pathlb.Path(
        config['csv']['csv_dir']) / pathlb.Path(config['csv']['preprocessed_data'])

    logging.info("Looking for data...")

    if preprocessed_file.is_file():
        logging.info("Preprocessed data exists.")
        data = pd.read_csv(preprocessed_file, index_col=0)
    else:
        logging.info("Preprocessing data...")
        data = preproc.preprocess_data()
        logging.info("Calculating turbulence...")
        data = preproc.add_turbulence(data)
        logging.info("Saving preprocessed data.")
        data.to_csv(preprocessed_file)

    print(data.head())
    print(data.size)

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    unique_trade_date = data[(data.datadate > int(config['dates']['validation_start_date'])) & (
        data.datadate <= int(config['dates']['trade_end_date']))].datadate.unique()
    print(unique_trade_date)

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    rebalance_window = 63
    validation_window = 63

    # Ensemble Strategy
    # run_ensemble_strategy(df=data,unique_trade_date=unique_trade_date,rebalance_window=rebalance_window,validation_window=validation_window)

    #_logger.info(f"saving model version: {_version}")


if __name__ == "__main__":
    logging.info("DRL start.")
    run_model()
