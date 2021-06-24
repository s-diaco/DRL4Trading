"""
predict trades for one specific day
"""
import functools
import logging
from datetime import datetime, timedelta
import pandas as pd

import tensorflow as tf
from absl import app
from absl import logging as absl_logging
from tf_agents.system import system_multiprocessing as multiprocessing

from config import settings
from envirement.trading_py_env import TradingPyEnv
from models.model_ppo import TradeDRLAgent
from preprocess_data import preprocess_data


def main(_):
    PREDICT_DAY = "2020-06-01"
    days_to_subtract = 60
    ticker_list = settings.DOW_30_TICKER
    data_columns = settings.DATA_COLUMNS

    # Preprocess data
    df_trade = preprocess_data.preprocess_data(
        tic_list=ticker_list,
        start_date=str(datetime.strptime(PREDICT_DAY, '%Y-%m-%d')
                       - timedelta(days=days_to_subtract)),
        end_date=PREDICT_DAY,
        field_mappings=settings.CSV_FIELD_MAPPINGS,
        baseline_filed_mappings=settings.BASELINE_FIELD_MAPPINGS,
        csv_file_info=settings.CSV_FILE_SETTINGS
    )
    information_cols = []
    for col in data_columns:
        if col in df_trade.columns:
            information_cols.append(col)
        else:
            logging.info(f'column {col} not in the train data. skipped')
    if not information_cols:
        logging.error('No column to train')
        raise ValueError
    else:
        logging.info(f'Columns used to predict: \n{information_cols}')

    # df_trade[information_cols].to_csv("temp.csv", index=1, encoding="utf-8")

    logging.info(f'TensorFlow v{tf.version.VERSION}')
    logging.info(
        f"Available [GPU] devices:\n{tf.config.list_physical_devices('GPU')}")

    # Predict
    test_py_env = TradingPyEnv(
        df=df_trade,
        daily_information_cols=information_cols,
    )
    model = TradeDRLAgent()
    _, df_actions = model.test_trade(env=test_py_env)
    assert len(df_trade.tic.unique()) == len(
        df_actions.tail(1).transactions.values[0])
    pred_inf_df = pd.DataFrame(
        {'ticker': df_trade.tic.unique()}
    )
    pred_inf_df['trade'] = pd.Series(df_actions.tail(1).transactions.values[0])
    last_day = pd.to_datetime(str(df_actions.tail(1).date.values[0]))
    last_day_str = last_day.strftime("%B %d, %Y")
    logging.info(f'\nPredicted trades for {last_day_str}:\n{pred_inf_df}')


if __name__ == '__main__':
    # FMT = '[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s'
    FMT = '[%(levelname)s] %(message)s'
    formatter = logging.Formatter(FMT)

    absl_logging.get_absl_handler().setFormatter(formatter)
    absl_logging.set_verbosity('info')
    # logging.basicConfig(format='%(message)s', level=logging.INFO)
    multiprocessing.handle_main(functools.partial(app.run, main))
