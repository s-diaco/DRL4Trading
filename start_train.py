from backtest.backtest import backtest_trades
import functools
import logging

import tensorflow as tf
from absl import app
from absl import logging as absl_logging
from tf_agents.system import system_multiprocessing as multiprocessing

from config import settings
from envirement.trading_py_env import TradingPyEnv
from models.model_ppo import TradeDRLAgent
from preprocess_data import preprocess_data

# from backtest.backtest import backtest_trades


def main(_):
    ticker_list = settings.DOW_30_TICKER
    data_columns = settings.DATA_COLUMNS

    # Preprocess data
    df_train = preprocess_data.preprocess_data(
        tic_list=ticker_list,
        start_date=settings.START_TRAIN_DATE,
        end_date=settings.END_TRAIN_DATE,
        field_mappings=settings.CSV_FIELD_MAPPINGS,
        baseline_filed_mappings=settings.BASELINE_FIELD_MAPPINGS,
        csv_file_info=settings.CSV_FILE_SETTINGS,
        user_columns=settings.USER_DEFINED_FEATURES
    )
    information_cols = []
    unavailable_cols = []
    for col in data_columns:
        if col in df_train.columns:
            information_cols.append(col)
        else:
            unavailable_cols.append(col)
    if not information_cols:
        logging.error('No column to train')
        raise ValueError
    else:
        logging.info(f'Columns used to train:\n{information_cols} ✅')
        if unavailable_cols:
            logging.info(f'Unavailable columns:\n{unavailable_cols} ❌')

    # df_train[information_cols].to_csv("temp.csv", index=1, encoding="utf-8")

    # Create the envoriments
    logging.info(f'TensorFlow v{tf.version.VERSION}')
    logging.info(
        f"Available [GPU] devices:\n{tf.config.list_physical_devices('GPU')}")

    class TrainEvalPyEnv(TradingPyEnv):
        def __init__(self):
            super().__init__(
                df=df_train,
                daily_information_cols=information_cols,
            )

    # Train
    model = TradeDRLAgent()
    model.train_PPO(
        py_env=TrainEvalPyEnv,
        collect_episodes_per_iteration=settings.NUM_EPISODES_PER_ITER,
        policy_checkpoint_interval=settings.POLIICY_CHKPT_INTERVAL,
        num_iterations=settings.NUM_ITERS,
        num_parallel_environments=settings.N_PARALLEL_CALLS,
    )

    # Predict
    
    # Preprocess prediction data
    df_trade = preprocess_data.preprocess_data(
        tic_list=ticker_list,
        start_date=settings.START_TRADE_DATE,
        end_date=settings.END_TRADE_DATE,
        field_mappings=settings.CSV_FIELD_MAPPINGS,
        baseline_filed_mappings=settings.BASELINE_FIELD_MAPPINGS,
        csv_file_info=settings.CSV_FILE_SETTINGS,
        user_columns=settings.USER_DEFINED_FEATURES
    )
    test_py_env = TradingPyEnv(
                # TODO should change to df_trade
                df=df_trade,
                daily_information_cols=information_cols,
    )
    df_account_value, _ = model.test_trade(env=test_py_env)

    baseline_df = preprocess_data.get_baseline_df(
        start_date=settings.START_TRADE_DATE,
        end_date=settings.END_TRADE_DATE,
        baseline_filed_mappings=settings.BASELINE_FIELD_MAPPINGS,
        csv_file_info=settings.CSV_FILE_SETTINGS
    ).reset_index()

    # Backtest stats & plots
    backtest_trades(df_account_value,
                    baseline_df)


if __name__ == '__main__':
    # FMT = '[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s'
    FMT = '[%(levelname)s] %(message)s'
    formatter = logging.Formatter(FMT)

    absl_logging.get_absl_handler().setFormatter(formatter)
    absl_logging.set_verbosity('info')
    # logging.basicConfig(format='%(message)s', level=logging.INFO)
    multiprocessing.handle_main(functools.partial(app.run, main))
