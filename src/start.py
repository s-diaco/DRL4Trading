import functools
import logging
from absl import app

import tensorflow as tf
from tf_agents.system import system_multiprocessing as multiprocessing
from backtest.backtest import backtest_trades

from config import settings
from envirement.trading_py_env import TradingPyEnv
from models.model_ppo import TradeDRLAgent
from preprocess_data import preprocess_data


def main(_):
    ticker_list = settings.DOW_30_TICKER
    data_columns = settings.DATA_COLUMNS

    # Preprocess data
    df_train = preprocess_data.preprocess_data(
        tic_list=ticker_list,
        start_date=settings.START_DATE,
        end_date=settings.END_DATE,
        field_mappings=settings.CSV_FIELD_MAPPINGS,
        baseline_filed_mappings=settings.BASELINE_FIELD_MAPPINGS,
        csv_file_info=settings.CSV_FILE_SETTINGS,
        tec_indicators=settings.TECHNICAL_INDICATORS_LIST
    )
    information_cols = data_columns
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
    test_py_env = TradingPyEnv(
                # TODO should change to df_trade
                df=df_train,
                daily_information_cols=information_cols,
    )
    df_account_value, _ = model.test_trade(env=test_py_env)

    # Backtest stats & plots
    backtest_trades(df_account_value,
                    settings.CSV_FILE_SETTINGS["baseline_file_name"],
                    settings.START_TRADE_DATE, settings.END_DATE)


if __name__ == '__main__':
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if settings.N_PARALLEL_CALLS > 1:
        multiprocessing.handle_main(functools.partial(app.run, main))


# TODO:
# - automatic optimization of hyperparameters
# - implement handling of variable episode lenghs
# - use other agents
# - implement single-day predict function
# - why is average episode lenght not constant?
