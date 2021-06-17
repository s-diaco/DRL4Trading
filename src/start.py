## import modules
import functools
import logging
from absl import app

import tensorflow as tf
from tf_agents.system import system_multiprocessing as multiprocessing

import backtest_tse.backtesting_tse as backtest
from config import config
from env_tse.py_env_trading import TradingPyEnv
from model.models import TradeDRLAgent
from preprocess_tse_data import preprocess_data


def main(_):
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    TICKER_LIST = config.TSE_TICKER_5
    DATA_COLUMNS = config.DATA_COLUMNS

    # Preprocess data
    df_train, df_trade = preprocess_data(
        tic_list=TICKER_LIST
    )
    information_cols = DATA_COLUMNS
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
                patient=True,
                cash_penalty_proportion=0,
                single_stock_action=config.DISCRETE_ACTION_SPACE,
            )

    # Train
    model = TradeDRLAgent()
    model.train_PPO(
        py_env=TrainEvalPyEnv,
        collect_episodes_per_iteration=config.NUM_EPISODES_PER_ITER,
        policy_checkpoint_interval=config.POLIICY_CHKPT_INTERVAL,
        num_iterations=config.NUM_ITERS,
        num_parallel_environments=config.N_PARALLEL_CALLS,
    )

    # Predict
    test_py_env = TradingPyEnv(
                df=df_trade,
                daily_information_cols=information_cols,
                patient=True,
                cash_penalty_proportion=0,
                single_stock_action=config.DISCRETE_ACTION_SPACE
    )
    df_account_value, _ = model.test_trade(env=test_py_env)

    # Backtest stats & plots
    backtest.backtest_tse_trades(
        df_account_value, "^TSEI", config.START_TRADE_DATE, config.END_DATE)


if __name__ == '__main__':
    if config.N_PARALLEL_CALLS > 1:
        multiprocessing.handle_main(functools.partial(app.run, main))


# TODO:
# - auto detect if file needs to redownload
# - set source dir to python path to access parent dirs
# - automatic optimization of hyperparameters
# - implement handling of variable episode lenghs
# - use other agents
# - implement single-day predict function
# - add option to add all csv files from a folder
# - why is average episode lenght not constant?
