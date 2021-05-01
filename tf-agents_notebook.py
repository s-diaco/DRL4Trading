# %% [markdown]
#### todo:
# - devide main notebook to multiple smaller files
# - develope a better logging system
# - fix parallel envoriments
# - use correct policy batch size for ppo
# - implement gym env in python
# - use drivers and replay buffer for predictions
# - use greedy policy to test (what is "eager mode"?)
# - policy_000000000 dir
# - dont use gather_all
# - organize folders created by modules
# - use original network numbers
# - replace print with logging is py_env
# - what is random_seed in 2 files
# - change num_parallel_environments
# - separate get_agent() from model train
# %% [markdown]
## import modules
import datetime
import logging
from pprint import pprint

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_agents
from IPython import get_ipython
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.environments.suite_gym import wrap_env
from tf_agents.environments import utils

import backtest_tse.backtesting_tse as backtest
from config import config
from env_tse.env_stocktrading_tse_stoploss import StockTradingEnvTSEStopLoss
from env_tse.py_env_trading import TradingPyEnv
# from model.models import TradeDRLAgent
from model.models_single_process import TradeDRLAgent
from preprocess_tse_data import preprocess_data

logging.basicConfig(format="%(message)s", level=logging.INFO)

# %% [markdown]
## Preprocess data
df_train, df_trade = preprocess_data()

# %% [markdown]
## Create the envoriments
information_cols = ["daily_variance", "change", "log_volume"]

logging.info(f'TensorFlow version: {tf.version.VERSION}')
logging.info(f"List of available [GPU] devices:\n{tf.config.list_physical_devices('GPU')}")


class TrainEvalPyEnv(TradingPyEnv):
    def __init__(self):
        super().__init__(
            df=df_train,
            daily_information_cols=information_cols,
            cache_indicator_data=False #todo: delete if needed,
            )


class TestPyEnv(TradingPyEnv):
    def __init__(self):
        super().__init__(
            df=df_trade,
            daily_information_cols=information_cols,
            cache_indicator_data=False,
            discrete_actions=True,
            shares_increment=10,
            patient=True,
            random_start=False,)

# %% todo: delete - test the envirement
# environment = TestPyEnv()
# utils.validate_py_environment(environment, episodes=2)

# %% [markdown]
## Agent
# todo: delete - tf_agent = TradeDRLAgent().get_agent(
#        train_eval_py_env=TrainEvalPyEnv,
#        )

# %% [markdown]
## Train
# tf_agents.system.multiprocessing.enable_interactive_mode()

# %%
TradeDRLAgent().train_eval(
    root_dir="./" + config.TRAINED_MODEL_DIR,
    py_env=TrainEvalPyEnv,
    # tf_agent=tf_agent,
    use_rnns=False,
    num_environment_steps=70,
    collect_episodes_per_iteration=30,
    num_parallel_environments=1,
    replay_buffer_capacity=1001,
    num_epochs=25,
    num_eval_episodes=30
    )

# %% [markdown]
## Predict
df_account_value, df_actions = TradeDRLAgent.predict_trades(TestPyEnv)

# %% [markdown]
## Trade info
logging.info(f"Model actions:\n{df_actions.head()}")
logging.info(f"Account value data shape: {df_account_value.shape}:\n{df_account_value.head(10)}")

# %% [markdown]
## Backtest stats & plots
backtest.backtest_tse_trades(df_account_value, "^TSEI", config.START_TRADE_DATE, config.END_DATE)
# %%