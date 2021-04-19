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

import backtest_tse.backtesting_tse as backtest
from config import config
from env_tse.env_stocktrading_tse_stoploss import StockTradingEnvTSEStopLoss
from env_tse.py_env_trading import TradingPyEnv
from model.models import TradeDRLAgent
from preprocess_tse_data import preprocess_data

logging.basicConfig(format="%(message)s", level=logging.INFO)
tf.compat.v1.enable_v2_behavior()

# %% [markdown]
## Preprocess data
train, trade = preprocess_data()

# %% [markdown]
## Create the envoriments
information_cols = ["daily_variance", "change", "log_volume"]

e_train_gym = StockTradingEnvTSEStopLoss(
    df=train,
    initial_amount=1e8,
    hmax=1e7,
    cache_indicator_data=False,
    daily_information_cols=information_cols,
    print_verbosity=500,
    buy_cost_pct=3.7e-3,
    sell_cost_pct=8.8e-3,
    cash_penalty_proportion=0
)

e_trade_gym = StockTradingEnvTSEStopLoss(
    df=trade,
    initial_amount=1e8,
    hmax=1e7,
    cache_indicator_data=False,
    daily_information_cols=information_cols,
    print_verbosity=500,
    discrete_actions=True,
    shares_increment=10,
    buy_cost_pct=3.7e-3,
    sell_cost_pct=8.8e-3,
    cash_penalty_proportion=0,
    patient=True,
    random_start=False,
)

logging.info(f'TensorFlow version: {tf.version.VERSION}')
logging.info(f"List of available [GPU] devices:\n{tf.config.list_physical_devices('GPU')}")

train_eval_py_env = wrap_env(e_train_gym)
trade_py_env = wrap_env(e_trade_gym)
train_eval_tf_env = tf_py_environment.TFPyEnvironment(train_eval_py_env)
trade_tf_env = tf_py_environment.TFPyEnvironment(trade_py_env)

# %% [markdown]
## Agent
tf_agent = TradeDRLAgent().get_agent(
        train_eval_tf_env=train_eval_tf_env,
        )

# %% [markdown]
## Train
TradeDRLAgent().train_eval(
    root_dir="./" + config.TRAINED_MODEL_DIR,
    train_eval_tf_env=train_eval_tf_env,
    tf_agent=tf_agent,
    use_rnns=False,
    num_environment_steps=50,
    collect_episodes_per_iteration=30,
    # num_parallel_environments=1,
    replay_buffer_capacity=1001,
    num_epochs=25,
    num_eval_episodes=30
    )

# %% [markdown]
## Predict
df_account_value, df_actions = TradeDRLAgent.predict_trades(trade_tf_env, trade_py_env)

# %% [markdown]
## Trade info
logging.info(f"Trade dates: {len(trade_py_env.dates)}")
logging.info(f"Model actions:\n{df_actions.head()}")
logging.info(f"Account value data shape: {df_account_value.shape}:\n{df_account_value.head(10)}")

# %% [markdown]
## Backtest stats & plots
backtest.backtest_tse_trades(df_account_value, "^TSEI", config.START_TRADE_DATE, config.END_DATE)
# %%
