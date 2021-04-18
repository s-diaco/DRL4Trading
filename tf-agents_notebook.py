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
# %%
import os
from absl import app
from absl import flags
from absl import logging
from pprint import pprint
from finrl.model.models import DRLAgent
from env_tse.env_stocktrading_tse_stoploss import StockTradingEnvTSEStopLoss
from config import config
import datetime
import numpy as np
import pandas as pd
from IPython import get_ipython
import matplotlib
import matplotlib.pyplot as plt
import backtest_tse.backtesting_tse as backtest
from preprocess_tse_data import preprocess_data
import tf_agents
import tensorflow as tf
from tf_agents.drivers import dynamic_episode_driver; # data collection driver
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from tf_agents.environments.suite_gym import wrap_env
from tf_agents.environments import utils
from model.models import TradeDRLAgent

# logging.basicConfig(format="%(message)s", level=logging.INFO)
logging.set_verbosity(logging.INFO)
tf.compat.v1.enable_v2_behavior()
# %%
## Preprocess data
train, trade = preprocess_data()

# %%
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

# %%
logging.info(f'TensorFlow version: {tf.version.VERSION}')
logging.info(f"List of available [GPU] devices:\n{tf.config.list_physical_devices('GPU')}")

# parameters

# flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'), 'Root directory for writing logs/summaries/checkpoints.')
# flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Name of an environment')
flags.DEFINE_integer('replay_buffer_capacity', 1001,
                     'Replay buffer capacity per env.')
flags.DEFINE_integer('num_parallel_environments', 30,
                     'Number of environments to run in parallel')
flags.DEFINE_integer('num_environment_steps', 25000000,
                     'Number of environment steps to run before finishing.')
flags.DEFINE_integer('num_epochs', 25,
                     'Number of epochs for computing policy updates.')
flags.DEFINE_integer(
    'collect_episodes_per_iteration', 30,
    'The number of episodes to take in the environment before '
    'each update. This is the total across all parallel '
    'environments.')
flags.DEFINE_integer('num_eval_episodes', 30,
                     'The number of episodes to run eval on.')
flags.DEFINE_boolean('use_rnns', False,
                     'If true, use RNN for policy and value function.')
FLAGS = flags.FLAGS

train_eval_py_env = wrap_env(e_train_gym)
trade_py_env = wrap_env(e_trade_gym)
eval_tf_env = tf_py_environment.TFPyEnvironment(train_eval_py_env)
trade_tf_env = tf_py_environment.TFPyEnvironment(trade_py_env)
# %%
## Agent
tf_agent = TradeDRLAgent.get_agent(
        train_eval_py_env=eval_tf_env,
        )

# %%
## Train
TradeDRLAgent.train_eval(
    root_dir="./" + config.TRAINED_MODEL_DIR,
    train_eval_py_env=eval_tf_env,
    tf_agent=tf_agent,
    use_rnns=False,
    num_environment_steps=50,
    collect_episodes_per_iteration=30,
    # num_parallel_environments=1,
    replay_buffer_capacity=1001,
    num_epochs=25,
    num_eval_episodes=30
    )

# %%
## Predict
df_account_value, df_actions = TradeDRLAgent.predict_trades(trade_tf_env, trade_py_env)





# %% Trade
logging.info(f"Trade dates: {len(e_trade_gym.dates)}")

# %%
df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=model, environment=e_trade_gym
)

# %%
logging.info(f"Model actions:\n{df_actions.head()}")

# %%
logging.info(f"Account value data shape: {df_account_value.shape}:\n{df_account_value.head(10)}")

# %% 7. Backtest
backtest.backtest_tse_trades(df_account_value, "^TSEI", config.START_TRADE_DATE, config.END_DATE)
# %%
