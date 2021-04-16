# %% [markdown]
#### todo:
# - devide main notebook to multiple smaller files
# - develope a better logging system
# - fix parallel envoriments
# - use correct policy batch size for ppo
# - implement gym env in python
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

# %% Agent
logging.info(f'TensorFlow version: {tf.version.VERSION}')
logging.info(f"List of available [GPU] devices:\n{tf.config.list_physical_devices('GPU')}")

# model parameters

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

TradeDRLAgent.train_eval(
    root_dir="./" + config.TRAINED_MODEL_DIR,
    train_eval_py_env=train_eval_py_env,
    use_rnns=False,
    num_environment_steps=100,
    collect_episodes_per_iteration=30,
    # num_parallel_environments=1,
    replay_buffer_capacity=1001,
    num_epochs=25,
    num_eval_episodes=30
    )

# %%
## Predict
train_eval_py_env = wrap_env(e_trade_gym)
TradeDRLAgent.predict_trades("model", e_trade_gym)






# %% 
## plot
steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns.result().numpy())
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim(top=250)

# %%
## predict
num_episodes = 3
for _ in range(num_episodes):
  status = eval_env.reset() #time_step
  policy_state = tf_agent.policy.get_initial_state(eval_env.batch_size)
  while not status.is_last():
    # todo: use greedy policy to test
    action = tf_agent.policy.action(status, policy_state)
    status = eval_env.step(action.action)
    logging.info(f'Action: {action}')





# %%
# single processing
env_train, _ = e_train_gym.get_sb_env()

# this is our observation environment. It allows full diagnostics
env_trade, _ = e_trade_gym.get_sb_env()

# %% 6. Implement DRL Algorithms
agent = DRLAgent(env=env_train)

# %% Model PPO
# from torch.nn import Softsign, ReLU
ppo_params = {
    "n_steps": 256,
    "ent_coef": 0.0,
    "learning_rate": 0.000005,
    "batch_size": 1024,
    "gamma": 0.99,
}

policy_kwargs = {
    #     "activation_fn": ReLU,
    "net_arch": [1024 for _ in range(10)],
    #     "squash_output": True
}

model = agent.get_model(
    "ppo", model_kwargs=ppo_params, policy_kwargs=policy_kwargs, verbose=0
)

# %%
# model = model.load("trained_models/different4_7_2000.model", env = env_train)

# %%
model.learn(
    total_timesteps=10000,
    eval_env=env_trade,
    eval_freq=500,
    log_interval=1,
    tb_log_name="env_tse",
    n_eval_episodes=1,
)

# %%
# model.save("trained_models/tse4_10_1000.model")

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
