# %% [markdown]
# todo:
# - auto detect if file needs to redownload
# - stet source dir to python path so i can access parent dirs
# - automatic optimization of hyperparameters

# %% [markdown]
## import modules
import logging
import tensorflow as tf
from IPython import get_ipython

import tf_agents.system
import backtest_tse.backtesting_tse as backtest
from config import config
from env_tse.py_env_trading import TradingPyEnv
from model.models import TradeDRLAgent
from preprocess_tse_data import preprocess_data


logging.basicConfig(format="%(message)s", level=logging.INFO)

N_PARALLEL_CALLS = 4
NUM_EPISODES_PER_ITER = 2
POLIICY_CHKPT_INTERVAL = 5
NUM_ITERS = 500

# %% [markdown]
# Preprocess data
df_train, df_trade = preprocess_data()

# %% [markdown]
# Create the envoriments
information_cols = config.DATA_COLUMNS

logging.info(f'TensorFlow version: {tf.version.VERSION}')
logging.info(
    f"List of available [GPU] devices:\n{tf.config.list_physical_devices('GPU')}")


class TrainEvalPyEnv(TradingPyEnv):
    def __init__(self):
        super().__init__(
            df=df_train,
            daily_information_cols=information_cols,
            patient=True,
            random_start=False,
            cash_penalty_proportion=0,
            single_stock_action=True,
        )


class TestPyEnv(TradingPyEnv):
    def __init__(self):
        super().__init__(
            df=df_trade,
            daily_information_cols=information_cols,
            patient=True,
            random_start=False,
            cash_penalty_proportion=0,
            single_stock_action=True
        )


# %% [markdown]
# Train
if N_PARALLEL_CALLS > 1:
    tf_agents.system.multiprocessing.enable_interactive_mode()

# %%
TradeDRLAgent().train_PPO(
    py_env=TrainEvalPyEnv,
    collect_episodes_per_iteration=NUM_EPISODES_PER_ITER,
    policy_checkpoint_interval=POLIICY_CHKPT_INTERVAL,
    num_iterations=NUM_ITERS,
    num_parallel_environments=N_PARALLEL_CALLS,
    use_parallel_envs=True
)

# %% [markdown]
# Predict
df_account_value, df_actions = TradeDRLAgent().predict_trades(py_test_env=TestPyEnv)

# %% [markdown]
# Trade info
logging.info(f"Model actions:\n{df_actions.head()}")
logging.info(
    f"Account value data shape: {df_account_value.shape}:\n{df_account_value.head(10)}")

# %% [markdown]
# Backtest stats & plots
backtest.backtest_tse_trades(
    df_account_value, "^TSEI", config.START_TRADE_DATE, config.END_DATE)
# %%
