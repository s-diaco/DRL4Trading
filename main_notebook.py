# %% [markdown]
#### todo:
# - what are the training warnings about
# - adjust total assets on division and buybacks
# - devide main notebook to multiple smaller files
# - get index info from tsetmc.com
# %%
import os
from pprint import pprint
from finrl.trade.backtest import backtest_stats
from finrl.model.models import DRLAgent
from env_tse.env_stocktrading_tse_stoploss import StockTradingEnvTSEStopLoss
from finrl.preprocessing.data import data_split
from finrl.preprocessing.preprocessors import FeatureEngineer
from config import config
import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from IPython import get_ipython
import tse_backtest_plot.tse_backtest_plot as bt_plt
import logging
from preprocess_tse_data import preprocess_data

logging.basicConfig(format="%(message)s", level=logging.INFO)

train, trade = preprocess_data()

# %%
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
    cash_penalty_proportion=0,
    patient=True,
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

# %% Environment for Training
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
    tb_log_name="env_stoploss_highlr",
    n_eval_episodes=1,
)

# %%
# model.save("trained_models/different4_7_10000.model")

# %% Trade
logging.info(f"Trade dates: {len(e_trade_gym.dates)}")

# %%
df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=model, environment=e_trade_gym
)

# %%
df_actions.head()

# %%
df_account_value.shape

# %%
df_account_value.head(50)

# %% 7. Backtest
# ## 7.1 Backtest Stats
print("==============Backtest Results===========")
perf_stats_all = backtest_stats(
    account_value=df_account_value, value_col_name="total_assets"
)

# %%
# ## 7.2 Backtest Plot
print("==============Compare to baseline===========")
get_ipython().run_line_magic("matplotlib", "inline")

bt_plt.backtest_plot(
    df_account_value,
    baseline_ticker="^TSEI",
    baseline_start=config.START_TRADE_DATE,
    baseline_end=config.END_DATE,
    value_col_name="total_assets",
)

# %%
