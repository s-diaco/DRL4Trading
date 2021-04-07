# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
from pprint import pprint
from finrl.trade.backtest import backtest_stats
from finrl.model.models import DRLAgent
from finrl.env.env_stocktrading_stoploss import StockTradingEnvStopLoss
from finrl.preprocessing.data import data_split
from finrl.preprocessing.preprocessors import FeatureEngineer
from config import config
import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from IPython import get_ipython
from get_tse_data.tse_data import tse_data
import tse_backtest_plot.tse_backtest_plot as bt_plt
import logging

logging.basicConfig(
    format='%(asctime)s - log - %(message)s', level=logging.INFO)

print(pd.__version__)

# %%
# matplotlib.use('Agg')
# get_ipython().run_line_magic('matplotlib', 'inline')
# %%
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)

# %%
# from config.py start_date is a string
logging.info(f'Start date: {config.START_DATE}')
# from config.py end_date is a string
logging.info(f'End date: {config.END_DATE}')
logging.info(f'Tickers: {config.TSE_TICKER_30}')
df = tse_data(start_date=config.START_DATE,
              end_date=config.END_DATE,
              ticker_list=config.TSE_TICKER_30).fetch_data()

# %% 4.Preprocess Data
fe = FeatureEngineer(
    use_technical_indicator=False,
    tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
    use_turbulence=False,
    user_defined_feature=False)

processed = fe.preprocess_data(df)

# %%
processed = df
processed['log_volume'] = np.log(processed.volume*processed.close)
processed['change'] = (processed.close-processed.open)/processed.close
processed['daily_variance'] = (processed.high-processed.low)/processed.close
processed=processed.fillna(0)
processed.head()

# %% 5.Design Environment
train = data_split(processed, config.START_DATE, config.START_TRADE_DATE)
trade = data_split(processed, config.START_TRADE_DATE, config.END_DATE)
logging.info(f'Training sample size: {len(train)}')
logging.info(f'Trading sample size: {len(trade)}')

# %%
processed.head()

# %%
information_cols = ['daily_variance', 'change']

e_train_gym = StockTradingEnvStopLoss(df=train, initial_amount=1e8, hmax=1e7,
                                         cache_indicator_data=False,
                                         daily_information_cols=information_cols,
                                         print_verbosity=500,
                                         patient=True)

e_trade_gym = StockTradingEnvStopLoss(df=trade, initial_amount=1e8, hmax=1e7,
                                         cache_indicator_data=False,
                                         daily_information_cols=information_cols,
                                         print_verbosity=500,
                                         discrete_actions=True,
                                         shares_increment=10,
                                         patient=True,
                                         random_start=False)

# %% Environment for Training
# single processing
env_train, _ = e_train_gym.get_sb_env()

# this is our observation environment. It allows full diagnostics
env_trade, _ = e_trade_gym.get_sb_env()

# %% 6.Implement DRL Algorithms
agent = DRLAgent(env=env_train)

# %% Model PPO
# from torch.nn import Softsign, ReLU
ppo_params = {'n_steps': 256,
              'ent_coef': 0.0,
              'learning_rate': 0.000005,
              'batch_size': 1024,
              'gamma': 0.99}

policy_kwargs = {
    #     "activation_fn": ReLU,
    "net_arch": [1024 for _ in range(10)],
    #     "squash_output": True
}

model = agent.get_model("ppo",
                        model_kwargs=ppo_params,
                        policy_kwargs=policy_kwargs, verbose=0)

# %%
model = model.load("trained_models/different4_7_2000.model", env = env_train)

# %%
model.learn(total_timesteps=10000,
            eval_env=env_trade,
            eval_freq=500,
            log_interval=1,
            tb_log_name='env_stoploss_highlr',
            n_eval_episodes=1)

# %%
model.save("trained_models/different4_7_10000.model")

# %% Trade
trade.head()

# %%
print(len(e_trade_gym.dates))

# %%
df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=model, environment=e_trade_gym)

# %%
df_actions.head()

# %%
df_account_value.shape

# %%
df_account_value.head(50)

# %% 7.Backtest
# ## 7.1 Backtest Stats
print("==============Backtest Results===========")
perf_stats_all = backtest_stats(
    account_value=df_account_value, value_col_name='total_assets')

# %%
# ## 7.2 Backtest Plot
print("==============Compare to baseline===========")
get_ipython().run_line_magic('matplotlib', 'inline')

bt_plt.backtest_plot(df_account_value,
                     baseline_ticker='^TSEI',
                     baseline_start=config.START_TRADE_DATE,
                     baseline_end=config.END_DATE,
                     value_col_name='total_assets')

# %%
