# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
from pprint import pprint
from finrl.trade.backtest import backtest_plot, backtest_stats
from finrl.model.models import DRLAgent
from finrl.env.env_stocktrading_cashpenalty import StockTradingEnvCashpenalty
from finrl.preprocessing.data import data_split
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.config import config
import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from IPython import get_ipython
from get_tse_data.tse_data import tse_data

import logging

logging.basicConfig(
    format='%(asctime)s - log - %(message)s', level=logging.INFO)

# %% [markdown]
# <a href="https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/FinRL_multiple_stock_trading.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# %% [markdown]
# # Deep Reinforcement Learning for Stock Trading from Scratch: Multiple Stock Trading
#
# Tutorials to use OpenAI DRL to trade multiple stocks in one Jupyter Notebook | Presented at NeurIPS 2020: Deep RL Workshop
#
# * This blog is based on our paper: FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance, presented at NeurIPS 2020: Deep RL Workshop.
# * Check out medium blog for detailed explanations: https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-multiple-stock-trading-7b00763b7530
# * Please report any issues to our Github: https://github.com/AI4Finance-LLC/FinRL-Library/issues
# * **Pytorch Version**
#
#
# %% [markdown]
# # Content
# %% [markdown]
# * [1. Problem Definition](#0)
# * [2. Getting Started - Load Python packages](#1)
#     * [2.1. Install Packages](#1.1)
#     * [2.2. Check Additional Packages](#1.2)
#     * [2.3. Import Packages](#1.3)
#     * [2.4. Create Folders](#1.4)
# * [3. Download Data](#2)
# * [4. Preprocess Data](#3)
#     * [4.1. Technical Indicators](#3.1)
#     * [4.2. Perform Feature Engineering](#3.2)
# * [5.Build Environment](#4)
#     * [5.1. Training & Trade Data Split](#4.1)
#     * [5.2. User-defined Environment](#4.2)
#     * [5.3. Initialize Environment](#4.3)
# * [6.Implement DRL Algorithms](#5)
# * [7.Backtesting Performance](#6)
#     * [7.1. BackTestStats](#6.1)
#     * [7.2. BackTestPlot](#6.2)
#     * [7.3. Baseline Stats](#6.3)
#     * [7.3. Compare to Stock Market Index](#6.4)
# %% [markdown]
# <a id='0'></a>
# # Part 1. Problem Definition
# %% [markdown]
# This problem is to design an automated trading solution for single stock trading. We model the stock trading process as a Markov Decision Process (MDP). We then formulate our trading goal as a maximization problem.
#
# The algorithm is trained using Deep Reinforcement Learning (DRL) algorithms and the components of the reinforcement learning environment are:
#
#
# * Action: The action space describes the allowed actions that the agent interacts with the
# environment. Normally, a ∈ A includes three actions: a ∈ {−1, 0, 1}, where −1, 0, 1 represent
# selling, holding, and buying one stock. Also, an action can be carried upon multiple shares. We use
# an action space {−k, ..., −1, 0, 1, ..., k}, where k denotes the number of shares. For example, "Buy
# 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or −10, respectively
#
# * Reward function: r(s, a, s′) is the incentive mechanism for an agent to learn a better action. The change of the portfolio value when action a is taken at state s and arriving at new state s',  i.e., r(s, a, s′) = v′ − v, where v′ and v represent the portfolio
# values at state s′ and s, respectively
#
# * State: The state space describes the observations that the agent receives from the environment. Just as a human trader needs to analyze various information before executing a trade, so
# our trading agent observes many different features to better learn in an interactive environment.
#
# * Environment: Dow 30 consituents
#
#
# The data of the single stock that we will be using for this case study is obtained from Yahoo Finance API. The data contains Open-High-Low-Close price and volume.
#
# %% [markdown]
# <a id='1'></a>
# # Part 2. Getting Started- ASSUMES USING DOCKER, see readme for instructions
# %% [markdown]
# <a id='1.1'></a>
# ## 2.1. Add FinRL to your path. You can of course install it as a pipy package, but this is for development purposes.
#
# %%
print(pd.__version__)

# %% [markdown]
#
# <a id='1.2'></a>
# ## 2.2. Check if the additional packages needed are present, if not install them.
# * Yahoo Finance API
# * pandas
# * numpy
# * matplotlib
# * stockstats
# * OpenAI gym
# * stable-baselines
# * tensorflow
# * pyfolio
# %% [markdown]
# <a id='1.3'></a>
# ## 2.3. Import Packages

# %%
# matplotlib.use('Agg')

# get_ipython().run_line_magic('matplotlib', 'inline')


# %% [markdown]
# <a id='1.4'></a>
# ## 2.4. Create Folders

# %%
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)

# %% [markdown]
# <a id='2'></a>
# # Part 3. Download Data
# Yahoo Finance is a website that provides stock data, financial news, financial reports, etc. All the data provided by Yahoo Finance is free.
# * FinRL uses a class **YahooDownloader** to fetch data from Yahoo Finance API
# * Call Limit: Using the Public API (without authentication), you are limited to 2,000 requests per hour per IP (or up to a total of 48,000 requests a day).
#
# %% [markdown]
#
#
# -----
# class YahooDownloader:
#     Provides methods for retrieving daily stock data from
#     Yahoo Finance API
#
#     Attributes
#     ----------
#         start_date : str
#             start date of the data (modified from config.py)
#         end_date : str
#             end date of the data (modified from config.py)
#         ticker_list : list
#             a list of stock tickers (modified from config.py)
#
#     Methods
#     -------
#     fetch_data()
#         Fetches data from yahoo API
#

# %%
# from config.py start_date is a string
logging.info(f'Start date: {config.START_DATE}')
# from config.py end_date is a string
logging.info(f'End date: {config.END_DATE}')
logging.info(f'Tickers: {config.DOW_30_TICKER}')

# %%

df = tse_data(start_date='2018-01-01',
              end_date='2020-01-01',
              ticker_list=config.DOW_30_TICKER).combine_csv()

# %% [markdown]
# # Part 4: Preprocess Data
# Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.
# * Add technical indicators. In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc. In this article, we demonstrate two trend-following technical indicators: MACD and RSI.
# * Add turbulence index. Risk-aversion reflects whether an investor will choose to preserve the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the financial turbulence index that measures extreme asset price fluctuation.

# %%
fe = FeatureEngineer(
    use_technical_indicator=False,
    tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
    use_turbulence=False,
    user_defined_feature=False)

processed = fe.preprocess_data(df)


# %%
processed['log_volume'] = np.log(processed.volume*processed.close)
processed['change'] = (processed.close-processed.open)/processed.close
processed['daily_variance'] = (processed.high-processed.low)/processed.close
processed.head()

# %% [markdown]
# <a id='4'></a>
# # Part 5. Design Environment
# Considering the stochastic and interactive nature of the automated stock trading tasks, a financial task is modeled as a **Markov Decision Process (MDP)** problem. The training process involves observing stock price change, taking an action and reward's calculation to have the agent adjusting its strategy accordingly. By interacting with the environment, the trading agent will derive a trading strategy with the maximized rewards as time proceeds.
#
# Our trading environments, based on OpenAI Gym framework, simulate live stock markets with real market data according to the principle of time-driven simulation.
#
# The action space describes the allowed actions that the agent interacts with the environment. Normally, action a includes three actions: {-1, 0, 1}, where -1, 0, 1 represent selling, holding, and buying one share. Also, an action can be carried upon multiple shares. We use an action space {-k,…,-1, 0, 1, …, k}, where k denotes the number of shares to buy and -k denotes the number of shares to sell. For example, "Buy 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or -10, respectively. The continuous action space needs to be normalized to [-1, 1], since the policy is defined on a Gaussian distribution, which needs to be normalized and symmetric.
# %% [markdown]
# ## Training data split: 2009-01-01 to 2016-01-01
# ## Trade data split: 2016-01-01 to 2021-01-01
#
# DRL model needs to update periodically in order to take full advantage of the data, ideally we need to retrain our model yearly, quarterly, or monthly. We also need to tune the parameters along the way, in this notebook I only use the in-sample data from 2009-01 to 2016-01 to tune the parameters once, so there is some alpha decay here as the length of trade date extends.
#
# Numerous hyperparameters – e.g. the learning rate, the total number of samples to train on – influence the learning process and are usually determined by testing some variations.

# %%
train = data_split(processed, '2018-01-01', '2019-01-01')
trade = data_split(processed, '2019-01-01', '2020-01-01')
logging.info(f'Training sample size: {len(train)}')
logging.info(f'Trading sample size: {len(trade)}')

# %%
print(StockTradingEnvCashpenalty.__doc__)

# %% [markdown]
# #### state space
# The state space of the observation is as follows
#
# `start_cash, <owned_shares_of_n_assets>, <<indicator_i_for_asset_j> for j in assets>`
#
# indicators are any daily measurement you can achieve. Common ones are 'volume', 'open' 'close' 'high', 'low'.
# However, you can add these as needed,
# The feature engineer adds indicators, and you can add your own as well.
#

# %%
processed.head()


# %%
information_cols = ['daily_variance', 'change']

e_train_gym = StockTradingEnvCashpenalty(df=train, initial_amount=1e6, hmax=5000,
                                         turbulence_threshold=None,
                                         currency='$',
                                         buy_cost_pct=3e-3,
                                         sell_cost_pct=3e-3,
                                         cash_penalty_proportion=0.2,
                                         cache_indicator_data=True,
                                         daily_information_cols=information_cols,
                                         print_verbosity=500,
                                         random_start=True)

e_trade_gym = StockTradingEnvCashpenalty(df=trade, initial_amount=1e6, hmax=5000,
                                         turbulence_threshold=None,
                                         currency='$',
                                         buy_cost_pct=3e-3,
                                         sell_cost_pct=3e-3,
                                         cash_penalty_proportion=0.2,
                                         cache_indicator_data=False,
                                         daily_information_cols=information_cols,
                                         print_verbosity=500,
                                         random_start=False)

# %% [markdown]
# ## Environment for Training
# There are two available environments. The multiprocessing and the single processing env.
# Some models won't work with multiprocessing.
#
# ```python
# # single processing
# env_train, _ = e_train_gym.get_sb_env()
#
#
# #multiprocessing
# env_train, _ = e_train_gym.get_multiproc_env(n = <n_cores>)
# ```
#

# %%
# single processing
env_train, _ = e_train_gym.get_sb_env()

# this is our observation environment. It allows full diagnostics
env_trade, _ = e_trade_gym.get_sb_env()

# %% [markdown]
# <a id='5'></a>
# # Part 6: Implement DRL Algorithms
# * The implementation of the DRL algorithms are based on **OpenAI Baselines** and **Stable Baselines**. Stable Baselines is a fork of OpenAI Baselines, with a major structural refactoring, and code cleanups.
# * FinRL library includes fine-tuned standard DRL algorithms, such as DQN, DDPG,
# Multi-Agent DDPG, PPO, SAC, A2C and TD3. We also allow users to
# design their own DRL algorithms by adapting these DRL algorithms.

# %%
agent = DRLAgent(env=env_train)

# %% [markdown]
# ### Model PPO
#

# %%
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

# model = model.load("scaling_reward.model", env = env_train)


# %%
model.learn(total_timesteps=10000,
            eval_env=env_trade,
            eval_freq=500,
            log_interval=1,
            tb_log_name='env_cashpenalty_highlr',
            n_eval_episodes=1)


# %%
# model.load("different3_20_20000.model")

# %%
model.save("different3_28_10000.model")

# %% [markdown]
# ### Trade
#
# DRL model needs to update periodically in order to take full advantage of the data, ideally we need to retrain our model yearly, quarterly, or monthly. We also need to tune the parameters along the way, in this notebook I only use the in-sample data from 2009-01 to 2018-12 to tune the parameters once, so there is some alpha decay here as the length of trade date extends.
#
# Numerous hyperparameters – e.g. the learning rate, the total number of samples to train on – influence the learning process and are usually determined by testing some variations.

# %%
trade.head()


# %%
e_trade_gym.hmax = 2000


# %%
print(len(e_trade_gym.dates))


# %%
e_trade_gym.patient = True
df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=model, environment=e_trade_gym)


# %%
df_actions.head()


# %%
df_account_value.shape


# %%
df_account_value.head(50)

# %% [markdown]
# <a id='6'></a>
# # Part 7: Backtest Our Strategy
# Backtesting plays a key role in evaluating the performance of a trading strategy. Automated backtesting tool is preferred because it reduces the human error. We usually use the Quantopian pyfolio package to backtest our trading strategies. It is easy to use and consists of various individual plots that provide a comprehensive image of the performance of a trading strategy.
# %% [markdown]
# <a id='6.1'></a>
# ## 7.1 BackTestStats
# pass in df_account_value, this information is stored in env class
#

# %%
print("==============Get Backtest Results===========")
perf_stats_all = backtest_stats(
    account_value=df_account_value, value_col_name='total_assets')

# %% [markdown]
# <a id='6.2'></a>
# ## 7.2 BackTestPlot

# %%
print("==============Compare to DJIA===========")
get_ipython().run_line_magic('matplotlib', 'inline')
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
backtest_plot(df_account_value,
              baseline_ticker='^DJI',
              baseline_start='2019-01-01',
              baseline_end='2020-01-01', value_col_name='total_assets')


# %%
