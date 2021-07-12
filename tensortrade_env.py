# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython
from backtest.backtest import backtest_trades
import functools
import logging

import tensorflow as tf
from absl import app
from absl import logging as absl_logging
from tf_agents.system import system_multiprocessing as multiprocessing

from config import settings
from envirement.trading_py_env import TradingPyEnv
from models.model_ppo import TradeDRLAgent
from preprocess_data import preprocess_data
# %% [markdown]
# ## Install TensorTrade

# %%
# get_ipython().system('python3 -m pip install git+https://github.com/tensortrade-org/tensortrade.git')

# %% [markdown]
# ## Setup Data Fetching

# %%
import pandas as pd
import tensortrade.env.default as default

from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC, ETH
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.agents import DQNAgent


get_ipython().run_line_magic('matplotlib', 'inline')


# %%
# cdd = CryptoDataDownload()

# data = cdd.fetch("Bitstamp", "USD", "BTC", "1h")


# %%
# data.head()

# %% [markdown]
# ## Create features with the feed module
FMT = '[%(levelname)s] %(message)s'
formatter = logging.Formatter(FMT)

absl_logging.get_absl_handler().setFormatter(formatter)
absl_logging.set_verbosity('info')
ticker_list = settings.DOW_30_TICKER
data_columns = settings.DATA_COLUMNS
data_columns.append('close')
data_columns.append('open')
data_columns.append('high')
data_columns.append('low')
data_columns.append('volume')
data_columns.append('date')


# Preprocess data
df_train = preprocess_data.preprocess_data(
    tic_list=ticker_list,
    start_date=settings.START_TRAIN_DATE,
    end_date=settings.END_TRAIN_DATE,
    field_mappings=settings.CSV_FIELD_MAPPINGS,
    baseline_filed_mappings=settings.BASELINE_FIELD_MAPPINGS,
    csv_file_info=settings.CSV_FILE_SETTINGS,
    user_columns=settings.USER_DEFINED_FEATURES
)
information_cols = []
unavailable_cols = []
for col in data_columns:
    if col in df_train.columns:
        information_cols.append(col)
    else:
        unavailable_cols.append(col)
if not information_cols:
    logging.error('No column to train')
    raise ValueError
else:
    logging.info(f'Columns used to train:\n{information_cols} ✅')
    if unavailable_cols:
        logging.info(f'Unavailable columns:\n{unavailable_cols} ❌')


data = df_train[information_cols]
# %%
def rsi(price: Stream[float], period: float) -> Stream[float]:
    r = price.diff()
    upside = r.clamp_min(0).abs()
    downside = r.clamp_max(0).abs()
    rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
    return 100*(1 - (1 + rs) ** -1)


def macd(price: Stream[float], fast: float, slow: float, signal: float) -> Stream[float]:
    fm = price.ewm(span=fast, adjust=False).mean()
    sm = price.ewm(span=slow, adjust=False).mean()
    md = fm - sm
    signal = md - md.ewm(span=signal, adjust=False).mean()
    return signal


features = []
for c in data.columns[1:]:
    s = Stream.source(list(data[c]), dtype="float").rename(data[c].name)
    features += [s]

# %%
cp = Stream.select(features, lambda s: s.name == "close")

features = [
    cp.log().diff().rename("lr"),
    rsi(cp, period=20).rename("rsi"),
    macd(cp, fast=10, slow=50, signal=5).rename("macd")
]

feed = DataFeed(features)
feed.compile()


# %%
for i in range(5):
    print(feed.next())

# %% [markdown]
# ## Setup Trading Environment

# %%
bitstamp = Exchange("bitstamp", service=execute_order)(
    Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
)

portfolio = Portfolio(USD, [
    Wallet(bitstamp, 10000 * USD)
])


renderer_feed = DataFeed([
    Stream.source(list(data["date"])).rename("date"),
    Stream.source(list(data["open"]), dtype="float").rename("open"),
    Stream.source(list(data["high"]), dtype="float").rename("high"),
    Stream.source(list(data["low"]), dtype="float").rename("low"),
    Stream.source(list(data["close"]), dtype="float").rename("close"), 
    Stream.source(list(data["volume"]), dtype="float").rename("volume") 
])


env = default.create(
    portfolio=portfolio,
    action_scheme="managed-risk",
    reward_scheme="risk-adjusted",
    feed=feed,
    renderer_feed=renderer_feed,
    renderer=default.renderers.ScreenLogger(),
    window_size=20
)


# %%
env.observer.feed.next()

# %% [markdown]
# ## Setup and Train DQN Agent

# %%
agent = DQNAgent(env)

agent.train(n_steps=200, n_episodes=2, save_path="agents/")

# %%
