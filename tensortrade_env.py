# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython
import functools
import logging
from numpy.lib.utils import source

import tensorflow as tf
from absl import app
from absl import logging as absl_logging
from tensortrade.oms import instruments
from tensortrade.oms.instruments.instrument import Instrument
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

from data.read_csv import ReadCSV
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import AAPL, MSFT, TSLA, USD, BTC, ETH
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.agents import DQNAgent


# get_ipython().run_line_magic('matplotlib', 'inline')


# %%
cdd = ReadCSV()
symbol_list = settings.TSE_TICKER
data_dict = {}
for symbol in symbol_list:
    data_dict[symbol] = cdd.fetch("tsetmc", "USD", symbol, "1d", False, "tickers_data/tse/adjusted/")


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


# data = df_train[information_cols]
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
streams = []
for symbol in symbol_list:
    streams.append(
        Stream.source(list(data_dict[symbol]['close'][-100:]), dtype="float").rename("USD-"+symbol))
bitstamp = Exchange("tsetmc", service=execute_order)(
    *streams
)

# %%
instrum_dict = {}
for symbol in symbol_list:
    instrum_dict[symbol] = Instrument(symbol, 1, 'Name of '+symbol)

wallet_list = [Wallet(bitstamp, 0 * instrum_dict[symbol]) for symbol in symbol_list]
wallet_list.append(Wallet(bitstamp, 10000000 * USD))

portfolio = Portfolio(USD, wallet_list)

# %%
streams_feed = []
for symbol in symbol_list:
    streams.append(
        Stream.source(list(data_dict[symbol]['volume'][-100:]), dtype="float").rename("volume:/USD-"+symbol))

feed = DataFeed(streams_feed)

env = default.create(
    portfolio=portfolio,
    action_scheme="managed-risk",
    reward_scheme="risk-adjusted",
    feed=feed,
    renderer=default.renderers.ScreenLogger(),
    window_size=20
)


# %%
env.observer.feed.next()

# %% [markdown]
# ## Setup and Train DQN Agent

# %%
agent = DQNAgent(env)

agent.train(n_steps=200, n_episodes=3, save_path="agents/")

# %%
portfolio.ledger.as_frame().head(20)
portfolio.total_balances
# %%
