# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from preprocess_data.add_user_features import add_features
from tensortrade.feed.core.base import IterableStream
from envirement.default.actions import DailyTLOrders
from itertools import product
from typing import List
from gym.spaces.discrete import Discrete
from gym.spaces.space import Space
from tensortrade.env.default.actions import ManagedRiskOrders
from tensortrade.oms.instruments.exchange_pair import ExchangePair
from tensortrade.oms.instruments.quantity import Quantity
from tensortrade.oms.orders.create import risk_managed_order
from tensortrade.oms.orders.criteria import Criteria, Stop
from tensortrade.oms.orders.order import Order
from tensortrade.oms.orders.order_spec import OrderSpec
from tensortrade.oms.orders.trade import TradeSide, TradeType
from preprocess_data.csv_data import CSVData
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
from preprocess_data import user_features, preprocess_data
# %% [markdown]
# ## Install TensorTrade

# %%
# get_ipython().system('python3 -m pip install git+https://github.com/tensortrade-org/tensortrade.git')

# %% [markdown]
# ## Setup Data Fetching

# %%
import pandas as pd
import tensortrade.env.default as default

from preprocess_data.read_csv import ReadCSV
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import AAPL, MSFT, TSLA, USD, BTC, ETH
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.agents import DQNAgent


# get_ipython().run_line_magic('matplotlib', 'inline')

FMT = '[%(levelname)s] %(message)s'
formatter = logging.Formatter(FMT)

absl_logging.get_absl_handler().setFormatter(formatter)
absl_logging.set_verbosity('info')

# %%
cdd = ReadCSV()
IRR = Instrument('IRR', 2, 'Iranian Rial')
symbol_list = settings.TSE_TICKER[0:5]
base_dirs = ["tickers_data/tse/adjusted/", "tickers_data/tse/client_types/"]
price_data_dict = {}
data_manager = CSVData(
    start_date="2018-01-01",
    end_date="2020-12-31",
    csv_dirs=base_dirs,
    baseline_file_name="tickers_data/tse/adjusted/شاخص كل6.csv",
    has_daily_trading_limit=True,
    use_baseline_data=True,
)
for quote_symbol in symbol_list:
    file_names = [f'{quote_symbol}-ت.csv', f'{quote_symbol}.csv']
    temp_df = data_manager.process_single_tic(
        file_names,
        None,
        'date'
    )
    if not temp_df.empty:
        price_data_dict[quote_symbol] = Stream.source(temp_df, dtype="float")

# %%
list(price_data_dict.values())[0].iterable.columns
# %%
pd.DataFrame(list(price_data_dict.values())[0].iterable).head()

# %% [markdown]
# ## Create features

# %%
# %% [markdown]
# ## Create features with the feed module

# %%
features = []
candid_features = [
    'MA', 'EMA', 'SMA', 'ChangeStream', 'DailyVarianceStream',
    'SMARatioIndStream', 'VolumeSMARatioStream', 'CountSMARatioStream',
    'IndvBSRatioStream', 'CorpBSRatioStream', 'IndCorpBRatioStream',
    'RSIIndicatorStream']
for quote_stream in price_data_dict:
    features.extend(
        add_features(
            quote_stream,
            price_data_dict[quote_stream],
            candid_features))

# %%
for symbol in symbol_list:
    cp = Stream.source(pd.DataFrame(
        price_data_dict[symbol].iterable).close,
        dtype="float").rename(f'close:/IRR-{symbol}')
    features += [
        cp.log().diff().rename(f'lr:/IRR-{symbol}')
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
    streams.extend([
        Stream.source(
            pd.DataFrame(price_data_dict[symbol].iterable).close,
            dtype="float").rename(f'IRR/{symbol}'),
        Stream.source(
            pd.DataFrame(price_data_dict[symbol].iterable).b_queue,
            dtype="float").rename(f'bqueue-IRR/{symbol}'),
        Stream.source(
            pd.DataFrame(price_data_dict[symbol].iterable).s_queue,
            dtype="float").rename(f'squeue-IRR/{symbol}'),
        Stream.source(
            pd.DataFrame(price_data_dict[symbol].iterable).stopped,
            dtype="float").rename(f'stopped-IRR/{symbol}')])

tsetmc = Exchange("tsetmc", service=execute_order)(
    *streams
)

# %%
instrum_dict = {}
for symbol in symbol_list:
    instrum_dict[symbol] = Instrument(symbol, 2, pd.DataFrame(
        price_data_dict[symbol].iterable).name[0])

wallet_list = [Wallet(tsetmc, 0 * instrum_dict[symbol])
               for symbol in symbol_list]
wallet_list.append(Wallet(tsetmc, 10000000 * IRR))

pfolio = Portfolio(IRR, wallet_list)

# %% [markdown]
# ## Create the action scheme with daily trading limit.
dtl_action_scheme = DailyTLOrders()
# %%

env = default.create(
    portfolio=pfolio,
    action_scheme=dtl_action_scheme,
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

agent.train(n_steps=200, n_episodes=2, save_path="agents/")

# %%
agent.restore("agents/")

# %%
# portfolio.ledger.as_frame().head(20)
for instr in pfolio.total_balances:
    print(f'- {instr}')
# %%
agent = DQNAgent.load("agents/")
action = agent.get_action(env.reset())
env.action_scheme.get_orders(action, pfolio)

# %%
