# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
# import functools
import glob
import logging
import os

import pandas as pd

# import tensorflow as tf
import tensortrade.env.default as default

# from absl import app
from absl import logging as absl_logging

# from IPython import get_ipython
from tensortrade.agents import DQNAgent
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.exchanges.exchange import ExchangeOptions
from tensortrade.oms.instruments.instrument import Instrument
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet

# from tf_agents.system import system_multiprocessing as multiprocessing

from src.drl4trading.config import settings
from src.drl4trading.envirement.default.actions import DailyTLOrders
from src.drl4trading.preprocess_data.add_user_features import add_features
from src.drl4trading.preprocess_data.csv_data import CSVData
from src.drl4trading.preprocess_data.read_csv import ReadCSV

import pytse_client as pytse
import quantstats as qs

# get_ipython().run_line_magic('matplotlib', 'inline')

FMT = "[%(levelname)s] %(message)s"
formatter = logging.Formatter(FMT)

absl_logging.get_absl_handler().setFormatter(formatter)
absl_logging.set_verbosity("info")
# %% [markdown]
# ## Setup Data Fetching
# %%
IRR = Instrument("IRR", 2, "Iranian Rial")
symbol_list = settings.TSE_TICKER[:5]
tickers = pytse.download(symbols=symbol_list, write_to_csv=True, adjust=True)

tse_exch = Exchange("tsetmc", service=execute_order)(
    *(
        Stream.source(list(ticker_df["close"][-100:]), dtype="float").rename(
            f"IRR-{ticker}"
        )
        for ticker, ticker_df in tickers.items()
    )
)
instrum_dict = {symb: Instrument(symbol=symb, precision=0) for symb in symbol_list}
wallet_list = [Wallet(tse_exch, 0 * instrum) for instrum in instrum_dict.values()]
wallet_list.append(Wallet(tse_exch, 10000000 * IRR))

portfolio = Portfolio(IRR, wallet_list)
feed = DataFeed(
    [
        Stream.source(list(ticker_df["volume"][-100:]), dtype="float").rename(
            f"volume:/IRR-{ticker}"
        )
        for ticker, ticker_df in tickers.items()
    ]
)
env = default.create(
    portfolio=portfolio,
    action_scheme=default.actions.SimpleOrders(),
    reward_scheme=default.rewards.SimpleProfit(),
    feed=feed,
)
done = False
obs = env.reset()
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

portfolio.ledger.as_frame().head(7)


# %% [markdown]
# ## Print basic quantstats report
# %%
def print_quantstats_full_report(env, data, output="dqn_quantstats"):
    performance = pd.DataFrame.from_dict(
        env.action_scheme.portfolio.performance, orient="index"
    )
    net_worth = performance["net_worth"]
    returns = net_worth.pct_change().iloc[1:]

    # WARNING! The dates are fake and default parameters are used!
    returns.index = pd.date_range(
        start=data["date"].iloc[0], freq="1d", periods=returns.size
    )

    qs.reports.full(returns)
    qs.reports.html(returns, output=output + ".html")


print_quantstats_full_report(env, tickers[symbol_list[0]][-100:])
# %%
