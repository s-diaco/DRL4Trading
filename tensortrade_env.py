# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from typing import List
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

from preprocess_data.read_csv import ReadCSV
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import AAPL, MSFT, TSLA, USD, BTC, ETH
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.agents import DQNAgent


# get_ipython().run_line_magic('matplotlib', 'inline')


# %%
cdd = ReadCSV()
IRR = Instrument('IRR', 2, 'Iranian Rial')
symbol_list = settings.TSE_TICKER[0:5]
base_dirs = ["tickers_data/tse/adjusted/", "tickers_data/tse/client_types/"]
price_data_dict = {}
data_manager = CSVData(
    start_date = "2018-01-01",
    end_date = "2020-12-31",
    csv_dirs = base_dirs,
    baseline_file_name = "tickers_data/tse/adjusted/شاخص كل6.csv",
    has_daily_trading_limit = True,
    use_baseline_data = True,
)
for quote_symbol in symbol_list:
    file_names = [f'{quote_symbol}-ت.csv', f'{quote_symbol}.csv']
    temp_df = data_manager.process_single_tic(
        file_names,
        None,
        'date'
    )
    if not temp_df.empty:
        price_data_dict[quote_symbol] = temp_df

# %%
list(price_data_dict.values())[0].head()
# %%
list(price_data_dict.values())[0].columns
# %% [markdown]
# ## Create features with the feed module
FMT = '[%(levelname)s] %(message)s'
formatter = logging.Formatter(FMT)

absl_logging.get_absl_handler().setFormatter(formatter)
absl_logging.set_verbosity('info')

# %%
def rsi(price: Stream[float], period: float) -> Stream[float]:
    r = price.diff()
    upside = r.clamp_min(0).abs()
    downside = r.clamp_max(0).abs()
    rs = upside.ewm(alpha=1 / period).mean() / \
        downside.ewm(alpha=1 / period).mean()
    return 100*(1 - (1 + rs) ** -1)


def macd(price: Stream[float], fast: float, slow: float, signal: float) -> Stream[float]:
    fm = price.ewm(span=fast, adjust=False).mean()
    sm = price.ewm(span=slow, adjust=False).mean()
    md = fm - sm
    signal = md - md.ewm(span=signal, adjust=False).mean()
    return signal

# %%
features = []
for symbol in symbol_list:
    cp = Stream.source(list(price_data_dict[symbol]['close']), dtype="float").rename(f'close:/IRR-{symbol}')

    features += [
        cp.log().diff().rename(f'lr:/IRR-{symbol}'),
        rsi(cp, period=20).rename(f'rsi:/IRR-{symbol}'),
        macd(cp, fast=10, slow=50, signal=5).rename(f'macd:/IRR-{symbol}')
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
        Stream.source(list(price_data_dict[symbol]['close']), dtype="float").rename(f'IRR/{symbol}'),
        Stream.source(list(price_data_dict[symbol]['b_queue']), dtype="float").rename(f'bqueue-IRR/{symbol}'),
        Stream.source(list(price_data_dict[symbol]['s_queue']), dtype="float").rename(f'squeue-IRR/{symbol}'),
        Stream.source(list(price_data_dict[symbol]['stopped']), dtype="float").rename(f'stopped-IRR/{symbol}')])
tsetmc = Exchange("tsetmc", service=execute_order)(
    *streams
)

# %%
instrum_dict = {}
for symbol in symbol_list:
    instrum_dict[symbol] = Instrument(symbol, 2, price_data_dict[symbol]['name'][0])

wallet_list = [Wallet(tsetmc, 0 * instrum_dict[symbol])
               for symbol in symbol_list]
wallet_list.append(Wallet(tsetmc, 10000000 * IRR))

pfolio = Portfolio(IRR, wallet_list)





# %% [markdown]
# ## Create the action scheme with daily trading limit.

class DailyTL(Criteria):
    """An order criteria that allows execution when daily trading limit allows.
    """

    def check(self, order: 'Order', exchange: 'Exchange') -> bool:
        b_queue = exchange._price_streams[f'bqueue/{order.pair}'].value
        s_queue = exchange._price_streams[f'squeue/{order.pair}'].value
        stopped = exchange._price_streams[f'stopped/{order.pair}'].value
        buy_satisfied = (order.side == TradeSide.BUY and not b_queue)
        sell_satisfied = (order.side == TradeSide.SELL and not s_queue)
        dtl_satisfied = (buy_satisfied or sell_satisfied) and not stopped
        return dtl_satisfied

    def __str__(self) -> str:
        return f"<Limit: price={self.limit_price}>"

def risk_managed_dtl_order(side: "TradeSide",
                       trade_type: "TradeType",
                       exchange_pair: "ExchangePair",
                       price: float,
                       quantity: "Quantity",
                       down_percent: float,
                       up_percent: float,
                       portfolio: "Portfolio",
                       start: int = None,
                       end: int = None):
    """Create a stop order that manages for percentages above and below the
    entry price of the order.

    Parameters
    ----------
    side : `TradeSide`
        The side of the order.
    trade_type : `TradeType`
        The type of trade to make when going in.
    exchange_pair : `ExchangePair`
        The exchange pair to perform the order for.
    price : float
        The current price.
    down_percent: float
        The percentage the price is allowed to drop before exiting.
    up_percent : float
        The percentage the price is allowed to rise before exiting.
    quantity : `Quantity`
        The quantity of the order.
    portfolio : `Portfolio`
        The portfolio being used in the order.
    start : int, optional
        The start time of the order.
    end : int, optional
        The end time of the order.

    Returns
    -------
    `Order`
        A stop order controlling for the percentages above and below the entry
        price.
    """

    side = TradeSide(side)
    instrument = side.instrument(exchange_pair.pair)

    order = Order(
        step=portfolio.clock.step,
        side=side,
        trade_type=TradeType(trade_type),
        exchange_pair=exchange_pair,
        price=price,
        start=start,
        end=end,
        quantity=quantity,
        portfolio=portfolio,
        criteria=DailyTL()
    )

    criteria = (Stop("down", down_percent) ^ Stop("up", up_percent)) & DailyTL()
    risk_management = OrderSpec(
        side=TradeSide.SELL if side == TradeSide.BUY else TradeSide.BUY,
        trade_type=TradeType.MARKET,
        exchange_pair=exchange_pair,
        criteria=criteria
    )

    order.add_order_spec(risk_management)

    return order


class DailyTLOrders(ManagedRiskOrders):
    """A discrete action scheme for markets with daily trading limits
    based on "ManagedRiskOrders" from tensortrade.
    """
    
    def get_orders(self, action: int, portfolio: 'Portfolio') -> 'List[Order]':

        if action == 0:
            return []

        (ep, (stop, take, proportion, duration, side)) = self.actions[action]

        side = TradeSide(side)

        instrument = side.instrument(ep.pair)
        wallet = portfolio.get_wallet(ep.exchange.id, instrument=instrument)

        balance = wallet.balance.as_float()
        size = (balance * proportion)
        size = min(balance, size)
        quantity = (size * instrument).quantize()

        if size < 10 ** -instrument.precision \
                or size < self.min_order_pct * portfolio.net_worth \
                or size < self.min_order_abs:
            return []

        params = {
            'side': side,
            'exchange_pair': ep,
            'price': ep.price,
            'quantity': quantity,
            'down_percent': stop,
            'up_percent': take,
            'portfolio': portfolio,
            'trade_type': self._trade_type,
            'end': self.clock.step + duration if duration else None
        }

        order = risk_managed_dtl_order(**params)

        if self._order_listener is not None:
            order.attach(self._order_listener)

        return [order]

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

agent.train(n_steps=200, n_episodes=3, save_path="agents/")

# %%
# portfolio.ledger.as_frame().head(20)
print(pfolio.total_balances)
# %%
