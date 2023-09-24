# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]
# ## Import packages
# %%
from __future__ import annotations

import glob
import logging
import os

import numpy as np
import pandas as pd
import pytse_client as pytse

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

from src.drl4trading.config import settings
from src.drl4trading.envirement.default.actions import DailyTLOrders
from src.drl4trading.preprocess_data.add_user_features import add_features
from src.drl4trading.preprocess_data.csv_data import CSVData
from src.drl4trading.preprocess_data.read_csv import ReadCSV

# from tf_agents.system import system_multiprocessing as multiprocessing


# get_ipython().run_line_magic('matplotlib', 'inline')

FMT = "[%(levelname)s] %(message)s"
formatter = logging.Formatter(FMT)

absl_logging.get_absl_handler().setFormatter(formatter)
absl_logging.set_verbosity("info")
# %% [markdown]
# ## Define global variables

n_steps = 1000
n_episodes = 20
window_size = 30
memory_capacity = n_steps * 10
save_path = "agents/"
# Number of bins to partition the dataset evenly in order to evaluate class sparsity.
n_bins = 5
seed = 1337
n_symbols = 5  # Number of symbols to download.
days_to_dl = 500
columns_to_dl = [
    # price_data
    "open",
    "high",
    "low",
    "adjClose",
    "value",
    "volume",
    "count",
    "yesterday",
    "close",
    # client_types
    "individual_buy_count",
    "corporate_buy_count",
    "individual_sell_count",
    "corporate_sell_count",
    "individual_buy_vol",
    "corporate_buy_vol",
    "individual_sell_vol",
    "corporate_sell_vol",
    "individual_buy_value",
    "corporate_buy_value",
    "individual_sell_value",
    "corporate_sell_value",
    "individual_buy_mean_price",
    "individual_sell_mean_price",
    "corporate_buy_mean_price",
    "corporate_sell_mean_price",
    "individual_ownership_change",
    # index_data
    "high_index",
    "low_index",
    "open_index",
    "close_index",
    "volume_index",
]
# %% [markdown]
# ## Setup Data Fetching
IRR = Instrument("IRR", 0, "Iranian Rial")
symbols = settings.TSE_TICKER[:n_symbols]


def prepare_data(df: pd.DataFrame):
    df.index = pd.DatetimeIndex(df.index)
    df = df[columns_to_dl].sort_index().iloc[-days_to_dl:]
    return df


def set_index(data: dict[str, pd.DataFrame], column_name: str):
    data = {
        symbol: price_df.set_index(column_name) for symbol, price_df in data.items()
    }
    return data


def fetch_tse_data(symbol_list: list[str]):
    # columns: ['date', 'open', 'high', 'low', 'adjClose',
    # 'value', 'volume', 'count', 'yesterday', 'close']
    price_data = pytse.download(
        symbols=symbol_list, write_to_csv=True, adjust=True, base_path="price_data/"
    )
    price_data = set_index(price_data, "date")

    # columns: ['date', 'individual_buy_count', 'corporate_buy_count',
    # 'individual_sell_count', 'corporate_sell_count', 'individual_buy_vol',
    # 'corporate_buy_vol', 'individual_sell_vol', 'corporate_sell_vol',
    # 'individual_buy_value', 'corporate_buy_value', 'individual_sell_value',
    # 'corporate_sell_value', 'individual_buy_mean_price', 'individual_sell_mean_price',
    # 'corporate_buy_mean_price', 'corporate_sell_mean_price',
    # 'individual_ownership_change']
    client_types = pytse.download_client_types_records(
        symbols=symbol_list, write_to_csv=True, base_path="client_types_data/"
    )
    client_types = set_index(client_types, "date")

    # columns: ['date', 'high', 'low', 'open', 'close', 'volume']
    index_data = pytse.download_financial_indexes(
        symbols="شاخص کل", write_to_csv=True, base_path="price_data/"
    )

    # first DataFrame in index_data dictionary.
    index_df = next(iter(set_index(index_data, "date").values()))

    tse_data = {
        symbol: index_df.join(
            price_data[symbol],
            lsuffix="_index",
        ).join(client_types[symbol])
        for symbol in price_data
    }
    tse_data = {
        tse_symbol: prepare_data(tse_df) for tse_symbol, tse_df in tse_data.items()
    }
    return tse_data


def load_csv(filename):
    df = pd.read_csv("data/" + filename, skiprows=1)
    df.drop(columns=["symbol", "volume_btc"], inplace=True)

    # Fix timestamp from "2019-10-17 09-AM" to "2019-10-17 09-00-00 AM"
    df["date"] = df["date"].str[:14] + "00-00 " + df["date"].str[-2:]

    return prepare_data(df)


data = fetch_tse_data(symbol_list=symbols)

# %% [markdown]
# ## Create features for the feed module
import os
import numpy as np
import ta as ta1
import pandas_ta as ta

import quantstats as qs

qs.extend_pandas()


def fix_dataset_inconsistencies(dataframe: pd.DataFrame, fill_value=None):
    # dataframe = dataframe.reset_index()
    dataframe = dataframe.replace([-np.inf, np.inf], np.nan)

    # This is done to avoid filling middle holes with backfilling.
    if fill_value:
        dataframe.iloc[0, :] = dataframe.iloc[0, :].fillna(fill_value)
    # TODO: fillna just for index? / is .dropna(axis="columns") necessary?
    dataframe = dataframe.ffill(axis="index").bfill(axis="index").dropna(axis="columns")
    # dataframe = dataframe.set_index("data")
    # dataframe.index = pd.DatetimeIndex(dataframe.index)
    return dataframe


# TODO: Delete manually created indicators.
def rsi(
    price: "pd.Series[pd.Float64Dtype]", period: float
) -> "pd.Series[pd.Float64Dtype]":
    r = price.diff()
    upside = np.minimum(r, 0).abs()
    downside = np.maximum(r, 0).abs()
    rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
    return 100 * (1 - (1 + rs) ** -1)


def macd(
    price: "pd.Series[pd.Float64Dtype]", fast: float, slow: float, signal: float
) -> "pd.Series[pd.Float64Dtype]":
    fm = price.ewm(span=fast, adjust=False).mean()
    sm = price.ewm(span=slow, adjust=False).mean()
    md = fm - sm
    signal = md - md.ewm(span=signal, adjust=False).mean()
    return signal


def generate_all_default_quantstats_features(data):
    excluded_indicators = [
        "compare",
        "greeks",
        "information_ratio",
        "omega",
        "r2",
        "r_squared",
        "rolling_greeks",
        "warn",
        "pct_rank",
        "treynor_ratio",
    ]

    indicators_list = [
        f for f in dir(qs.stats) if f[0] != "_" and f not in excluded_indicators
    ]

    df = data.copy()

    for indicator_name in indicators_list:
        try:
            print(f"genertating qt indicator: {indicator_name}")
            indicator = qs.stats.__dict__[indicator_name](df["close"])
            if isinstance(indicator, pd.Series):
                indicator = indicator.to_frame(name=indicator_name)
                df = pd.concat([df, indicator], axis="columns")
        except (pd.errors.InvalidIndexError, ValueError):
            pass

    return df


def generate_features(data: pd.DataFrame):
    # Automatically-generated using pandas_ta
    df = data.copy()

    strategies = [
        "candles",
        "cycles",
        # "momentum",
        # "overlap",
        "performance",
        "statistics",
        # "trend",
        # "volatility",
        "volume",
    ]

    # cores = os.cpu_count()
    cores = 0
    df.ta.cores = cores
    print(f"using {df.ta.cores} cpu cores")

    for strategy in strategies:
        print(f"studying strategy: {strategy}")
        df.ta.study(strategy, exclude=["kvo"])

    # Generate all default indicators from ta library
    ta1.add_all_ta_features(data, "open", "high", "low", "close", "volume", fillna=True)

    # Naming convention across most technical indicator libraries
    data = data.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    # Custom indicators
    features = pd.DataFrame.from_dict(
        {
            "prev_open": data["Open"].shift(1),
            "prev_high": data["High"].shift(1),
            "prev_low": data["Low"].shift(1),
            "prev_close": data["Close"].shift(1),
            "prev_volume": data["Volume"].shift(1),
            "vol_5": data["Close"].rolling(window=5).std().abs(),
            "vol_10": data["Close"].rolling(window=10).std().abs(),
            "vol_20": data["Close"].rolling(window=20).std().abs(),
            "vol_30": data["Close"].rolling(window=30).std().abs(),
            "vol_50": data["Close"].rolling(window=50).std().abs(),
            "vol_60": data["Close"].rolling(window=60).std().abs(),
            "vol_100": data["Close"].rolling(window=100).std().abs(),
            "vol_200": data["Close"].rolling(window=200).std().abs(),
            "ma_5": data["Close"].rolling(window=5).mean(),
            "ma_10": data["Close"].rolling(window=10).mean(),
            "ma_20": data["Close"].rolling(window=20).mean(),
            "ma_30": data["Close"].rolling(window=30).mean(),
            "ma_50": data["Close"].rolling(window=50).mean(),
            "ma_60": data["Close"].rolling(window=60).mean(),
            "ma_100": data["Close"].rolling(window=100).mean(),
            "ma_200": data["Close"].rolling(window=200).mean(),
            "ema_5": ta1.trend.ema_indicator(data["Close"], window=5, fillna=True),
            "ema_10": ta1.trend.ema_indicator(data["Close"], window=10, fillna=True),
            "ema_20": ta1.trend.ema_indicator(data["Close"], window=20, fillna=True),
            "ema_60": ta1.trend.ema_indicator(data["Close"], window=60, fillna=True),
            "ema_64": ta1.trend.ema_indicator(data["Close"], window=64, fillna=True),
            "ema_120": ta1.trend.ema_indicator(data["Close"], window=120, fillna=True),
            "lr_open": np.log(data["Open"]).diff().fillna(0),
            "lr_high": np.log(data["High"]).diff().fillna(0),
            "lr_low": np.log(data["Low"]).diff().fillna(0),
            "lr_close": np.log(data["Close"]).diff().fillna(0),
            "r_volume": data["Close"].diff().fillna(0),
            "rsi_5": rsi(data["Close"], period=5),
            "rsi_10": rsi(data["Close"], period=10),
            "rsi_100": rsi(data["Close"], period=100),
            "rsi_7": rsi(data["Close"], period=7),
            "rsi_28": rsi(data["Close"], period=28),
            "rsi_6": rsi(data["Close"], period=6),
            "rsi_14": rsi(data["Close"], period=14),
            "rsi_26": rsi(data["Close"], period=24),
            "macd_normal": macd(data["Close"], fast=12, slow=26, signal=9),
            "macd_short": macd(data["Close"], fast=10, slow=50, signal=5),
            "macd_long": macd(data["Close"], fast=200, slow=100, signal=50),
        }
    )

    # Concatenate both manually and automatically generated features
    data = pd.concat([data, features], axis="columns").ffill()

    # Remove potential column duplicates
    data = data.loc[:, ~data.columns.duplicated()]

    # Revert naming convention
    data = data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    # Concatenate both manually and automatically generated features
    data = pd.concat([data, df], axis="columns").ffill()

    # Remove potential column duplicates
    data = data.loc[:, ~data.columns.duplicated()]

    # Generate all default quantstats features
    df_quantstats = generate_all_default_quantstats_features(data)

    # Concatenate both manually and automatically generated features
    data = pd.concat([data, df_quantstats], axis="columns").ffill()

    # Remove potential column duplicates
    data = data.loc[:, ~data.columns.duplicated()]

    # A lot of indicators generate NaNs at the beginning of DataFrames, so remove them
    # TODO: Why 200?
    data = data.iloc[200:]

    data = fix_dataset_inconsistencies(data, fill_value=None)
    return data


data = {tse_symbol: generate_features(tse_df) for tse_symbol, tse_df in data.items()}

# %% [markdown]
# ## Remove features with low variance before splitting the dataset
# TODO: What is this?
from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))


def rem_low_variance(data, sel):
    sel.fit(data)
    # TODO: does it change data?
    data[data.columns[sel.get_support(indices=True)]]
    return data


data = {
    tse_symbol: rem_low_variance(tse_df, sel) for tse_symbol, tse_df in data.items()
}

# %% [markdown]
# ## Split datasets
from sklearn.model_selection import train_test_split


def split_data(data: pd.DataFrame):
    X = data.copy()
    y = X["close"].pct_change()

    X_train_test, X_valid, y_train_test, y_valid = train_test_split(
        data, data["close"].pct_change(), train_size=0.67, test_size=0.33, shuffle=False
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_test, y_train_test, train_size=0.50, test_size=0.50, shuffle=False
    )

    return X_train, X_test, X_valid, y_train, y_test, y_valid


splitted_data = {symbol: split_data(df) for symbol, df in data.items()}

import os

cwd = os.getcwd()
for symbol, split_tpl in splitted_data.items():
    X_train, X_test, X_valid, _, _, _ = split_tpl
    train_csv = os.path.join(cwd, f"{symbol}_train.csv")
    test_csv = os.path.join(cwd, f"{symbol}_test.csv")
    valid_csv = os.path.join(cwd, f"{symbol}_valid.csv")
    X_train.to_csv(train_csv)
    X_test.to_csv(test_csv)
    X_valid.to_csv(valid_csv)

# %% [markdown]
# ## Get dataset statistics

from scipy.stats import iqr


def estimate_outliers(data):
    return iqr(data) * 1.5


def estimate_percent_gains(data, column="close"):
    returns = get_returns(data, column=column)
    gains = estimate_outliers(returns)
    return gains


def get_returns(data, column="close"):
    return fix_dataset_inconsistencies(data[[column]].pct_change(), fill_value=0)


def precalculate_ground_truths(data, column="close", threshold=None):
    returns = get_returns(data, column=column)
    gains = estimate_outliers(returns) if threshold is None else threshold
    binary_gains = (returns[column] > gains).astype(int)
    return binary_gains


def is_null(data):
    return data.isnull().sum().sum() > 0


def is_sparse(data, column="close"):
    binary_gains = precalculate_ground_truths(data, column=column)
    bins = [n * (binary_gains.shape[0] // n_bins) for n in range(n_bins)]
    bins += [binary_gains.shape[0]]
    bins = [binary_gains.iloc[bins[n] : bins[n + 1]] for n in range(n_bins)]
    return all([bin.astype(bool).any() for bin in bins])


def is_data_predictible(data, column):
    return not is_null(data) & is_sparse(data, column)


for symbol, sym_data in data.items():
    sym_data.describe(include="all")
    import matplotlib.pyplot as plt

    plt.title(f"returns for {symbol}")
    plt.plot(get_returns(sym_data, column="close"))
    plt.show()
    is_data_predictible(sym_data, "close")
    # Percentage of the dataset generating rewards
    # (keep between 5% to 15% or just rely on is_data_predictible())
    plt.title(f"% of {symbol} dataset generating rewards. keep between 5% to 15%.")
    plt.plot(precalculate_ground_truths(sym_data, column="close").iloc[:1000])
    plt.show()
    percent_rewardable = (
        str(
            round(
                100
                + precalculate_ground_truths(sym_data, column="close")
                .value_counts()
                .pct_change()
                .iloc[-1]
                * 100,
                2,
            )
        )
        + "%"
    )
    print(percent_rewardable)

# %% [markdown]
# ## Threshold to pass to AnomalousProfit reward scheme
for symbol, split_tpl in splitted_data.items():
    X_train, X_test, _, _, _, _ = split_tpl
    X_train_test = pd.concat([X_train, X_test], axis="index")
    # threshold = estimate_percent_gains(X_train_test, 'close')
    threshold = estimate_percent_gains(X_train, "close")
    print(f"threshold: {threshold} ({symbol})")
# %% [markdown]
# ## Implement basic feature engineering

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from feature_engine.selection import SelectBySingleFeaturePerformance

rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=6)

sel = SelectBySingleFeaturePerformance(
    variables=None, estimator=rf, scoring="roc_auc", cv=5, threshold=0.5
)
feature_performances = {}
for symbol, splitted_dfs in splitted_data.items():
    X_train, _, _, _, _, _ = splitted_dfs
    sel.fit(X_train, precalculate_ground_truths(X_train, column="close"))
    feature_performances[symbol] = pd.Series(sel.feature_performance_).sort_values(
        ascending=False
    )
    print(f"feature performances for {symbol}:")
    print(feature_performances[symbol])
feature_performance = pd.DataFrame(feature_performances).mean(axis=1)
feature_performance = feature_performance.sort_values(ascending=False)
# %%
feature_performance.plot.bar(figsize=(20, 5))
plt.title("Performance of ML models trained with individual features")
plt.ylabel("roc-auc")

# %%
features_to_drop = sel.features_to_drop_
# %%
to_drop = list(set(features_to_drop) - set(["open", "high", "low", "close", "volume"]))
print(f"{len(to_drop)} features will be dropped:")
print(features_to_drop)

# %% checkpoint
data = next(iter(data.values()))
X_train, X_test, X_valid, y_train, y_test, y_valid = next(iter(splitted_data.values()))

# %%
X_train = X_train.drop(columns=to_drop)
X_test = X_test.drop(columns=to_drop)
X_valid = X_valid.drop(columns=to_drop)

print(X_train.shape, X_test.shape, X_valid.shape)

# %%
X_train.columns.tolist()

# %% [markdown]
# ## Normalize the dataset subsets to make the model converge faster

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

scaler_type = MinMaxScaler


def get_feature_scalers(X, scaler_type=scaler_type):
    scalers = []
    for name in list(X.columns[X.columns != "date"]):
        scalers.append(scaler_type().fit(X[name].values.reshape(-1, 1)))
    return scalers


def get_scaler_transforms(X, scalers):
    X_scaled = []
    for name, scaler in zip(list(X.columns[X.columns != "date"]), scalers):
        X_scaled.append(scaler.transform(X[name].values.reshape(-1, 1)))
    X_scaled = pd.concat(
        [
            pd.DataFrame(column, columns=[name])
            for name, column in zip(list(X.columns[X.columns != "date"]), X_scaled)
        ],
        axis="columns",
    )
    return X_scaled


def normalize_data(X_train, X_test, X_valid):
    X_train_test = pd.concat([X_train, X_test], axis="index")
    X_train_test_valid = pd.concat([X_train_test, X_valid], axis="index")

    X_train_test_dates = X_train_test[["date"]]
    X_train_test_valid_dates = X_train_test_valid[["date"]]

    X_train_test = X_train_test.drop(columns=["date"])
    X_train_test_valid = X_train_test_valid.drop(columns=["date"])

    train_test_scalers = get_feature_scalers(X_train_test, scaler_type=scaler_type)
    train_test_valid_scalers = get_feature_scalers(
        X_train_test_valid, scaler_type=scaler_type
    )

    X_train_test_scaled = get_scaler_transforms(X_train_test, train_test_scalers)
    X_train_test_valid_scaled = get_scaler_transforms(
        X_train_test_valid, train_test_scalers
    )
    X_train_test_valid_scaled_leaking = get_scaler_transforms(
        X_train_test_valid, train_test_valid_scalers
    )

    X_train_test_scaled = pd.concat(
        [X_train_test_dates, X_train_test_scaled], axis="columns"
    )
    X_train_test_valid_scaled = pd.concat(
        [X_train_test_valid_dates, X_train_test_valid_scaled], axis="columns"
    )
    X_train_test_valid_scaled_leaking = pd.concat(
        [X_train_test_valid_dates, X_train_test_valid_scaled_leaking], axis="columns"
    )

    X_train_scaled = X_train_test_scaled.iloc[: X_train.shape[0]]
    X_test_scaled = X_train_test_scaled.iloc[X_train.shape[0] :]
    X_valid_scaled = X_train_test_valid_scaled.iloc[X_train_test.shape[0] :]
    X_valid_scaled_leaking = X_train_test_valid_scaled_leaking.iloc[
        X_train_test.shape[0] :
    ]

    return (
        train_test_scalers,
        train_test_valid_scalers,
        X_train_scaled,
        X_test_scaled,
        X_valid_scaled,
        X_valid_scaled_leaking,
    )


# %%
(
    train_test_scalers,
    train_test_valid_scalers,
    X_train_scaled,
    X_test_scaled,
    X_valid_scaled,
    X_valid_scaled_leaking,
) = normalize_data(X_train, X_test, X_valid)

# %% [markdown]
# ## Write a reward scheme encouraging rare volatile upside trades

# %%
from tensortrade.env.default.rewards import TensorTradeRewardScheme


class AnomalousProfit(TensorTradeRewardScheme):
    """A simple reward scheme that rewards the agent for exceeding a
    precalculated percentage in the net worth.

    Parameters
    ----------
    threshold : float
        The minimum value to exceed in order to get the reward.

    Attributes
    ----------
    threshold : float
        The minimum value to exceed in order to get the reward.
    """

    registered_name = "anomalous"

    def __init__(self, threshold: float = 0.02, window_size: int = 1):
        self._window_size = self.default("window_size", window_size)
        self._threshold = self.default("threshold", threshold)

    def get_reward(self, portfolio: "Portfolio") -> float:
        """Rewards the agent for incremental increases in net worth over a
        sliding window.

        Parameters
        ----------
        portfolio : `Portfolio`
            The portfolio being used by the environment.

        Returns
        -------
        int
            Whether the last percent change in net worth exceeds the predefined
            `threshold`.
        """
        performance = pd.DataFrame.from_dict(portfolio.performance).T
        current_step = performance.shape[0]
        if current_step > 1:
            # Hint: make it cumulative.
            net_worths = performance["net_worth"]
            ground_truths = precalculate_ground_truths(
                performance, column="net_worth", threshold=self._threshold
            )
            reward_factor = 2.0 * ground_truths - 1.0
            # return net_worths.iloc[-1] / net_worths.iloc[-min(current_step, self._window_size + 1)] - 1.0
            return (reward_factor * net_worths.abs()).iloc[-1]

        else:
            return 0.0


# %%
class PenalizedProfit(TensorTradeRewardScheme):
    """A reward scheme which penalizes net worth loss and
    decays with the time spent.

    Parameters
    ----------
    cash_penalty_proportion : float
        cash_penalty_proportion

    Attributes
    ----------
    cash_penalty_proportion : float
        cash_penalty_proportion.
    """

    registered_name = "penalized"

    def __init__(self, cash_penalty_proportion: float = 0.10):
        self._cash_penalty_proportion = self.default(
            "cash_penalty_proportion", cash_penalty_proportion
        )

    def get_reward(self, portfolio: "Portfolio") -> float:
        """Rewards the agent for gaining net worth while holding the asset.

        Parameters
        ----------
        portfolio : `Portfolio`
            The portfolio being used by the environment.

        Returns
        -------
        int
            A penalized reward.
        """
        performance = pd.DataFrame.from_dict(portfolio.performance).T
        current_step = performance.shape[0]
        if current_step > 1:
            initial_amount = portfolio.initial_net_worth
            net_worth = performance["net_worth"].iloc[-1]
            cash_worth = performance["bitstamp:/USD:/total"].iloc[-1]
            cash_penalty = max(
                0, (net_worth * self._cash_penalty_proportion - cash_worth)
            )
            net_worth -= cash_penalty
            reward = (net_worth / initial_amount) - 1
            reward /= current_step
            return reward
        else:
            return 0.0


# %% [markdown]
# ## TODO: implement tuning

# %%


# %% [markdown]
# ## Setup Trading Environment

# %%
import tensortrade.env.default as default

from tensortrade.agents import DQNAgent
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.feed.core.base import NameSpace
from tensortrade.env.default.actions import BSH
from tensortrade.env.default.rewards import RiskAdjustedReturns, SimpleProfit
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC, ETH
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.orders import TradeType

# TODO: adjust according to your commission percentage, if present
commission = 0.001
price = Stream.source(list(X_train["close"]), dtype="float").rename("USD-BTC")
# bitstamp_options = ExchangeOptions(commission=commission)
# bitstamp = Exchange("bitstamp",
#                    service=execute_order,
#                    options=bitstamp_options)(price)
bitstamp = Exchange("bitstamp", service=execute_order)(price)

cash = Wallet(bitstamp, 50000 * USD)
asset = Wallet(bitstamp, 0 * BTC)

portfolio = Portfolio(USD, [cash, asset])

with NameSpace("bitstamp"):
    features = [
        Stream.source(list(X_train_scaled[c]), dtype="float").rename(c)
        for c in X_train_scaled.columns[1:]
        # Stream.source(list(X_train_scaled['lr_close']), dtype="float").rename('lr_close')
    ]

feed = DataFeed(features)
feed.compile()

renderer_feed = DataFeed(
    [
        Stream.source(list(X_train["date"])).rename("date"),
        Stream.source(list(X_train["open"]), dtype="float").rename("open"),
        Stream.source(list(X_train["high"]), dtype="float").rename("high"),
        Stream.source(list(X_train["low"]), dtype="float").rename("low"),
        Stream.source(list(X_train["close"]), dtype="float").rename("close"),
        Stream.source(list(X_train["volume"]), dtype="float").rename("volume"),
    ]
)

action_scheme = BSH(cash=cash, asset=asset)

# reward_scheme = RiskAdjustedReturns(return_algorithm='sortino',
#                                    window_size=30)

# reward_scheme = SimpleProfit(window_size=30)

reward_scheme = AnomalousProfit(threshold=threshold)

# reward_scheme = PenalizedProfit(cash_penalty_proportion=0.1)

env = default.create(
    portfolio=portfolio,
    action_scheme=action_scheme,
    reward_scheme=reward_scheme,
    feed=feed,
    renderer_feed=renderer_feed,
    renderer=default.renderers.PlotlyTradingChart(),
    window_size=30,
)

# %%
env.observer.feed.next()

# %% [markdown]
# ## Setup and Train DQN Agent


# %%
def get_optimal_batch_size(window_size=30, n_steps=1000, batch_factor=4, stride=1):
    """
    lookback = 30          # Days of past data (also named window_size).
    batch_factor = 4       # batch_size = (sample_size - lookback - stride) // batch_factor
    stride = 1             # Time series shift into the future.
    """
    lookback = window_size
    sample_size = n_steps
    batch_size = (sample_size - lookback - stride) // batch_factor
    return batch_size


batch_size = get_optimal_batch_size(
    window_size=window_size, n_steps=n_steps, batch_factor=4
)
batch_size

# %%
agent = DQNAgent(env)

agent.train(
    batch_size=batch_size,
    n_steps=n_steps,
    n_episodes=n_episodes,
    memory_capacity=memory_capacity,
    save_path=save_path,
)

# %% [markdown]
# ## Implement validation here

# %%


# %% [markdown]
# ## Print basic quantstats report


# %%
def print_quantstats_full_report(env, data, output="dqn_quantstats"):
    performance = pd.DataFrame.from_dict(
        env.action_scheme.portfolio.performance, orient="index"
    )
    net_worth = performance["net_worth"].iloc[window_size:]
    returns = net_worth.pct_change().iloc[1:]

    # WARNING! The dates are fake and default parameters are used!
    returns.index = pd.date_range(
        start=data["date"].iloc[0], freq="1d", periods=returns.size
    )

    qs.reports.full(returns)
    qs.reports.html(returns, output=output + ".html")


print_quantstats_full_report(env, data)

# %%
