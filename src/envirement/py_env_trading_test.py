# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Tests for environment."""

import pathlib

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import tf_agents
from tf_agents.environments import parallel_py_environment, tf_py_environment
from tf_agents.environments import utils as env_utils

from .py_env_trading import TradingPyEnv


@pytest.fixture
def init_state():
    """
    build a test env
    """

    path = pathlib.Path.cwd()
    env_df = pd.read_csv(path/'env_tse'/'env_sample_data.csv')
    env = TradingPyEnv(df=env_df)
    yield env


@pytest.fixture
def get_env_df():
    """
    return a test dataframe to build test env
    """

    path = pathlib.Path.cwd()
    test_df = pd.read_csv(path/'env_tse'/'env_sample_data.csv')
    yield test_df


class TestTradingPyEnv():
    """
    tests for trading env
    """

    def test_env_init_state(self, init_state):
        """
        test  if env initial state is correct
        """

        env = init_state
        ts_env = env.reset()
        init_state = np.array(
            [env.initial_amount]
            + [0] * len(env.assets)
            + env.get_date_vector(env.date_index),
            dtype=np.float32
        )
        np.testing.assert_array_equal(
            init_state, ts_env.observation)

    def test_validate_specs(self, init_state):
        """ validate env """

        env_utils.validate_py_environment(init_state)

    def test_single_stock_action(self, get_env_df):
        """ validate env when env_single_stock_action = True """
        env_single_stock_action = TradingPyEnv(df=get_env_df, single_stock_action=True)
        env_utils.validate_py_environment(env_single_stock_action)

    def test_zero_step(self, get_env_df):
        """
        Prove that zero actions results in zero stock buys, and no price changes
        """

        init_amt = 1e6
        env = TradingPyEnv(
            df=get_env_df, initial_amount=init_amt, cache_indicator_data=False
        )
        ticker_list = get_env_df["tic"].unique()
        _ = env.reset()

        # step with all zeros
        for i in range(2):

            actions = np.zeros(len(ticker_list))
            _, _, _, next_state = env.step(actions)
            cash = next_state[0]
            holdings = next_state[1 : 1 + len(ticker_list)]
            asset_value = env.account_information["asset_value"][-1]
            total_assets = env.account_information["total_assets"][-1]
            np.testing.assert_equal(cash, init_amt)
            np.testing.assert_equal(0.0, np.sum(holdings))
            np.testing.assert_equal(0.0, asset_value)
            np.testing.assert_equal(init_amt, total_assets)
            np.testing.assert_equal(i + 1, env.current_step)
            

    def test_shares_increment(self, get_env_df):
        """
        Prove that we can only buy/sell multiplies of shares
        based on shares_increment parameter
        """

        test_tic = "بترانس"
        stock_first_close = get_env_df[get_env_df['tic']==test_tic].head(1)['close'].values[0]
        init_amt = 1e6
        hmax = stock_first_close * 100
        shares_increment = 10
        env = TradingPyEnv(discrete_actions = True,
            df=get_env_df, initial_amount=init_amt, hmax=hmax,
            cache_indicator_data=False,shares_increment=shares_increment,
            random_start=False
        )
        _ = env.reset()

        actions = np.array([0.29,0.0,0.0,0.0,0.0])
        _, _, _, next_state = env.step(actions)
        ticker_list = get_env_df["tic"].unique()
        holdings = next_state[1 : 1 + len(ticker_list)]
        np.testing.assert_equal(holdings[0], 20.0)
        np.testing.assert_equal(holdings[1], 0.0)

        hmax_mc = get_env_df[get_env_df['tic']==test_tic].head(2).iloc[-1]['close'] / stock_first_close
        actions = np.array([-0.12 * hmax_mc,0.0,0.0,0.0,0.0])
        _, _, _, next_state = env.step(actions)
        holdings = next_state[1 : 1 + len(ticker_list)]
        np.testing.assert_equal(holdings[0], 10.0)
        np.testing.assert_equal(holdings[1], 0.0)

    def test_patient(self, get_env_df):
        """
        Prove that we just not buying any new assets 
        if running out of cash 
        and the cycle is not ended
        """

        test_tic = "بترانس"
        stock_first_close = get_env_df[get_env_df['tic']==test_tic].head(1)['close'].values[0]
        init_amt = stock_first_close
        hmax = stock_first_close * 100
        env = TradingPyEnv(
            df=get_env_df, initial_amount=init_amt, hmax=hmax,
            cache_indicator_data=False,patient=True,
            random_start=False, 
        )
        _ = env.reset()
        ticker_list = get_env_df["tic"].unique()

        actions = np.array([1.0,1.0,1.0,1.0,1.0])
        is_done, _, _, next_state = env.step(actions)
        holdings = next_state[1 : 1 + len(ticker_list)]
        np.testing.assert_equal(1, is_done)
        np.testing.assert_equal(0.0, np.sum(holdings))

    def test_cost_penalties(self):
        # TODO: Requesting contributions!
        pass

    def test_purchases(self):
        # TODO: Requesting contributions!
        pass

    def test_gains(self):
        # TODO: Requesting contributions!
        pass

    def test_validate_caching(self, get_env_df):
        """
        prove that results with or without caching
        don't change anything
        """
        
        env_uncached = TradingPyEnv(
            df=get_env_df, cache_indicator_data=False, random_start=False
        )
        env_cached = TradingPyEnv(
            df=get_env_df, cache_indicator_data=True, random_start=False
        )
        _ = env_uncached.reset()
        _ = env_cached.reset()
        for _ in range(10):
            actions = np.random.uniform(low=-1, high=1, size=5)
            _, un_reward, _, un_state = env_uncached.step(actions)
            _, ca_reward, _, ca_state = env_cached.step(actions)

            np.testing.assert_equal(un_state, ca_state)
            np.testing.assert_equal(un_reward, ca_reward)

    def test_parallel_env(self, get_env_df):
        """ test parallel envs """
        tf_agents.system.multiprocessing.enable_interactive_mode()
        num_parallel_environments = 4
        parall_py_env = parallel_py_environment.ParallelPyEnvironment(
            [lambda: TradingPyEnv(df=get_env_df, single_stock_action=True)] * num_parallel_environments)
        env_utils.validate_py_environment(parall_py_env)
