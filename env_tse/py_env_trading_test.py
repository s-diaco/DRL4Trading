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

        env_utils.validate_py_environment(init_state, episodes=4)

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
            _, _, _, next_state= env.step(actions)
            cash = next_state[0]
            holdings = next_state[1 : 1 + len(ticker_list)]
            asset_value = env.account_information["asset_value"][-1]
            total_assets = env.account_information["total_assets"][-1]
            np.testing.assert_equal(cash, init_amt)
            np.testing.assert_equal(0.0, np.sum(holdings))
            np.testing.assert_equal(0.0, asset_value)
            np.testing.assert_equal(init_amt, total_assets)
            np.testing.assert_equal(i + 1, env.current_step)
