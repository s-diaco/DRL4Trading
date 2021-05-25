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
    path = pathlib.Path.cwd()
    df = pd.read_csv(path/'env_tse'/'env_sample_data.csv')
    env = TradingPyEnv(df=df)
    yield env


class TestTradingPyEnv():

    def test_env_init_state(self, init_state):
        env = init_state
        ts = env.reset()
        init_state = np.array(
            [env.initial_amount]
            + [0] * len(env.assets)
            + env.get_date_vector(env.date_index),
            dtype=np.float32
        )
        np.testing.assert_array_equal(
            init_state, ts.observation)

    def test_validate_specs(self, init_state):
        env_utils.validate_py_environment(init_state, episodes=4)
