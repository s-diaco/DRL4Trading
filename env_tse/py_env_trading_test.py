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
"""Tests for tf_agents.environments.examples.tic_tac_toe_environment."""

import pathlib
import numpy as np
import pandas as pd

from tf_agents.environments import utils as env_utils
from .py_env_trading import TradingPyEnv
from tf_agents.utils import test_utils


class TradingPyEnvTest(test_utils.TestCase):

    def setUp(self):
        path = pathlib.Path.cwd()
        df = pd.read_csv(path/'env_tse'/'env_sample_data.csv')
        super(TradingPyEnvTest, self).setUp()
        np.random.seed(0)
        self.discount = np.asarray(1., dtype=np.float32)
        self.env = TradingPyEnv(df=df)
        ts = self.env.reset()
        init_state = np.array(
            [self.env.initial_amount]
            + [0] * len(self.env.assets)
            + self.env.get_date_vector(self.env.date_index),
            dtype=np.float32
        )
        np.testing.assert_array_equal(
            init_state, ts.observation)

    def test_validate_specs(self):
        env_utils.validate_py_environment(self.env, episodes=4)


if __name__ == '__main__':
    test_utils.main()
