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

r"""Train and Eval PPO.
To run:
```bash
tensorboard --logdir $HOME/tmp/ppo/gym/HalfCheetah-v2/ --port 2223 &
python tf_agents/agents/ppo/examples/v2/train_eval_clip_agent.py \
  --root_dir=$HOME/tmp/ppo/gym/HalfCheetah-v2/ \
  --logtostderr
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.utils import common


class TradeDRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        train_PPO()
            the implementation for PPO algorithm
        train_A2C()
            the implementation for A2C algorithm
        train_DDPG()
            the implementation for DDPG algorithm
        train_TD3()
            the implementation for TD3 algorithm
        train_SAC()
            the implementation for SAC algorithm
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    @gin.configurable
    def train_eval(
        self,
        root_dir,
        py_env,
        # tf_agent,
        random_seed=None,
        # TODO(b/127576522): rename to policy_fc_layers.
        actor_fc_layers=(200, 100),
        value_fc_layers=(200, 100),
        use_rnns=False,
        lstm_size=(20,),
        # Params for collect
        num_environment_steps=25000000,
        collect_episodes_per_iteration=30,
        num_parallel_environments=5,
        replay_buffer_capacity=1001,  # Per-environment
        # Params for train
        num_epochs=25,
        learning_rate=1e-3,
        # Params for eval
        num_eval_episodes=30,
        eval_interval=500,
        # Params for summaries and logging
        train_checkpoint_interval=500,
        policy_checkpoint_interval=500,
        log_interval=50,
        summary_interval=50,
        summaries_flush_secs=1,
        use_tf_functions=True,
        debug_summaries=False,
        summarize_grads_and_vars=False,
    ):
        """A simple train and eval for PPO."""
        if root_dir is None:
            raise AttributeError("train_eval requires a root_dir.")

        root_dir = os.path.expanduser(root_dir)
        train_dir = os.path.join(root_dir, "train")
        eval_dir = os.path.join(root_dir, "eval")
        saved_model_dir = os.path.join(root_dir, "policy_saved_model")

        train_summary_writer = tf.compat.v2.summary.create_file_writer(
            train_dir, flush_millis=summaries_flush_secs * 1000
        )
        train_summary_writer.set_as_default()

        eval_summary_writer = tf.compat.v2.summary.create_file_writer(
            eval_dir, flush_millis=summaries_flush_secs * 1000
        )
        eval_metrics = [
            tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
        ]

        global_step = tf.compat.v1.train.get_or_create_global_step()
        with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)
        ):
            if random_seed is not None:
                tf.compat.v1.set_random_seed(random_seed)

            
            eval_py_env = py_env()
            eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)
            tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(
               [py_env] * num_parallel_environments))

            tf_agent=self.get_agent(tf_env)

            environment_steps_metric = tf_metrics.EnvironmentSteps()
            step_metrics = [
                tf_metrics.NumberOfEpisodes(),
                environment_steps_metric,
            ]

            train_metrics = step_metrics + [
                tf_metrics.AverageReturnMetric(batch_size=num_parallel_environments),
                tf_metrics.AverageEpisodeLengthMetric(batch_size=num_parallel_environments),
            ]

            eval_policy = tf_agent.policy
            collect_policy = tf_agent.collect_policy

            replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                tf_agent.collect_data_spec,
                batch_size=num_parallel_environments,
                max_length=replay_buffer_capacity,
            )

            train_checkpointer = common.Checkpointer(
                ckpt_dir=train_dir,
                agent=tf_agent,
                global_step=global_step,
                metrics=metric_utils.MetricsGroup(train_metrics, "train_metrics"),
            )
            policy_checkpointer = common.Checkpointer(
                ckpt_dir=os.path.join(train_dir, "policy"),
                policy=eval_policy,
                global_step=global_step,
            )
            saved_model = policy_saver.PolicySaver(eval_policy, train_step=global_step)

            train_checkpointer.initialize_or_restore()

            collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
                tf_env,
                collect_policy,
                observers=[replay_buffer.add_batch] + train_metrics,
                num_episodes=collect_episodes_per_iteration,
            )

                    
            # TODO delete
            def train_step_2():
                # Convert the replay buffer to a tf.data.Dataset and iterate through it
                dataset = replay_buffer.as_dataset(
                    sample_batch_size=30,
                        num_steps=64+1,
                        num_parallel_calls=1
                ).prefetch(3)
                
                iterator = iter(dataset)

                num_train_steps = 10

                # trajectories = []
                for _ in range(num_train_steps):
                    t, _ = next(iterator)
                    # trajectories.append(t)
                    loss_info = tf_agent.train(experience=t)
                return loss_info

            # TODO delete this or the other one
            def train_step():
                trajectories = replay_buffer.gather_all()
                return tf_agent.train(experience=trajectories)

            if use_tf_functions:
                # TODO(b/123828980): Enable once the cause for slowdown was identified.
                collect_driver.run = common.function(
                    collect_driver.run, autograph=False
                )
                tf_agent.train = common.function(tf_agent.train, autograph=False)
                train_step = common.function(train_step)

            collect_time = 0
            train_time = 0
            timed_at_step = global_step.numpy()

            while environment_steps_metric.result() < num_environment_steps:
                global_step_val = global_step.numpy()
                if global_step_val % eval_interval == 0:
                    metric_utils.eager_compute(
                        eval_metrics,
                        eval_tf_env,
                        eval_policy,
                        num_episodes=num_eval_episodes,
                        train_step=global_step,
                        summary_writer=eval_summary_writer,
                        summary_prefix="Metrics",
                    )

                start_time = time.time()
                collect_driver.run()
                collect_time += time.time() - start_time

                start_time = time.time()
                total_loss, _ = train_step()
                replay_buffer.clear()
                train_time += time.time() - start_time

                for train_metric in train_metrics:
                    train_metric.tf_summaries(
                        train_step=global_step, step_metrics=step_metrics
                    )

                if global_step_val % log_interval == 0:
                    logging.info("step = %d, loss = %f", global_step_val, total_loss)
                    steps_per_sec = (global_step_val - timed_at_step) / (
                        collect_time + train_time
                    )
                    logging.info("%.3f steps/sec", steps_per_sec)
                    logging.info(
                        "collect_time = %.3f, train_time = %.3f",
                        collect_time,
                        train_time,
                    )
                    with tf.compat.v2.summary.record_if(True):
                        tf.compat.v2.summary.scalar(
                            name="global_steps_per_sec",
                            data=steps_per_sec,
                            step=global_step,
                        )

                    if global_step_val % train_checkpoint_interval == 0:
                        train_checkpointer.save(global_step=global_step_val)

                    if global_step_val % policy_checkpoint_interval == 0:
                        policy_checkpointer.save(global_step=global_step_val)
                        saved_model_path = os.path.join(
                            saved_model_dir,
                            "policy_" + ("%d" % global_step_val).zfill(9),
                        )
                        saved_model.save(saved_model_path)

                    timed_at_step = global_step_val
                    collect_time = 0
                    train_time = 0

            # One final eval before exiting.
            metric_utils.eager_compute(
                eval_metrics,
                eval_tf_env,
                eval_policy,
                num_episodes=num_eval_episodes,
                train_step=global_step,
                summary_writer=eval_summary_writer,
                summary_prefix="Metrics",
            )

    @staticmethod
    def predict_trades(py_test_env):
        """make a prediction"""
        # load envoirement
        pred_py_env = py_test_env()
        pred_tf_env = tf_py_environment.TFPyEnvironment(pred_py_env)
        # load policy
        policy_path = os.path.join("trained_models/policy_saved_model/policy_000000000")
        policy = tf.saved_model.load(policy_path)
        # account_memory = []
        # actions_memory = []
        transitions = []
        time_step = pred_tf_env.reset()
        while not time_step.is_last():
            policy_step = policy.action(time_step)
            time_step = pred_tf_env.step(policy_step.action)
        if time_step.is_last():
            account_memory = pred_py_env.save_asset_memory()
            actions_memory = pred_py_env.save_action_memory()
            # account_memory = tf_test_env.env_method(method_name="save_asset_memory")
            # actions_memory = tf_test_env.env_method(method_name="save_action_memory")
            # transitions.append([time_step, policy_step])
        # return account_memory[0], actions_memory[0]
        return account_memory, actions_memory

    def create_networks(
        self, train_eval_tf_env, use_rnns, actor_fc_layers, value_fc_layers, lstm_size
    ):
        if use_rnns:
            actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                train_eval_tf_env.observation_spec(),
                train_eval_tf_env.action_spec(),
                input_fc_layer_params=actor_fc_layers,
                output_fc_layer_params=None,
                lstm_size=lstm_size,
            )
            value_net = value_rnn_network.ValueRnnNetwork(
                train_eval_tf_env.observation_spec(),
                input_fc_layer_params=value_fc_layers,
                output_fc_layer_params=None,
            )
        else:
            actor_net = actor_distribution_network.ActorDistributionNetwork(
                train_eval_tf_env.observation_spec(),
                train_eval_tf_env.action_spec(),
                fc_layer_params=actor_fc_layers,
                activation_fn=tf.keras.activations.tanh,
            )
            value_net = value_network.ValueNetwork(
                train_eval_tf_env.observation_spec(),
                fc_layer_params=value_fc_layers,
                activation_fn=tf.keras.activations.tanh,
            )
        return actor_net, value_net

    def get_agent(
        self,
        tf_env,
        # TODO(b/127576522): rename to policy_fc_layers.
        actor_fc_layers=(200, 100),
        value_fc_layers=(200, 100),
        use_rnns=False,
        lstm_size=(20,),
        # Params for train
        num_epochs=25,
        learning_rate=1e-3,
        # Params for summaries and logging
        debug_summaries=False,
        summarize_grads_and_vars=False,
    ):
        """An agent for PPO."""
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        actor_net, value_net = self.create_networks(
            tf_env, use_rnns, actor_fc_layers, value_fc_layers, lstm_size
        )
        tf_agent = ppo_agent.PPOAgent(
			tf_env.time_step_spec(),
			tf_env.action_spec(),
			optimizer,
			actor_net=actor_net,
			value_net=value_net,
			num_epochs=num_epochs,
			gradient_clipping=0.5,
			entropy_regularization=1e-2,
			importance_ratio_clipping=0.2,
			use_gae=True,
			use_td_lambda_return=True
		)
        tf_agent.initialize()
        return tf_agent
