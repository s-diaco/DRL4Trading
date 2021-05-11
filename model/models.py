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
from tf_agents.replay_buffers import episodic_replay_buffer
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.utils import common
from halo import Halo


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
        collect_episodes_per_iteration=100,
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
        policy_checkpoint_interval=5,
        log_interval=50,
        summary_interval=50,
        summaries_flush_secs=1,
        use_tf_functions=True,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        num_iterations = 10
    ):
        """A simple train and eval for PPO."""

        # params from sachag678/Reinforcement_learning/blob/master/tf-agents-example/simulate.py

        initial_collect_steps = 1000  # @param
        collect_steps_per_iteration = 1  # @param
        replay_buffer_capacity = 100000  # @param

        batch_size = 128  # @param
        learning_rate = 1e-5  # @param
        log_interval = 200  # @param

        num_eval_episodes = 2  # @param
        eval_interval = 1000  # @param
        # end of params from sachag678...

        if root_dir is None:
            raise AttributeError("train_eval requires a root_dir.")

        root_dir = os.path.expanduser(root_dir)
        train_dir = os.path.join(root_dir, "train")
        eval_dir = os.path.join(root_dir, "eval")
        saved_model_dir = os.path.join(root_dir, "policy_saved_model")

        train_summary_writer = tf.summary.create_file_writer(
            train_dir, flush_millis=summaries_flush_secs * 1000
        )
        train_summary_writer.set_as_default()

        eval_summary_writer = tf.summary.create_file_writer(
            eval_dir, flush_millis=summaries_flush_secs * 1000
        )
        eval_metrics = [
            tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
        ]
        # TODO replace with tf_agent.train_step_counter.numpy()
        global_step = tf.compat.v1.train.get_or_create_global_step()
        with tf.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)
        ):
            if random_seed is not None:
                tf.random.set_seed(random_seed)

            
            # eval_py_env = py_env()
            # eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)
            train_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(
                [py_env] * num_parallel_environments))
               
            # train_py_env = py_env()
            # train_env=tf_py_environment.TFPyEnvironment(train_py_env)
            eval_py_env = py_env()
            eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

            tf_agent=self.get_agent(train_env)

            environment_steps_metric = tf_metrics.EnvironmentSteps()
            step_metrics = [
                tf_metrics.NumberOfEpisodes(),
                environment_steps_metric,
            ]

            train_metrics = step_metrics + [
                tf_metrics.AverageReturnMetric(),
                tf_metrics.AverageEpisodeLengthMetric(),
            ]

            eval_policy = tf_agent.policy
            collect_policy = tf_agent.collect_policy

            replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
                tf_agent.collect_data_spec,
                capacity = replay_buffer_capacity,
                completed_only = True
            )

            stateful_buffer = episodic_replay_buffer.StatefulEpisodicReplayBuffer(
                replay_buffer, train_env.batch_size)

            replay_observer = [stateful_buffer.add_batch]

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
                train_env,
                collect_policy,
                observers=replay_observer + train_metrics,
                num_episodes=collect_episodes_per_iteration,
            )
        
            def train_step():
                dataset = replay_buffer.as_dataset(
                    # sample_batch_size=300,
                    num_steps=2,
                    single_deterministic_pass=True
                )
                iterator = iter(dataset)
                trajectories = next(iterator)
                # batched_ds = tf.expand_dims(dataset, axis=1)
                batched_traj = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), trajectories)
                train_loss = tf_agent.train(experience=batched_traj)
                return train_loss

            def save_policy(saved_model, saved_model_dir, step_metrics):
                saved_model_path = os.path.join(
                    saved_model_dir,
                    "policy_" + ("%d" % step_metrics[1].environment_steps.numpy()).zfill(9),
                )
                saved_model.save(saved_model_path)

            if use_tf_functions:
                # TODO(b/123828980): Enable once the cause for slowdown was identified.
                collect_driver.run = common.function(
                    collect_driver.run, autograph=False
                )
                tf_agent.train = common.function(tf_agent.train, autograph=False)
                train_step = common.function(train_step)

            tf_agent.train_step_counter.assign(0)
            collect_time = 0
            train_time = 0
            timed_at_step = global_step.numpy()

            # TODO what is this step metrics for? 
            # while environment_steps_metric.result() < num_environment_steps:
            for _ in range(num_iterations):
                global_step_val = global_step.numpy()
                if global_step_val % eval_interval == 0:
                    metric_utils.eager_compute(
                        eval_metrics,
                        eval_env,
                        eval_policy,
                        num_episodes=num_eval_episodes,
                        train_step=global_step,
                        summary_writer=eval_summary_writer,
                        summary_prefix="Metrics",
                    )

                logging.info(f'start collecting data to replay buffer')
                start_time = time.time()
                final_time_step, _ = collect_driver.run()
                collect_time += time.time() - start_time
                logging.info(f'collect ended')
                logging.info(f'collect time: {collect_time}')



                logging.info(f'start training')
                with Halo(text='Training the model', spinner='dots'):
                    # Run time consuming work here
                    start_time = time.time()
                    total_loss = train_step()
                    step = tf_agent.train_step_counter.numpy()
                    clear_replay_op = replay_buffer.clear()
                    train_time += time.time() - start_time
                logging.info(f'train ended')
                logging.info(f'train time: {collect_time}')

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
                    with tf.summary.record_if(True):
                        tf.summary.scalar(
                            name="global_steps_per_sec",
                            data=steps_per_sec,
                            step=global_step,
                        )

                if global_step_val % train_checkpoint_interval == 0:
                    train_checkpointer.save(global_step=global_step_val)

                if global_step_val % policy_checkpoint_interval == 0:
                    policy_checkpointer.save(global_step=global_step_val)
                    save_policy(saved_model=saved_model,
                                saved_model_dir=saved_model_dir,
                                step_metrics=step_metrics
                                )

            # One final eval before exiting.
            metric_utils.eager_compute(
                eval_metrics,
                eval_env,
                eval_policy,
                num_episodes=num_eval_episodes,
                train_step=global_step,
                summary_writer=eval_summary_writer,
                summary_prefix="Metrics",
            )

            # save the final policy
            save_policy(saved_model=saved_model,
                        saved_model_dir=saved_model_dir,
                        step_metrics=step_metrics
                        )

    @staticmethod
    def predict_trades(py_test_env):
        """make a prediction"""
        # load envoirement
        pred_py_env = py_test_env()
        pred_tf_env = tf_py_environment.TFPyEnvironment(pred_py_env)
        # load policy
        # TODO load the best policy file
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
        # TODO check this parameters from the stable baselines implementation.
        """An agent for PPO.
        {'n_steps': 256, 'ent_coef': 0.0, 'learning_rate': 5e-06, 'batch_size': 1024, 'gamma': 0.99}
        UserWarning: You have specified a mini-batch size of 1024, but because the `RolloutBuffer` is of size `n_steps * n_envs = 256`, after every 0 untruncated mini-batches, there will be a truncated mini-batch of size 256
        We recommend using a `batch_size` that is a multiple of `n_steps * n_envs`.
        Info: (n_steps=256 and n_envs=1)
        """

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        actor_net, value_net = self.create_networks(
            tf_env, use_rnns, actor_fc_layers, value_fc_layers, lstm_size
        )

        # dtype arg. is not used in any tutorial but my code doesn't work without this
        train_step_counter = tf.Variable(0, dtype='int64')

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
			use_td_lambda_return=True,
            train_step_counter=train_step_counter
		)
        tf_agent.initialize()
        return tf_agent
