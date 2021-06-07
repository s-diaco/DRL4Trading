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

import logging
import os
import time

import gin
import tensorflow as tf
from halo import Halo
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment, batched_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
# from tf_agents.drivers import dynamic_step_driver
# from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.replay_buffers import episodic_replay_buffer
# from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.utils import common
from .ppo_clip_agent import get_agent


class TradeDRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: environment class
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
        predict_trades()
            make a prediction in a test dataset and get results
    """

    def _get_model_dirs(self, root_dir):
        root_dir = os.path.expanduser(root_dir)
        train_dir = os.path.join(root_dir, "train")
        eval_dir = os.path.join(root_dir, "eval")
        saved_model_dir = os.path.join(root_dir, "policy_saved_model")
        return train_dir, eval_dir, saved_model_dir

    def _eval_model(
        self,
        eval_metrics,
        eval_env,
        eval_policy,
        num_episodes,
        train_step,
        summary_writer,
        summary_prefix
    ):
        logging.info(f'start eval')
        metric_utils.eager_compute(
            eval_metrics,
            eval_env,
            eval_policy,
            num_episodes=num_episodes,
            train_step=train_step,
            summary_writer=summary_writer,
            summary_prefix=summary_prefix,
        )
        logging.info(f'end of eval')

    def _save_policy(
        self,
        eval_policy,
        global_step,
        saved_model_dir,
        step_metrics,
        is_complete=False
    ):
        if is_complete:
            saved_model_path = os.path.join(
                saved_model_dir,
                "complete_policy",
            )
        else:
            saved_model_path = os.path.join(
                saved_model_dir,
                "policy_" +
                ("%d" %
                    step_metrics[1].environment_steps.numpy()).zfill(9),
            )
        saved_model = policy_saver.PolicySaver(
            eval_policy, train_step=global_step)
        saved_model.save(saved_model_path)

    @gin.configurable
    def train_PPO(
        self,
        py_env,
        root_dir = "./trained_models",
        random_seed=None,
        # Params for collect
        # num_environment_steps=25000000,
        collect_episodes_per_iteration=1,
        num_parallel_environments=1,
        replay_buffer_capacity=100000,  # Per-environment
        # Params for eval
        num_eval_episodes=2,
        eval_interval=5,
        # Params for summaries and logging
        train_checkpoint_interval=5,
        policy_checkpoint_interval=5,
        log_interval=5,
        summary_interval=5,
        summaries_flush_secs=1,
        use_tf_functions=True,
        num_iterations=10,
        use_parallel_envs=False
    ):
        """A train and eval for PPO."""

        if root_dir is None:
            raise AttributeError("train_eval requires a root_dir.")

        train_dir, eval_dir, saved_model_dir = self._get_model_dirs(root_dir)

        train_summary_writer = tf.summary.create_file_writer(
            train_dir, flush_millis=summaries_flush_secs * 1000
        )
        train_summary_writer.set_as_default()

        eval_summary_writer = tf.summary.create_file_writer(
            eval_dir, flush_millis=summaries_flush_secs * 1000
        )
        eval_metrics = [
            tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
            tf_metrics.AverageEpisodeLengthMetric(
                buffer_size=num_eval_episodes),
        ]

        # Create environment
        # replacing 'parallel_py_environment.ParallelPyEnvironment'
        # with      'batched_py_environment.BatchedPyEnvironment' 
        # to avoid using multi-process and use multi-thread instead. Although it would be slower
        if num_parallel_environments > 1:
            if use_parallel_envs:
                train_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(
                    [py_env] * num_parallel_environments))

            else:
                train_py_env = py_env()
                batched_py_env = batched_py_environment.BatchedPyEnvironment([
                    train_py_env
                    for _ in range(num_parallel_environments)
                ])
                train_env = tf_py_environment.TFPyEnvironment(batched_py_env)
        else:
            train_py_env = py_env()
            train_env = tf_py_environment.TFPyEnvironment(train_py_env)

        eval_py_env = py_env()
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

        tf_agent = get_agent(train_env)
        global_step = tf_agent.train_step_counter

        with tf.summary.record_if(
            lambda: tf.math.equal(
                global_step/tf_agent._num_epochs % summary_interval, 0)
        ):
            if random_seed is not None:
                tf.random.set_seed(random_seed)

            eval_policy = tf_agent.policy
            collect_policy = tf_agent.collect_policy

            environment_steps_metric = tf_metrics.EnvironmentSteps()
            step_metrics = [
                tf_metrics.NumberOfEpisodes(),
                environment_steps_metric,
            ]
            train_metrics = step_metrics + [
                tf_metrics.AverageReturnMetric(
                    batch_size=train_env.batch_size
                ),
                tf_metrics.AverageEpisodeLengthMetric(
                    batch_size=train_env.batch_size
                ),
            ]

            train_checkpointer = common.Checkpointer(
                ckpt_dir=train_dir,
                agent=tf_agent,
                global_step=global_step,
                metrics=metric_utils.MetricsGroup(
                    train_metrics, "train_metrics"),
            )
            policy_checkpointer = common.Checkpointer(
                ckpt_dir=os.path.join(train_dir, "policy"),
                policy=eval_policy,
                global_step=global_step,
            )

            train_checkpointer.initialize_or_restore()

            replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
                tf_agent.collect_data_spec,
                capacity=replay_buffer_capacity,
                completed_only=True
            )

            stateful_buffer = episodic_replay_buffer.StatefulEpisodicReplayBuffer(
                replay_buffer,
                train_env.batch_size
            )

            replay_observer = [stateful_buffer.add_batch]

            # TODO it has a bias toward shorter episodes in parallel envs 
            # (more info: tf_agents/drivers/dynamic_episode_driver.py) 
            # or just use one paralleled or batched env per iteration.
            collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
                train_env,
                collect_policy,
                observers=replay_observer + train_metrics,
                num_episodes=collect_episodes_per_iteration,
            )

            def train_step():
                dataset = replay_buffer.as_dataset(
                    # sample_batch_size=1024,
                    # num_steps=256,
                    single_deterministic_pass=True
                )
                iterator = iter(dataset)
                for _ in range(collect_episodes_per_iteration):
                    trajectories = next(iterator)
                    batched_traj = tf.nest.map_structure(
                        # lambda t: tf.reshape(t, [1, 256, t.get_shape().as_list()[0]])
                        lambda t: tf.expand_dims(t, axis=0),
                        trajectories
                    )
                    train_loss = tf_agent.train(experience=batched_traj)
                return train_loss

            if use_tf_functions:
                # TODO(b/123828980): Enable once the cause for slowdown was identified.
                collect_driver.run = common.function(
                    collect_driver.run, autograph=False
                )
                tf_agent.train = common.function(
                    tf_agent.train, autograph=False)
                train_step = common.function(train_step)

            collect_time = 0
            train_time = 0
            timed_at_step = global_step.numpy()

            # TODO what is this step metrics for?
            # while environment_steps_metric.result() < num_environment_steps:
            for _ in range(num_iterations):
                global_step_val = global_step.numpy()/(tf_agent._num_epochs*collect_episodes_per_iteration)
                if global_step_val % eval_interval == 0:
                    self._eval_model(
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
                collect_driver.run()
                collect_time += time.time() - start_time
                logging.info(f'collect ended')

                logging.info(f'start training')
                with Halo(text='Training the model', spinner='arrow3'):
                    start_time = time.time()
                    total_loss = train_step()
                    replay_buffer._clear(
                        clear_all_variables=True)
                    train_time += time.time() - start_time
                logging.info(f'train ended')

                for train_metric in train_metrics:
                    train_metric.tf_summaries(
                        train_step=global_step, step_metrics=step_metrics
                    )

                if global_step_val % log_interval == 0:
                    logging.info("step = %d, loss = %f",
                                 global_step_val, total_loss.loss)
                    steps_per_sec = (global_step.numpy() - timed_at_step) / (
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

                    timed_at_step = global_step_val
                    collect_time = 0
                    train_time = 0

                if global_step_val % train_checkpoint_interval == 0:
                    train_checkpointer.save(global_step=global_step)

                if global_step_val % policy_checkpoint_interval == 0:
                    policy_checkpointer.save(global_step=global_step)
                    self._save_policy(
                        eval_policy=eval_policy,
                        global_step=global_step,
                        saved_model_dir=saved_model_dir,
                        step_metrics=step_metrics
                    )

            # One final eval before exiting.
            self._eval_model(
                eval_metrics,
                eval_env,
                eval_policy,
                num_episodes=num_eval_episodes,
                train_step=global_step,
                summary_writer=eval_summary_writer,
                summary_prefix="Metrics",
            )

            # save the final policy
            self._save_policy(
                eval_policy=eval_policy,
                global_step=global_step,
                saved_model_dir=saved_model_dir,
                step_metrics=step_metrics,
                is_complete=True
            )

    @staticmethod
    def predict_trades(py_test_env):
        """make a prediction"""

        # load envoirement
        pred_py_env = py_test_env()
        pred_tf_env = tf_py_environment.TFPyEnvironment(pred_py_env)

        # load policy
        policy_path = os.path.join(
            "trained_models/policy_saved_model/complete_policy")
        policy = tf.saved_model.load(policy_path)
        # account_memory = []
        # actions_memory = []
        # transitions = []
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

    def predict_single_day(self, date):
        """make a prediction for specified date"""

        #TODO under construction! do not use this one.

        # set dates needed to calculate indicators
        # last_day = date_time.today()
        # first_day = date_time.today()

        # load envoirement
        pred_py_env = py_test_env()
        pred_tf_env = tf_py_environment.TFPyEnvironment(pred_py_env)

        # load policy
        policy_path = os.path.join(
            "trained_models/policy_saved_model/complete_policy")
        policy = tf.saved_model.load(policy_path)
        time_step = pred_tf_env.reset()
        policy_step = policy.action(time_step)
        return policy_step

