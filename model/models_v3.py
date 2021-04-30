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

"""Train and Eval PPO.
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
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory


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
        def compute_avg_return(environment, policy, num_episodes=10, policy_state=()):
            total_return = 0.0
            for _ in range(num_episodes):

                time_step = environment.reset()
                episode_return = 0.0

                while not time_step.is_last():
                    if policy_state:
                        action_step = policy.action(time_step, policy_state)
                        policy_state = action_step.state
                    else:
                        action_step = policy.action(time_step)

                    time_step = environment.step(action_step.action)
                    episode_return += time_step.reward
                total_return += episode_return

            avg_return = total_return / num_episodes
            return avg_return.numpy()[0]


        def collect_step(environment, policy, buffer, id, policy_state):
            time_step = environment.current_time_step()
            if policy_state:
                action_step = policy.action(time_step, policy_state)
                policy_state = action_step.state
            else:
                action_step = policy.action(time_step)

            next_time_step = environment.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)

            id_tensor = tf.constant(id, dtype=tf.int64)
            buffer.add_batch(traj, id_tensor)
            if time_step.is_last():
                id[0] += 1

            return policy_state

        def collect_data(env, policy, buffer, steps, id, policy_state=()):
            for _ in range(steps):
                policy_state = collect_step(env, policy, buffer, id, policy_state)

            return policy_state
            

        tf.compat.v1.enable_v2_behavior()

        num_iterations = 100000 
        collect_steps_per_iteration = 1  
        initial_collect_steps = 100  
        replay_buffer_max_length = 100000  
        batch_size = 55 
        learning_rate = 1e-4 


        log_interval = 200 

        num_eval_episodes = 1  
        eval_interval = 1000  

        eval_py_env = py_env()
        train_py_env = py_env()
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
        train_env=tf_py_environment.TFPyEnvironment(train_py_env)


        random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                        train_env.action_spec())

        agent = self.get_agent(train_env)
        replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
            data_spec=agent.collect_data_spec,
            capacity = 1000,
            completed_only = True)

        episode_id = [0]
        collect_data(train_env, random_policy, replay_buffer, initial_collect_steps, episode_id)

        # Dataset generates trajectories with shape [Bx2x...]
        dataset = replay_buffer.as_dataset(
            sample_batch_size=None,
            num_steps=None)

        iterator = iter(dataset)

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        agent.train = common.function(agent.train)

        # Reset the train step
        agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        policy_state = agent.policy.get_initial_state(batch_size=train_env.batch_size)
        avg_return = 0
        returns = [avg_return]

        print('step = {0}: Average Return = {1}'.format(0, avg_return))

        collect_policy_state = agent.collect_policy.get_initial_state(batch_size=train_env.batch_size)

        step = 0
        train_loss = None
        for _ in range(num_iterations):

            # Collect a few steps using collect_policy and save to the replay buffer.
            collect_policy_state = collect_data(train_env,
                                            agent.collect_policy,
                                            replay_buffer,
                                            collect_steps_per_iteration,
                                            episode_id,
                                            collect_policy_state)


            # Sample a batch of data from the buffer and update the agent's network.

            step += 1
                
            if step % 150 == 0:
                experience, unused_info = next(iterator)
                train_loss = agent.train(experience).loss


            if step % log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % eval_interval == 0:
                policy_state = agent.policy.get_initial_state(batch_size=train_env.batch_size)
                avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes, policy_state)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)
            
                
        iterations = range(0, num_iterations + 1, eval_interval)

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
        train_step_counter = tf.Variable(0)
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
            train_step_counter = train_step_counter
		)
        tf_agent.initialize()
        return tf_agent
