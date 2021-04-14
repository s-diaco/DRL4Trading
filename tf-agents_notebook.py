# %% [markdown]
#### todo:
# - devide main notebook to multiple smaller files
# - develope a better logging system
# %%
import os
from pprint import pprint
from finrl.model.models import DRLAgent
from env_tse.env_stocktrading_tse_stoploss import StockTradingEnvTSEStopLoss
from config import config
import datetime
import numpy as np
import pandas as pd
from IPython import get_ipython
import matplotlib
import matplotlib.pyplot as plt
import backtest_tse.backtesting_tse as backtest
import logging
from preprocess_tse_data import preprocess_data
import tf_agents
import tensorflow as tf
from tf_agents.drivers import dynamic_episode_driver; # data collection driver
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from tf_agents.environments.suite_gym import wrap_env
from tf_agents.environments import utils

tf.compat.v1.enable_v2_behavior()

logging.basicConfig(format="%(message)s", level=logging.INFO)
# %%
## Preprocess data
train, trade = preprocess_data()

# %%
## Create the envoriments
information_cols = ["daily_variance", "change", "log_volume"]

e_train_gym = StockTradingEnvTSEStopLoss(
    df=train,
    initial_amount=1e8,
    hmax=1e7,
    cache_indicator_data=False,
    daily_information_cols=information_cols,
    print_verbosity=500,
    buy_cost_pct=3.7e-3,
    sell_cost_pct=8.8e-3,
    cash_penalty_proportion=0
)

e_trade_gym = StockTradingEnvTSEStopLoss(
    df=trade,
    initial_amount=1e8,
    hmax=1e7,
    cache_indicator_data=False,
    daily_information_cols=information_cols,
    print_verbosity=500,
    discrete_actions=True,
    shares_increment=10,
    buy_cost_pct=3.7e-3,
    sell_cost_pct=8.8e-3,
    cash_penalty_proportion=0,
    patient=True,
    random_start=False,
)

# %% 
logging.info(f'TensorFlow version: {tf.version.VERSION}')
logging.info(f"List of available [GPU] devices:\n{tf.config.list_physical_devices('GPU')}")
num_iterations = 5 # 250 # @param {type:"integer"}
collect_steps_per_iteration = 2 # @param {type:"integer"}
replay_buffer_capacity = 2000 # @param {type:"integer"}

fc_layer_params = (100,)

learning_rate = 1e-3 # @param {type:"number"}
log_interval = 25 # @param {type:"integer"}
num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 100 # @param {type:"integer"}

train_py_env = wrap_env(e_train_gym)
eval_py_env = wrap_env(e_train_gym)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
# utils.validate_py_environment(train_env, episodes=2)

# %%
## Model
actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.compat.v2.Variable(0)

tf_agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)
tf_agent.initialize()

# Agents contain two policies:
# for evaluation/deployment 
eval_policy = tf_agent.policy
# for data collection
collect_policy = tf_agent.collect_policy

# %%
# The most common metric used to evaluate a policy is the average return.

# todo: remove to replace with tf_metrics.AverageReturnMetric
def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  # we usually average this over a few episodes
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

# collect_data_spec is a Trajectory named tuple containing the observation, action, reward etc.
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

# policy saver
saver = policy_saver.PolicySaver(tf_agent.policy)

# todo: remove to replace with train_driver
# collect an episode using the given data collection policy and save the data 
# (observations, actions, rewards etc.) as trajectories in the replay buffer.
def collect_episode(environment, policy, num_episodes):

  episode_counter = 0
  environment.reset()

  while episode_counter < num_episodes:
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)

    if traj.is_boundary():
      episode_counter += 1

# define trajectory collector
train_episode_count = tf_metrics.NumberOfEpisodes()
train_total_steps = tf_metrics.EnvironmentSteps()
train_avg_reward = tf_metrics.AverageReturnMetric(batch_size = train_env.batch_size)
train_avg_episode_len = tf_metrics.AverageEpisodeLengthMetric(batch_size = train_env.batch_size)
train_driver = dynamic_episode_driver.DynamicEpisodeDriver(
  train_env,
  tf_agent.collect_policy, # NOTE: use PPOPolicy to collect episode
  observers = [
    replay_buffer.add_batch,
    train_episode_count,
    train_total_steps,
    train_avg_reward,
    train_avg_episode_len
  ], # callbacks when an episode is completely collected
  num_episodes = collect_steps_per_iteration, # how many episodes are collected in an iteration
)
# %%
## Training
# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)
eval_avg_reward = tf_metrics.AverageReturnMetric(buffer_size = num_eval_episodes)
eval_avg_episode_len = tf_metrics.AverageEpisodeLengthMetric(buffer_size = num_eval_episodes)

# Evaluate the agent's policy once before training.
returns = [eval_avg_reward]

while train_total_steps.result() < num_iterations:

  # Collect a few episodes using train_driver and save to the replay buffer.
  train_driver.run()
  # todo: is deprecated
  # Use data from the buffer and update the agent's network.
  trajectories = replay_buffer.gather_all()
  train_loss, _ = tf_agent.train(experience=trajectories)
   # clear collected episodes right after training
  replay_buffer.clear()

  step = tf_agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step % eval_interval == 0:
    # save checkpoint
    saver.save('checkpoints/policy_%d' % step)
    # evaluate the updated policy
    eval_avg_reward.reset()
    eval_avg_episode_len.reset()
    eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(
      eval_env,
      tf_agent.policy,
      observers = [
        eval_avg_reward,
        eval_avg_episode_len,
      ],
      num_episodes = num_eval_episodes, # how many epsiodes are collected in an iteration
    )
    eval_driver.run()
    # todo: delete
    # avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    # print('step = {0}: Average Return = {1}'.format(step, avg_return))
    print('step = {0}: Average Return = {1} Average Episode Length = {2}'.format(step, train_avg_reward.result(), train_avg_episode_len.result()))
    returns.append(train_avg_reward)

# %% 
## plot
steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim(top=250)

# %%
## predict
num_episodes = 3
for _ in range(num_episodes):
  status = eval_env.reset() #time_step
  policy_state = tf_agent.policy.get_initial_state(eval_env.batch_size)
  while not status.is_last():
    # todo: use greedy policy to test
    action = tf_agent.policy.action(status, policy_state)
    status = eval_env.step(action.action)
    logging.info(f'Action: {action}')





# %%
# single processing
env_train, _ = e_train_gym.get_sb_env()

# this is our observation environment. It allows full diagnostics
env_trade, _ = e_trade_gym.get_sb_env()

# %% 6. Implement DRL Algorithms
agent = DRLAgent(env=env_train)

# %% Model PPO
# from torch.nn import Softsign, ReLU
ppo_params = {
    "n_steps": 256,
    "ent_coef": 0.0,
    "learning_rate": 0.000005,
    "batch_size": 1024,
    "gamma": 0.99,
}

policy_kwargs = {
    #     "activation_fn": ReLU,
    "net_arch": [1024 for _ in range(10)],
    #     "squash_output": True
}

model = agent.get_model(
    "ppo", model_kwargs=ppo_params, policy_kwargs=policy_kwargs, verbose=0
)

# %%
# model = model.load("trained_models/different4_7_2000.model", env = env_train)

# %%
model.learn(
    total_timesteps=10000,
    eval_env=env_trade,
    eval_freq=500,
    log_interval=1,
    tb_log_name="env_tse",
    n_eval_episodes=1,
)

# %%
# model.save("trained_models/tse4_10_1000.model")

# %% Trade
logging.info(f"Trade dates: {len(e_trade_gym.dates)}")

# %%
df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=model, environment=e_trade_gym
)

# %%
logging.info(f"Model actions:\n{df_actions.head()}")

# %%
logging.info(f"Account value data shape: {df_account_value.shape}:\n{df_account_value.head(10)}")

# %% 7. Backtest
backtest.backtest_tse_trades(df_account_value, "^TSEI", config.START_TRADE_DATE, config.END_DATE)
# %%
