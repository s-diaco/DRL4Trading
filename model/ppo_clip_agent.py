import tensorflow as tf
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.networks import (actor_distribution_network,
                                actor_distribution_rnn_network, value_network,
                                value_rnn_network)


def create_networks(train_eval_tf_env, use_rnns, actor_fc_layers, value_fc_layers, lstm_size
                    ):
    # TODO replace with one "fc_layer_params" in the last version
    if use_rnns:
        actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
            train_eval_tf_env.observation_spec(),
            train_eval_tf_env.action_spec(),
            input_fc_layer_params=actor_fc_layers,
            output_fc_layer_params=actor_fc_layers,
            lstm_size=lstm_size,
        )
        value_net = value_rnn_network.ValueRnnNetwork(
            train_eval_tf_env.observation_spec(),
            input_fc_layer_params=value_fc_layers,
            output_fc_layer_params=value_fc_layers,
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
    env,
    # TODO test these values from stable baselines with batch size = 1024 and num_steps 256
    # should be (1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)
    actor_fc_layers=(200, 100),
    # should be (1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)
    value_fc_layers=(200, 100),
    use_rnns=False,
    lstm_size=(20,),
    # Params for train
    num_epochs=10,
    learning_rate=5e-06,
    discount_factor=0.99
):
    """
    An agent for PPO.
    """

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    actor_net, value_net = create_networks(
        env, use_rnns, actor_fc_layers, value_fc_layers, lstm_size
    )

    # dtype arg. is not used in any tutorial but my code doesn't work without this
    train_step_counter = tf.Variable(0, dtype='int64')

    tf_agent = ppo_clip_agent.PPOClipAgent(
        env.time_step_spec(),
        env.action_spec(),
        optimizer,
        actor_net=actor_net,
        value_net=value_net,
        train_step_counter=train_step_counter,
        discount_factor=discount_factor,
        num_epochs=num_epochs,
        # gradient_clipping=0.5,
        # entropy_regularization=1e-2,
        # importance_ratio_clipping=0.2,
        # use_gae=True,
        # use_td_lambda_return=True,
    )
    tf_agent.initialize()
    return tf_agent
