import tensorflow as tf


@tf.function
def predict(model, observations):
    """
    TO DO

    :param model:
    :param observations:
    :return:
    """

    action_probs, value = model(observations)
    return action_probs, value


@tf.function
def update_model(state_batch, action_probs_batch, reward_batch, model, optimizer):
    """
    TO DO

    :param state_batch:
    :param action_probs_batch:
    :param reward_batch:
    :param model:
    :param optimizer:
    :return:
    """

    with tf.GradientTape() as tape:
        loss_function_actor = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        loss_function_critic = tf.keras.losses.MeanSquaredError()
    
        actor_probs, critic_value = model(state_batch)
        actor_loss = loss_function_actor(action_probs_batch, actor_probs)
        critic_loss = loss_function_critic(reward_batch, critic_value) * 0.01
        loss = actor_loss + critic_loss

    gradients = tape.gradient(loss, tape.watched_variables())
    optimizer.apply_gradients(zip(gradients, tape.watched_variables()))
    return loss
