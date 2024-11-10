import pickle
from datetime import date

import numpy as np
import tensorflow as tf


class UniformBufferActorCritic:
    """
    Class implementing a training buffer with a uniform sampling policy for the reinforcement learning algorithm.
    It saves previous experiences to stabilize the training process
    """

    def __init__(self, state_shape, action_shape, buffer_capacity, batch_size):
        """
        Constructor for the Buffer class.
        Initializes the arrays for different samples info (state, action, reward)
        based on the provided environment info

        :param state_shape: dimensions of the game state (model input)
        :param action_shape: number of actions
        :param buffer_capacity: number of samples to store in the buffer
        :param batch_size: number of samples to get in one batch
        """

        self.buffer_counter = 0
        self.batch_size = batch_size

        # Save environment info
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.buffer_capacity = buffer_capacity

        # Initialize buffer arrays
        self.state_buffer = np.zeros((self.buffer_capacity, *state_shape), dtype=np.float32)
        self.action_probs_buffer = np.zeros((self.buffer_capacity, self.action_shape), dtype=np.float32)
        self.reward_buffer = np.zeros(self.buffer_capacity, dtype=np.float32)

    def record(self, obs_tuple):
        """
        Save the provided observation tuple obtained from the environment in the buffer,
        then increment the buffer size counter

        :param obs_tuple: tuple with environment information (state, action, reward)
        """

        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_probs_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]

        self.buffer_counter += 1

    def get_batch(self):
        """
        Get random batch of data from the buffer and convert it to tf tensors

        :return: 3 tensors of batch size (state, action, reward)
        """

        # Calculate the possible index range
        index_range = min(self.buffer_counter, self.buffer_capacity)

        # Get an array of 64 random indexes from the index range
        batch_indexes = np.random.choice(index_range, self.batch_size)

        # Create an array from the respective buffers and convert them to tf tensors
        state_tensor = tf.convert_to_tensor(self.state_buffer[batch_indexes])
        action_probs_tensor = tf.convert_to_tensor(self.action_probs_buffer[batch_indexes])
        reward_tensor = tf.convert_to_tensor(self.reward_buffer[batch_indexes])
        return state_tensor, action_probs_tensor, reward_tensor

    def create_dataset(self):
        """
        Create a tf dataset from the saved data
        for reinforcement learning training pipeline,

        :return: tf dataset, size of dataset
        """

        # Get the effective size of the buffer
        end_index = min(self.buffer_capacity, self.buffer_counter)

        # Convert buffers to tf tensors
        state_tensor = tf.convert_to_tensor(self.state_buffer[0:end_index])
        action_probs_tensor = tf.convert_to_tensor(self.action_probs_buffer[0:end_index])
        reward_tensor = tf.convert_to_tensor(self.reward_buffer[0:end_index])

        # Create tf dataset from tf tensors
        dataset_state = tf.data.Dataset.from_tensor_slices(state_tensor)
        dataset_labels = tf.data.Dataset.from_tensor_slices((action_probs_tensor, reward_tensor))
        dataset_training = tf.data.Dataset.zip((dataset_state, dataset_labels))
        return dataset_training, end_index

    def save(self):
        """
        Save the buffer as a pickle file
        The file name is determined by the number of elements in the buffer and the month/day of the saving
        """

        today = date.today()
        elements = np.min(self.buffer_capacity, self.buffer_counter)
        file_name = f"buffer-uniform-month-{today.month}-day-{today.day}-elements-{elements}"

        # Save the buffer with the specified file name
        with open(f"../trainer/buffers/saved_buffers/{file_name}.pkl", 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
