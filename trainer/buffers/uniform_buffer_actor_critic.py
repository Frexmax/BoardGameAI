import numpy as np
import tensorflow as tf
import pickle
from datetime import date


class UniformBufferActorCritic:
    def __init__(self, state_shape, action_shape, buffer_capacity, batch_size):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.buffer_capacity = buffer_capacity
        self.buffer_counter = 0
        self.batch_size = batch_size

        self.state_buffer = np.zeros((self.buffer_capacity, *state_shape), dtype=np.float32)
        self.action_probs_buffer = np.zeros((self.buffer_capacity, self.action_shape), dtype=np.float32)
        self.reward_buffer = np.zeros(self.buffer_capacity, dtype=np.float32)

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_probs_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.buffer_counter += 1

    def get_batch(self):
        # GET RANDOM BATCH
        index_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(index_range, self.batch_size)

        # CONVERT BUFFERS TO TENSORS
        state_tensor = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_probs_tensor = tf.convert_to_tensor(self.action_probs_buffer[batch_indices])
        reward_tensor = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        return state_tensor, action_probs_tensor, reward_tensor

    def create_dataset(self):
        # CONVERT BUFFERS TO TF DATASET
        end_index = min(self.buffer_capacity, self.buffer_counter)
        state_tensor = tf.convert_to_tensor(self.state_buffer[0:end_index])
        action_probs_tensor = tf.convert_to_tensor(self.action_probs_buffer[0:end_index])
        reward_tensor = tf.convert_to_tensor(self.reward_buffer[0:end_index])

        dataset_state = tf.data.Dataset.from_tensor_slices(state_tensor)
        dataset_labels = tf.data.Dataset.from_tensor_slices((action_probs_tensor, reward_tensor))
        dataset_training = tf.data.Dataset.zip((dataset_state, dataset_labels))
        return dataset_training, end_index

    def save(self):
        today = date.today()
        elements = np.min(self.buffer_capacity, self.buffer_counter)
        file_name = f"buffer-uniform-month-{today.month}-day-{today.day}-elements-{elements}"
        with open(f"../Trainer/Buffers/SavedBuffers/{file_name}.pkl", 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
