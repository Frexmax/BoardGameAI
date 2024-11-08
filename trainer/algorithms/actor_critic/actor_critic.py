import sys
import path

sys.path.append(path.Path("algorithms").absolute())

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

from datetime import date

import tensorflow as tf
import numpy as np

from actor_critic_trainers import predict
from keras.src.regularizers import L2

from build_networks.build_networks import build_shared_model_simple, build_shared_model_checkers


class ActorCritic:
    def __init__(self, parameters, input_shape, output_shape, env_name, training_mode):
        # ENV NAME
        self.env_name = env_name

        # TRAINING MODE: STEP - TRAIN EVERY STEP, ITERATION - TRAIN EVERY N EPISODES
        self.training_mode = training_mode

        # DIMENSIONS - STATE SHAPE AND ACTION SHAPE
        self.input_shape = input_shape
        self.output_shape = output_shape

        # MODEL PARAMETERS
        self.batch_size = parameters["BATCH_SIZE"]
        self.training_iterations = parameters["TRAINING_ITERATIONS"]
        self.num_kernels = parameters["NUM_KERNELS"]

        # CONV2D AND DENSE REGULARIZATION PARAMETERS
        self.conv_regularization = L2(parameters['CONV_REGULARIZATION'])
        self.dense_regularization = L2(parameters['DENSE_REGULARIZATION'])

        # LEARNING RATE CYCLE CALCULATIONS
        self.learning_rate = parameters["LEARNING_RATE"]
        self.step_size = parameters["STEP_SIZE"]
        self.training_iterations_count = self.training_iterations

        if self.env_name == "CHECKERS":
            self.model = build_shared_model_checkers(self.input_shape, self.output_shape, self.num_kernels,
                                                     self.conv_regularization, self.dense_regularization)
        else:
            self.model = build_shared_model_simple(self.input_shape, self.output_shape, self.num_kernels,
                                                   self.conv_regularization, self.dense_regularization)
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                           optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss_weights=[1, 0.01])

    def predict(self, observations):
        return predict(self.model, observations)

    def set_weights(self, model_weights):
        self.model.set_weights(model_weights)

    def get_weights(self):
        return self.model.get_weights()

    def update_dataset(self, dataset, size):
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.shuffle(size)
        dataset = dataset.cache()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def load_model(self, filename):
        self.model = tf.keras.models.load_model(filename, compile=True)
    
    def train(self, buffer):
        loss_log = np.zeros(self.training_iterations, dtype=np.float32)
        if self.training_mode == "STEP":
            state_batch, action_probs_batch, reward_batch = buffer.get_batch()
            iteration_history = self.model.fit(x=state_batch, y=[action_probs_batch, reward_batch],
                                               batch_size=self.batch_size,
                                               epochs=self.training_iterations, verbose=0)
            for training_iteration in range(self.training_iterations):
                loss_log[training_iteration] = iteration_history[training_iteration].history["loss"][training_iteration]

        elif self.training_mode == "ITERATION":
            dataset_training, size = buffer.create_dataset()
            dataset_training = self.update_dataset(dataset_training, size)
            iteration_history = self.model.fit(dataset_training, batch_size=self.batch_size,
                                               epochs=self.training_iterations, verbose=0)
            for training_iteration in range(self.training_iterations):
                loss_log[training_iteration] = iteration_history.history["loss"][training_iteration]
        return np.mean(loss_log)

    def save(self, steps, win_rate):
        today = date.today()
        model_name = f"actor-critic-{self.env_name}-month-{today.month}-day-{today.day}-ep-{steps}-{int(win_rate * 100)}%"
        self.model.save(f'SavedModels/{model_name}.h5')

    def save_self_play(self):
        model_name = f"actor-critic-self_play"
        self.model.save(f'SavedModels/DataGenerationModels/{model_name}.h5')

    def save_tournament(self, target=False):
        model_name = f"actor-critic-tournament-target-{target}"
        self.model.save(f'SavedModels/TournamentModels/{model_name}.h5')

    def save_test(self):
        model_name = f"actor-critic-test"
        self.model.save(f'SavedModels/TestModels/{model_name}.h5')
