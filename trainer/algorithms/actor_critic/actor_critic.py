import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

from datetime import date

import tensorflow as tf
import numpy as np
from keras.src.regularizers import L2

from trainer.algorithms.actor_critic.actor_critic_trainers import predict
from trainer.algorithms.build_networks import build_shared_model_simple, build_shared_model_checkers


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
        """
        Run the model on input to predict state action probabilities and state value

        :param observations: model input, data received from the environment
        :return: action probabilities, value (model predictions)
        """

        return predict(self.model, observations)

    def set_weights(self, model_weights):
        """
        Set the weights of the model's neurons to the specified weights

        :param model_weights: new weights of the model
        """

        self.model.set_weights(model_weights)

    def get_weights(self):
        """
        Get the weights of the model's neurons

        :return: weights of the model
        """

        return self.model.get_weights()

    def update_dataset(self, dataset, size):
        """
        TO DO

        :param dataset:
        :param size:
        :return:
        """

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.shuffle(size)
        dataset = dataset.cache()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def load_model(self, filename):
        """
        Load the model from the specified file

        :param filename: path to the model file
        """

        self.model = tf.keras.models.load_model(filename, compile=True)
    
    def train(self, buffer):
        """
        TO DO

        :param buffer: buffer with the generated training data
        :return: the average loss for the training cycle
        """

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
        """
        Save model for later use

        :param steps: the number of environment steps performed during reinforcement learning
        :param win_rate: win rate the model has achieved in the last test
        """

        today = date.today()
        model_name = f"actor-critic-{self.env_name}-month-{today.month}-day-{today.day}-ep-{steps}-{int(win_rate * 100)}%"
        self.model.save(f'saved_models/{model_name}.h5')

    def save_self_play(self):
        """
        Save model in the path specified for the data generation phase of reinforcement learning,
        to be later loaded during data generation
        """

        model_name = f"actor-critic-self_play"
        self.model.save(f'saved_models/data_generation_models/{model_name}.h5')

    def save_tournament(self, target=False):
        """
        Save model in the path specified for the tournament phase of reinforcement learning,
        to be later loaded for performing the tournament

        :param target: flag whether the model is currently trained or is an old (target) model
        """

        model_name = f"actor-critic-tournament-target-{target}"
        self.model.save(f'saved_models/tournament_models/{model_name}.h5')

    def save_test(self):
        """
        Save model in the path specified for the testing phase of reinforcement learning,
        to be later loaded for performing model tests
        """

        model_name = f"actor-critic-test"
        self.model.save(f'saved_models/test_models/{model_name}.h5')
