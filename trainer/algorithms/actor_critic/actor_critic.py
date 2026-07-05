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
        """
        Constructor for the Actor-Critic class.
        Stores the provided model and training parameters, then builds and compiles the underlying
        Keras model, using the shared checkers network architecture for checkers and the shared simple
        network architecture for the other games

        :param parameters: parameters for the Actor-Critic model (batch size, training iterations, kernels,
                           regularization, learning rate and learning rate cycle step size)
        :param input_shape: shape of the model's input (refactored game state)
        :param output_shape: shape of the model's output (action space)
        :param env_name: name of the environment the model is being trained for
        :param training_mode: STEP - train every step, ITERATION - train every N episodes
        """

        # ENV NAME
        self._env_name = env_name

        # TRAINING MODE: STEP - TRAIN EVERY STEP, ITERATION - TRAIN EVERY N EPISODES
        self._training_mode = training_mode

        # DIMENSIONS - STATE SHAPE AND ACTION SHAPE
        self._input_shape = input_shape
        self._output_shape = output_shape

        # MODEL PARAMETERS
        self._batch_size = parameters["BATCH_SIZE"]
        self._training_iterations = parameters["TRAINING_ITERATIONS"]
        self._num_kernels = parameters["NUM_KERNELS"]

        # CONV2D AND DENSE REGULARIZATION PARAMETERS
        self._conv_regularization = L2(parameters['CONV_REGULARIZATION'])
        self._dense_regularization = L2(parameters['DENSE_REGULARIZATION'])

        # LEARNING RATE CYCLE CALCULATIONS
        self._learning_rate = parameters["LEARNING_RATE"]
        self._step_size = parameters["STEP_SIZE"]
        self._training_iterations_count = self._training_iterations

        if self._env_name == "CHECKERS":
            self._model = build_shared_model_checkers(self._input_shape, self._output_shape, self._num_kernels,
                                                     self._conv_regularization, self._dense_regularization)
        else:
            self._model = build_shared_model_simple(self._input_shape, self._output_shape, self._num_kernels,
                                                   self._conv_regularization, self._dense_regularization)
        self._model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                           optimizer=tf.keras.optimizers.Adam(self._learning_rate), loss_weights=[1, 0.01])

    def _predict(self, observations):
        """
        Run the model on input to predict state action probabilities and state value

        :param observations: model input, data received from the environment
        :return: action probabilities, value (model predictions)
        """

        return predict(self._model, observations)

    def set_weights(self, model_weights):
        """
        Set the weights of the model's neurons to the specified weights

        :param model_weights: new weights of the model
        """

        self._model.set_weights(model_weights)

    def get_weights(self):
        """
        Get the weights of the model's neurons

        :return: weights of the model
        """

        return self._model.get_weights()

    def _update_dataset(self, dataset, size):
        """
        TO DO

        :param dataset:
        :param size:
        :return:
        """

        dataset = dataset.batch(self._batch_size)
        dataset = dataset.shuffle(size)
        dataset = dataset.cache()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def load_model(self, filename):
        """
        Load the model from the specified file

        :param filename: path to the model file
        """

        self._model = tf.keras.models.load_model(filename, compile=True)
    
    def train(self, buffer):
        """
        TO DO

        :param buffer: buffer with the generated training data
        :return: the average loss for the training cycle
        """

        loss_log = np.zeros(self._training_iterations, dtype=np.float32)
        if self._training_mode == "STEP":
            state_batch, action_probs_batch, reward_batch = buffer.get_batch()
            iteration_history = self._model.fit(x=state_batch, y=[action_probs_batch, reward_batch],
                                               batch_size=self._batch_size,
                                               epochs=self._training_iterations, verbose=0)
            for training_iteration in range(self._training_iterations):
                loss_log[training_iteration] = iteration_history[training_iteration].history["loss"][training_iteration]

        elif self._training_mode == "ITERATION":
            dataset_training, size = buffer.create_dataset()
            dataset_training = self._update_dataset(dataset_training, size)
            iteration_history = self._model.fit(dataset_training, batch_size=self._batch_size,
                                               epochs=self._training_iterations, verbose=0)
            for training_iteration in range(self._training_iterations):
                loss_log[training_iteration] = iteration_history.history["loss"][training_iteration]
        return np.mean(loss_log)

    def save(self, steps, win_rate):
        """
        Save model for later use

        :param steps: the number of environment steps performed during reinforcement learning
        :param win_rate: win rate the model has achieved in the last test
        """

        today = date.today()
        model_name = f"actor-critic-{self._env_name}-month-{today.month}-day-{today.day}-ep-{steps}-{int(win_rate * 100)}%"
        self._model.save(f'saved_models/{model_name}.h5')

    def save_self_play(self):
        """
        Save model in the path specified for the data generation phase of reinforcement learning,
        to be later loaded during data generation
        """

        model_name = f"actor-critic-self_play"
        self._model.save(f'saved_models/data_generation_models/{model_name}.h5')

    def save_tournament(self, target=False):
        """
        Save model in the path specified for the tournament phase of reinforcement learning,
        to be later loaded for performing the tournament

        :param target: flag whether the model is currently trained or is an old (target) model
        """

        model_name = f"actor-critic-tournament-target-{target}"
        self._model.save(f'saved_models/tournament_models/{model_name}.h5')

    def save_test(self):
        """
        Save model in the path specified for the testing phase of reinforcement learning,
        to be later loaded for performing model tests
        """

        model_name = f"actor-critic-test"
        self._model.save(f'saved_models/test_models/{model_name}.h5')
