import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

from tensorflow import device
from tensorflow.keras.saving import load_model

from trainer.algorithms.actor_critic.actor_critic_trainers import predict


class SelfPlayModel:
    """
    Class storing an Actor-Critic model
    """

    def __init__(self, file_name):
        """
        Load the model from the file

        :param file_name: path to the file where the self-play model is stored
        """

        self.model = load_model(file_name)

    def predict(self, observations):
        """
        Predict action probabilities and value for the provided input.
        Prediction run on the CPU to reduce GPU transfer overheads

        :param observations: input to the model (usually game state)
        :return: action probabilities, state value (model output values)
        """

        with device('/cpu:0'):
            return predict(self.model, observations)

    def set_weights(self, model_weights):
        """
        Set the weights of this model's layers to the ones provided

        :param model_weights: weights of the another model to copied
        """

        self.model.set_weights(model_weights)

    def get_weights(self):
        """
        Get the weights of this model's layers

        :return: the weights of this model's layers
        """

        return self.model.get_weights()
