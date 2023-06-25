import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

from tensorflow import device
from tensorflow.keras.models import load_model
from ActorCriticTrainers import predict


class SelfPlayModel:
    def __init__(self, filename):
        self.model = load_model(filename)

    def predict(self, observations):
        with device('/cpu:0'):
            return predict(self.model, observations)

    def set_weights(self, model_weights):
        self.model.set_weights(model_weights)

    def get_weights(self):
        return self.model.get_weights()
