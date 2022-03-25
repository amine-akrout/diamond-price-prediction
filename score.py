
import json
import os
import tensorflow as tf
import numpy as np

from azureml.core.model import Model
import logging
logging.basicConfig(level=logging.DEBUG)


def init():
    global model
    model_path = Model.get_model_path('diamond_model')
    model = tf.keras.models.load_model(model_path)


def run(data):
    input_data = json.loads(data)
    input_dict = {name: tf.convert_to_tensor([value]) for name, value in input_data.items()}
    predictions = model.predict(input_dict)
    predicted = predictions[0]
    result = str(round(predicted[0],2))
    return result
