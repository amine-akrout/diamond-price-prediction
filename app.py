from flask import Flask, render_template, request
import pandas as pd
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import requests

'''
CLASSES = ['Non Smoker', 'Smoker']
SIZE = 150
MODEL_URI = 'http://tf_serving:8501/v1/models/smoker_detector:predict'


def preprocess_img(path):
    img = image.load_img(path, target_size=(SIZE, SIZE))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = np.vstack([img])
    return img
'''

app = Flask(__name__)


@app.route('/')
def entry_page():
    # Jinja template of the webpage
    return render_template('index.html')

''''


'''
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)