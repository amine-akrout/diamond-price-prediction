from __future__ import print_function
import sys

from flask import Flask, render_template, request
import pandas as pd
import tensorflow as tf
import numpy as np
import json
import requests



app = Flask(__name__)
reloaded_model = tf.keras.models.load_model('diamond_price_predictor')

@app.route('/')
def entry_page():
    # Jinja template of the webpage
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def render_message():
    try:
        # Get data input
        carat = float(request.form['carat'])
        print(carat, file=sys.stderr)
        cut = request.form['cut']
        print(cut, file=sys.stderr)
        color = request.form['color']
        print(color, file=sys.stderr)
        clarity = request.form['clarity']
        print(clarity, file=sys.stderr)
        depth = float(request.form['depth'])
        print(depth, file=sys.stderr)
        table = float(request.form['table'])
        print(table, file=sys.stderr)
        x = float(request.form['x'])
        print(x, file=sys.stderr)
        y = float(request.form['y'])
        print(y, file=sys.stderr)
        z = float(request.form['z'])
        print("y", y, file=sys.stderr)
        volume = x*y*z
        print(volume, file=sys.stderr)
        sample = {
            'carat': carat,
            'cut': cut,
            'color': color,
            'clarity': clarity,
            'depth': depth,
            'table':table,
            'x': x,
            'y': y,
            'z': z,
            'volume': volume,
        }
        print(sample, file=sys.stderr)
        # message = sample
        input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
        predictions = reloaded_model.predict(input_dict)
        predicted = predictions[0]
        print(predicted)

        print('Python module executed successfully')
        message = 'Estimated price : {:.2f} dollar USD +/- 9%'.format(predicted[0])
        print(message, file=sys.stderr)

    except Exception as e:
        # Store error to pass to the web page
        message = "Error encountered. Try with other values. ErrorClass: {}, Argument: {} and Traceback details are: {}".format(
            e.__class__, e.args, e.__doc__)

    # Return the model results to the web page
    return render_template('index.html' ,message=message)

    print(message, file=sys.stderr)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)