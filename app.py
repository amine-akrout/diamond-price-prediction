# Import Libraries
from __future__ import print_function
import sys
import requests
import json
from flask import Flask, render_template, request
import tensorflow as tf
import os

# Deactivate GPU for inference
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

# Uri of the deployed model
URI = "http://affdaeb4-9abd-473d-a1c2-e35c67dd4b0f.francecentral.azurecontainer.io/score"

app = Flask(__name__)

@app.route('/')
def entry_page():
    # Nicepage template of the webpage
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def render_message():
    try:
        # Get data input
        carat = float(request.form['carat'])
        cut = request.form['cut']
        color = request.form['color']
        clarity = request.form['clarity']
        depth = float(request.form['depth'])
        table = float(request.form['table'])
        x = float(request.form['x'])
        y = float(request.form['y'])
        z = float(request.form['z'])
        volume = x*y*z
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

        data = json.dumps(sample)
        headers = {'Content-Type':'application/json'}
        response  = requests.post(URI, data, headers=headers)

        print('Python module executed successfully')
        message = 'Estimated price : {} dollar USD +/- 9%'.format(response.json())
        print(message, file=sys.stderr)

    except Exception as e:
        # Store error to pass to the web page
        message = "Error encountered. Try with other values. ErrorClass: {}, Argument: {} and Traceback details are: {}".format(
            e.__class__, e.args, e.__doc__)

    # Return the model results to the web page
    return render_template('index.html' ,message=message)

if __name__ == '__main__':
    app.run(debug=True) #, host='0.0.0.0', port=8080
