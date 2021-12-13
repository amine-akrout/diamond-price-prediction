import urllib.request
import json
import os

# Request data goes here
data = {
    'carat': 0.23,
    'cut': 'Ideal',
    'color': 'E',
    'clarity': 'SI2',
    'depth': 61.5,
    'table': 55.0,
    'x': 3.95,
    'y': 3.98,
    'z': 2.43,
}

body = str.encode(json.dumps(data))

url = 'http://affdaeb4-9abd-473d-a1c2-e35c67dd4b0f.francecentral.azurecontainer.io/score'
api_key = '' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read().decode("utf8", 'ignore')))
