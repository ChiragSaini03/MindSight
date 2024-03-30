#  fetching data from API

import requests
import json
from json import JSONEncoder
import cv2
import numpy as np

def get_data(api):
    response = requests.get(api)
    if response.status_code == 200:
        print("API Gateway success! Successfull fetched data")
        print(response.json())
        # print(json.dumps(response.json(), sort_keys=True, indent=4))
    else:
        print(f"Error: {response.status_code}. Failed to fetch data.")
        print("Response content:", response.content)

class NumpyArrayEncoder(JSONEncoder): # encoder to encode numpy array to json
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    

def post_data(api, dic):
    data = {"data": dic}
    data = json.dumps(data, cls=NumpyArrayEncoder)
    response = requests.post(api, json=data)
    if response.status_code == 200:
        print("API Gateway success! Successfull fetched data")
        print(response.json())
    else:
        print(f"Errror {response.status_code}. Failed to fetch data.")
        print(f"Response Content {response.content}")

link="http://127.0.0.1:5000"
# get_data(f"{link}/demo");


# post_data(f"{link}/sreiot", np_image)
