from flask import Flask, request, render_template
import pickle
import pandas as pd
from flask_cors import CORS, cross_origin
import json
import numpy as np

app=Flask(__name__)
CORS(app)

@app.route("/demo", methods=["GET"])
def demo():
    # data = request.get_json()
    res="final data fetched!!"
    res={"data":res}
    return json.dumps(res)


@app.route("/sreiot", methods=["POST","GET"])
def sreiot():
    data = request.get_json()
    data = json.loads(data)
    image = np.asarray(data["data"])
    
    print(data)
    res=10
    res={"data":res}
    return json.dumps(res)


app.run(debug=True)


#virtualenv venv
#.\\venv\Scripts\activate

#python server.py
