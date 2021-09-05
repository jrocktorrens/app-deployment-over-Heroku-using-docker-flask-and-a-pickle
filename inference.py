import os
import pickle
from flask import Flask, request
import pandas as pd
import numpy as np
"""
this function runs flask and opens an inference server that allows requesting predictions.
the function loads a model (from pickle format) and returns a prediction based on a client request.
"""
app = Flask(__name__) # running flask

with open('churn_model.pkl', 'rb') as f:
    clf = pickle.load(f) # extracting a trained model    


@app.route('/predict_churn')
def predict_churn():
    """
    this functions predicts a result for the client and returns it.
    """
    X_pred = pd.DataFrame(dict(request.args), index=[0])
    y_pred = clf.predict(X_pred)
    return str(y_pred)


if __name__ == '__main__':
    # Heroku provides environment variable 'PORT' that should be listened on by Flask
    port = os.environ.get('PORT')
    if port:
        # 'PORT' variable exists - running on Heroku, listen on external IP and on given by Heroku port
        app.run(host='0.0.0.0', port=int(port))
    else:
        # 'PORT' variable doesn't exist, running not on Heroku, presumabely running locally, run with default
        #   values for Flask (listening only on localhost on default Flask port)
        app.run()
