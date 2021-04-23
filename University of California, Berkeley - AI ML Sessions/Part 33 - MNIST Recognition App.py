# An app that uses the MNIST Machine Learning Library
# Objective: Have the app output a number that is displayed on a submitted image

# Step 1: Import Dependencies
import os
from flask import Flask, request, jsonify

import keras # dependencies for ML
from keras.preprocessing import image
from keras import backend as K

# Step 2: create the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Step 3: create the model by importing a pre-made MNIST model from keras
model = keras.models.load_model("mnist_trained.h5")
graph = K.get_session().graph # in tensorflow, all computation must exist in a graph. So we import a graph session from the backend

