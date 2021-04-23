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

# Step 4: create a function that prepares the image. This should mimic the data preprocessing in ucbmlai32
def prepare_image(img):
    
    # Convert the image into a numpy array
    img = image.img_to_array(img)

    # Scale the image (remember that pixels are 0-255, so we scale it by dividing 255)
    img /= 255

    # Invert the pixels (because the pixes are currently inverted and will pop up as opposite colors (white number with black background))
    img = 1 - img

    # 'flatten' the image and reshape it using the 28*28 (this is unique only to MNIST database)
    image_array = img.flatten().reshape(-1, 28 * 28)

    # return the processed image array (because the output will be passed into another function)