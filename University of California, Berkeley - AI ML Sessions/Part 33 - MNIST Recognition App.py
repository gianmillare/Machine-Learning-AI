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
    return image_array

# Step 5: Create the app route using the GET and POST methods
@app.route('/', methods=['GET', 'POST'])

# Step 6: the function will ask the user to upload an image, if not pulling the home screen
def upload_file():

    # Create a dictionary and set success equal to false. The JSON associated with this should return True if the model predicts correctly (at the end)
    data = {'success': False}

    # if the method is POST, then we pull that file from an internal folder
    if request.files.get('file'):
        # read the file
        file = request.files['file']
        filename = file.filename
        # create the path to the uploads folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # save the uploaded file to the UPLOAD_FOLDER
        file.save(filepath)

        # Load the saved image using keras and resize it to the MNIST requirement of 28 * 28 pixels
        im = image.load_img(filepath, target_size=(28, 28), grayscale=True)

        # Convers the 2D image into an array of pixels. This is where the prepare_image function is used
        image_array = prepare_image(im)

        # Get the tensorflow default graph from line 18 to make predictions
        global graph
        with graph.as_default():

            # make predictions using the model
            predicted_digit = model.predict_classes(image_array)[0] # we return [0] because the prediction is usually returned as an array '[5]'
            data['prediction'] = str(predicted_digit)

            # change data success to be true to indicate that the image was processed through the model
            data['success'] = True
        
        # return data as a JSON Object
        return jsonify(data)
    
    # if the request was GET, return the HTML page
    return '''
    <!doctype html>

    <title>Upload new File</title>

    <h1>Upload new File</h1>

    <form method=post enctype=multipart/form-data>

      <p><input type=file name=file>
         <input type=submit value=Upload>

    </form>
    '''

# Boiler Plate code
if __name__ == "__main__":
    app.run(debug=True)