from __future__ import division, print_function
# coding=utf-8
import sys
import os
import cv2
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf
global graph
graph = tf.compat.v1.get_default_graph()

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'trained.h5'

#Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    #img = image.load_img(img_path, target_size=(200, 200)) #target_size must agree with what the trained model expects!!

    # Preprocessing the image
    #img = image.img_to_array(img)
    #img = np.expand_dims(img, axis=0)
    img= cv2.imread(img_path)
    tempimg = img
    img = cv2.resize(img,(300,300))
    img = img/255.0
    img = img.reshape(1,300,300,3)
    preds = model.predict(img)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('Home.html')
@app.route('/base',methods=['GET'])
def base():
    # Main page
    return render_template('index.html')
@app.route('/CONTACT US',methods=['GET'])
def contactus():
    # Main page
    return render_template('CONTACT US.php')
@app.route('/ABOUTUS',methods=['GET'])
def aboutus():
    # Main page
    return render_template('ABOUTUS.html')
@app.route('/normal',methods=['GET'])
def normal():
    # Main page
    return render_template('normal.html')
@app.route('/viral',methods=['GET'])
def viral():
    # Main page
    return render_template('viral.html')
@app.route('/bacterial',methods=['GET'])
def bacterial():
    # Main page
    return render_template('bacterial.html')
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        os.remove(file_path)
        #removes file from the server after prediction has been returned
        str1 = 'DETECTED PNEUMONIA -- PLEASE,CONSULT DOCTOR IMMEDIATELY..!'
        str2 = 'NORMAL -- STAY HAPPIE:)'
        if preds >= 0.5:
            return str1
        else:
            return str2
    return None
        



    #this section is used by gunicorn to serve the app on Heroku
if __name__ == '__main__':
        app.run()
    #uncomment this section to serve the app locally with gevent at:  http://localhost:5000
    # Serve the app with gevent 


    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
