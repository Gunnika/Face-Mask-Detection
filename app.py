#importing libraries
import os,cv2
import numpy as np
import flask
from flask import jsonify
import pickle
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

# Load the pre-trained model.
model = load_model('model/face_mask_model.h5')

def classify(image, model):
    class_names = ['without mask','with mask']
    preds = model.predict(image)
    classification = np.argmax(preds)
    return class_names[classification]

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/' ,  methods=['GET', 'POST'])
def index():
    urls = []
    if flask.request.method == 'GET':
        return flask.render_template('index.html')

    if flask.request.method == 'POST':
        # Get image URL as input
        image_url = flask.request.form['url_field']
        image = cv2.imread(image_url)
        processed_img = image/255
        prediction = classify(processed_img, model)
        return flask.render_template('index.html', result = str(prediction[0]))


if __name__=='__main__':
    app.secret_key = os.urandom(12)
    app.run(host='0.0.0.0', debug=True)