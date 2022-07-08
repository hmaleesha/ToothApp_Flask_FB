

# Importing flask module in the project is mandatory 
# An object of Flask class is our WSGI application. 
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import numpy as np 
import pandas as pd
from flask import jsonify
from flask import render_template, url_for
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from util import base64_to_pil, pil2datauri
from mrcnn.config import Config
import warnings
import os
import sys
from mrcnn import model as modellib
from PIL import Image
import io
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.model import MaskRCNN
from mrcnn.model import log
import cv2
#from flask_cors import CORS
import pyrebase
import urllib.request

firebaseConfig = {

    "apiKey": "AIzaSyBNn9rUN4XdSAFPSlfqz6_0o-VOtos2zDQ",

    "authDomain": "toothapp-expo.firebaseapp.com",

	"databaseURL": "https://toothapp-expo-default-rtdb.firebaseio.com/",

    "projectId": "toothapp-expo",

    "storageBucket": "toothapp-expo.appspot.com",

    "messagingSenderId": "478583396054",

    "appId": "1:478583396054:web:2d0db93098ab22b65c5b77"

  }

firebase = pyrebase.initialize_app(firebaseConfig)

storage = firebase.storage()

UPLOAD_PATH = 'uploads/'
UPLOAD_PRED_PATH = 'static/prediction/'
app = Flask(__name__)

# The route() function of the Flask class is a decorator,  
# which tells the application which URL should call  
# the associated function. 

class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 8  # Background + phone,laptop and mobile
    GPU_COUNT = 1
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

config = CustomConfig()


MODEL_PATH = './models/mask_rcnn_object_0096.h5'  #PATH TO YOUR MODEL


model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_PATH,config = config)

model.load_weights(MODEL_PATH, by_name=True)
model.keras_model._make_predict_function()
print('Loading Weights')

# model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

def model_predict(img):
    image = img
    results = model.detect([image], verbose=1)
    return results


def getwidth(path):
	img = Image.open(path)
	size = img.size # width and height
	aspect = size[0]/size[1] # width / height
	w = 300 * aspect
	return int(w)


@app.route('/predictapp/predict',methods=['GET', 'POST']) 
def predict():
	fileupload = False
	cost_for_damage = False
	if request.data:
		print("I have data")
	responsejson = {
                'path': 'None'
    }
	if request.method == 'POST':
		# File Upload
		fileupload=True
		f = request.files['fileToUpload']

		image_path = f.filename.split('.')[0] + '/' + f.filename

		storage.child(UPLOAD_PATH + image_path).put(f)

		links = storage.child(UPLOAD_PATH + image_path).get_url(None)

		# Class Prediction
		print(UPLOAD_PATH + image_path)

		#s= storage.child(UPLOAD_PATH + image_path).download(UPLOAD_PATH, f.filename)
		
		req = urllib.request.urlopen(links)
		arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
		img = cv2.imdecode(arr, -1) # 'Load it as it is'
		image=img

		#image = cv2.imread(links)

		results = model_predict(image)

		class_names = ['BG', 'upper-molar', 'upper-premolar', 'upper-canine', 'upper-incisor', 'lower-molar', 'lower-premolar', 'lower-canine', 'lower-incisor']

		r = results[0]


		if not os.path.exists(UPLOAD_PRED_PATH + f.filename.split('.')[0]):
			os.mkdir(os.path.join(UPLOAD_PRED_PATH, f.filename.split('.')[0]))

		pred_path = UPLOAD_PRED_PATH + f.filename.split('.')[0]

		# Save Predicted Class Image
		visualize.save_instances(image, r['rois'], r['masks'], r['class_ids'], class_names,  r['scores'], path=pred_path + '/' + f.filename)
		print(pred_path + '/' + f.filename)
		storage.child(pred_path + '/' + f.filename).put(pred_path + '/' + f.filename)

		if cost_for_damage:
			total, cost = cost_assessment.costEstimate(image, r['rois'], r['masks'], r['class_ids'])
			print(f'File Successfully Manipulated @ {pred_path}')
			data = {
			'visualize': f.filename.split('.')[0] + '/' + f.filename,
			'width': getwidth(UPLOAD_PATH + f.filename.split('.')[0] + '/' + f.filename),
			'masks': get_masks_filenames,
			'top_masks': top_masks_filenames,
			'roi': get_roi_filenames,
			'cost': cost,
			'total': total,
			'tax': round(total * 0.1, 3),
			'tax_total': total + round(total * 0.1, 3)
			}

			responsejson = {
                'path': pred_path + '/' + f.filename
            }

			json_resp = jsonify(responsejson)

		else:
			
			responsejson = {
                'path': pred_path + '/' + f.filename
            }

			json_resp = jsonify(responsejson)
	return json_resp

if __name__ == "__main__":
	warnings.filterwarnings("ignore", category=DeprecationWarning)
	app.run()