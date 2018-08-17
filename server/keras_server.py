# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py
from glob import glob

import os
import cv2
import matplotlib.pyplot as plt
import matplotlib

from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label
from sklearn.model_selection import train_test_split
# import the necessary packages
from keras import Input, Model
from keras import activations
from keras.models import load_model
from keras.applications import ResNet50
from keras.layers import Dense, LeakyReLU, Dropout, Concatenate, Lambda
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
import sys
import tensorflow as tf
sys.setrecursionlimit(100000)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# initialize our Flask application and the Keras model
# model = load_model('best_model.h5')
app = flask.Flask(__name__)

def __get_model(weight_path):
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess =  tf.Session(graph=graph, config=config)
    with graph.as_default(), sess.as_default():
        model = load_model(weight_path)
        return model, sess, graph

model, __sess, __graph = __get_model('best_model.h5')

# model = load_model('best_model.h5')
# print(model.summary())

# def load_model(shape=(200,200,3)):
		# load the pre-trained Keras model (here we are using a model
		# pre-trained on ImageNet and provided by Keras, but you can
		# substitute in your own networks just as easily)
        
		# model = ResNet50(weights="imagenet")
# 		inputs = Input(shape=shape)
# 		base_model = ResNet50(input_shape=inputs.shape[1:],
# 													input_tensor=inputs,
# 													include_top=False,
# 													classes=1,
# 													pooling='avg'
# 													)
# 		x = base_model.output
# 		x = Dense(394)(x)
# 		x = LeakyReLU()(x)
# 		x = Dropout(0.1)(x)
# 		x = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)

# 		model = Model(inputs, x)
# 		model.compile(loss='binary_crossentropy',
# 									optimizer=Adam(lr=0.0001),
# 									metrics=['accuracy'])
		# #     model.summary()
    
#         model = load_model("best_model.h5")
def crop_roi(img, debug=False):
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray_s = hsv[:, :, 1]

    thresh = threshold_otsu(gray_s)
    # print(thresh)
    mask = gray_s > thresh

    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    print(closed)
    if debug:
        f, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
        axarr[0, 0].imshow(rgb)
        axarr[0, 1].imshow(gray_s)
        axarr[1, 0].imshow(mask)
        axarr[1, 1].imshow(closed)
        plt.show()

    labeled = label(closed)
    props = regionprops(labeled)
    props = [p for p in props if not np.any(np.array(p.bbox) == 0)]
    # bbox = props[0].bbox

    max_r = 0
    max_idx = -1
    for count, prop in enumerate(props):
        count
        bb = prop.bbox
        h0, w0, h1, w1 = bb
        h = h1 - h0
        w = w1 - w0
        if h > w:
            r = 1.2 * 0.5 * h
        else:
            r = 1.2 * 0.5 * w
        if max_r < r:
            max_r = r
            max_idx = count
    bbox = props[max_idx].bbox
    h0, w0, h1, w1 = bbox
    h = h1 - h0
    w = w1 - w0
    if h > w:
        r = 1.2 * 0.5 * h
    else:
        r = 1.2 * 0.5 * w
    ch = (h0 + h1) * 0.5
    cw = (w0 + w1) * 0.5
    c = np.int32([ch, cw])

    s = c - np.int32([r, r])
    e = c + np.int32([r, r])
    print(r)
    print(c)
    print(s[0])
    print(e[0])
    print(s[1])
    print(e[1])
    img[closed==0] = 255
    cropped = img[s[0]:e[0], s[1]:e[1]]
    if debug:
        plt.imshow(cropped)
        plt.show()

    return cropped

def prepare_image(image, target):
		# if the image mode is not RGB, convert it
		if image.mode != "RGB":
				image = image.convert("RGB")

		# resize the input image and preprocess it
		image = image.resize(target)
		image = img_to_array(image).astype(float)
# 		image /= 255.
		image = np.expand_dims(image, axis=0)
		# image = imagenet_utils.preprocess_input(image)

		# return the processed image
		return image

@app.route("/predict", methods=["POST"])
def predict():
		# initialize the data dictionary that will be returned from the
		# view
    data = {"success": False}

		# ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
			# read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
#             image = cv2.imread(io.BytesIO(image))
            # Etraxt object and remove background
            image = Image.fromarray(crop_roi(np.array(image), debug = False))
            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(200, 200))
            # classify the input image and then initialize the list
            # of predictions to return to the client
            with __graph.as_default(), __sess.as_default():
                preds = model.predict(image)

#             print(preds)

            data["predictions"] = [{
                    "label": "Result",
                    "probability": float(preds[0][0])
            }]

            # indicate that the request was a success
            data["success"] = True

		# return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
		print(("* Loading Keras model and Flask starting server..."
			"please wait until server has fully started"))
# 		load_model()
		app.run(host="0.0.0.0",port=9898)
