# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

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


# graph_num_model = tf.Graph()
# with graph_num_model.as_default():
#     sess_num_model = tf.Session()
#     with sess_num_model.as_default():
#         num_model = load_model('netCRNN_001_000032000_trainloss_0.01610_valloss_0.51269_valacc_0.90906.h5')

# graph_word_model = tf.Graph()
# with graph_word_model.as_default():
#     sess_word_model = tf.Session()
#     with sess_word_model.as_default():
#         word_model = load_model('word_model.h5')


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

                # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(200, 200))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            with __graph.as_default(), __sess.as_default():
                
                preds = model.predict(image)

            print(preds)

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