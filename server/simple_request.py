# USAGE
# python simple_request.py

# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:9898/predict"
# IMG_PATHS = ["image_aoff_001.jpg", "image_aoff_002.jpg", "image_aoff_003.jpg", "image_openclose_001.jpg", "image_openclose_002.jpg"]
IMG_PATHS = ["img/test.jpg","img/ng1.jpg"]

for img_path in IMG_PATHS:
    print("image: ", img_path)
    # load the input image and construct the payload for the request
    image = open(img_path, "rb").read()
    payload = {"image": image}

    # submit the request
    
    r = requests.post(KERAS_REST_API_URL, files=payload).json()

    # ensure the request was sucessful
    if r["success"]:
        # loop over the predictions and display them
        for (i, result) in enumerate(r["predictions"]):
            print("{}. {}: {:.4f}".format(i + 1, result["label"],
                result["probability"]))

    # otherwise, the request failed
    else:
        print("Request failed")