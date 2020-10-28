from flask import Flask, request
from flask_cors import CORS
import sys
sys.path.append('../app/')
import engine
import numpy as np
import json
import joblib
from PIL import Image
import urllib.request

app = Flask(__name__) # create the Flask app
CORS(app) # allow requests from other domains

@app.route('/img-url', methods=['POST'])
def img_url():
    url = request.get_json()
    urllib.request.urlretrieve(url, 'image.bmp')
    im = Image.open("image.bmp")
    img = np.array(im)
    loaded_model = joblib.load(open('data/models/image_classifier_pipe.pkl',
                                    'rb'))
    loaded_model.predict(np.reshape(img,(1,240,320,3)))

    return loaded_model.predict(np.reshape(img,(1,240,320,3)))[0]

if __name__ == '__main__':
    app.run(debug=True, port=5000) #run app in debug mode on port 5000
