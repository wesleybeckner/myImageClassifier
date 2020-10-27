from flask import Flask, request
from flask_cors import CORS
import sys
sys.path.append('../app/')
import engine
import numpy as np
import json
import joblib

app = Flask(__name__) #create the Flask app
CORS(app)

#@app.route('/query-example')
#def query_example():
#    language = request.args.get('language') #if key doesn't exist, returns None
#    loaded_model = joblib.load(open('data/models/image_classifier_pipe.pkl', 'rb'))
#    img_clf = ImageLoader()
#    train_raw, train_labels = img_clf.load_data_from_folder('train/')
#    test_raw, test_labels = img_clf.load_data_from_folder('test/')
#
#    return loaded_model.predict(np.reshape(test_raw[0],(1,240,320,3)))[0]

@app.route('/form-example', methods=['GET', 'POST']) #allow both GET and POST requests
def form_example():
    if request.method == 'POST':  #this block is only entered when the form is submitted
        language = request.form.get('language')
        framework = request.form.get('framework')

        return '''<h1>The language value is: {}</h1>
                  <h1>The framework value is: {}</h1>'''.format(language, framework)

    return '''<form method="POST">
                  Language: <input type="text" name="language"><br>
                  Framework: <input type="text" name="framework"><br>
                  <input type="submit" value="Submit"><br>
              </form>'''

@app.route('/post-example', methods=['POST'])
def front_api():
    if request.method == 'POST':
        req = request.get_json()
        img = np.array(req)
        loaded_model = joblib.load(open('data/models/image_classifier_pipe.pkl', 'rb'))
        return loaded_model.predict(img)[0]


if __name__ == '__main__':
    app.run(debug=True, port=5000) #run app in debug mode on port 5000
