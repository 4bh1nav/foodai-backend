#!/usr/bin/env python
from flask import Flask
from flask import request
from flask import jsonify
from io import BytesIO
from fastai.vision import *
import time

app = Flask(__name__)
path = Path(__file__).parent

@app.route('/')
def hello():
    start = time.time()
    """Return a friendly HTTP greeting."""
    end = time.time()
    print("%.5f " % (end - start))
    return 'Hello World!'

def setup_learner():
    #await download_file(model_file_url, path/'models'/f'{model_file_name}.pkl')
    learn = load_learner('main/models')
    return learn

learn = setup_learner()

@app.route('/analyze', methods=['POST'])
def analyze():
    start = time.time()
    data = request.files['file']
    img = open_image(BytesIO(data.read()))
    cat,index,preds = learn.predict(img)
    end = time.time()
    print("%.5f " % (end - start))
    return jsonify({'result': top_5_pred_labels(preds,learn.data.classes)})


def top_5_pred_labels(preds, classes):
    top_5 = np.flip(np.argsort(preds.numpy()))[:5]
    labels = []
    for i in range(len(top_5)):
        labels.append(classes[top_5[i]])
    return labels

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='0.0.0.0', port=8080, debug=True)