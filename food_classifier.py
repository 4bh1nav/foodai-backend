#!/usr/bin/env python
from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template
from io import BytesIO
from fastai.vision import *
import time
import yaml

app = Flask(__name__)
path = Path(__file__).parent

with open("config.yaml", 'r') as stream:
    APP_CONFIG = yaml.full_load(stream)


@app.route('/<path:path>')
def static_file(path):
    if ".js" in path or ".css" in path:
        return app.send_static_file(path)
    else:
        return app.send_static_file('index.html')

        
@app.route('/')
def hello():
    return app.send_static_file('index.html')

def setup_learner():
    #await download_file(model_file_url, path/'models'/f'{model_file_name}.pkl')
    learn = load_learner('main/models')
    return learn

learn = setup_learner()

def load_image_url(url: str) -> Image:
    response = requests.get(url)
    img = open_image(BytesIO(response.content))
    return img

@app.route('/api/classify', methods=['POST','GET'])
def analyze():
    if request.method == 'GET':
        url = request.args.get("url")
        img = load_image_url(url)
    else:
        data = request.files['file']
        img = open_image(BytesIO(data.read()))

    start = time.time()
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

@app.route('/config')
def config():
    return jsonify(APP_CONFIG)

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='0.0.0.0', port=8080, debug=True)