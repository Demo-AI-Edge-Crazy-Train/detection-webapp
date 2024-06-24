import os
import logging
import onnxruntime as ort
import ast

from flask import Flask, jsonify, request, render_template
from utils import *

REST_ENDPOINT = os.environ.get("REST_ENDPOINT", "http://modelmesh-serving:8008")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "base")
MODEL_PATH = "models/" + MODEL_VERSION + ".onnx"
PROVIDERS = ast.literal_eval(os.environ.get("ONNXRUNTIME_PROVIDERS", '[]'))
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.7))


application = Flask(__name__)

@application.route('/', methods=['GET'])
def status():
    return render_template('index.html')

@application.route('/predictions', methods=['POST'])
def inference():
    img_b64 = request.json["image_base64"]
    application.logger.debug(f"PNG in base64: {img_b64}")
    detections = process_image(img_b64, ort_sess, CONF_THRESHOLD)
    return jsonify({"data": detections})

if __name__ == '__main__':
    ort_sess = ort.InferenceSession(MODEL_PATH, providers=PROVIDERS)
    application.logger.setLevel(logging.INFO)
    application.run(host='0.0.0.0',port=8080)