import os
import sys
import time
from datetime import timedelta
from collections import OrderedDict

import cv2
import numpy as np
from flask import Flask
from flask import request
from flask import render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

sys.path.insert(0, '..')
import plantid


app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)
plant_identifier = plantid.PlantIdentifier()

MIN_IMAGE_SIZE_LENGTH = 16
MAX_IMAGE_SIZE_LENGTH = 5160


def convert_to_dict_list(chinese_names, latin_names, probs):
    dict_list = []
    for chinese_name, latin_name, prob in zip(chinese_names, latin_names, probs):
        one_result = OrderedDict()
        one_result['ChineseName'] = chinese_name
        one_result['LatinName'] = latin_name
        one_result['Confidence'] = prob
        dict_list.append(one_result)
    return dict_list
    
    
def predict(image):
    if image is None:
        return {"status": 1002, "message": "Image parsing error!", "results": []}
    if image.dtype != np.uint8:
        return {"status": 1003, "message": "Image data type error, "
                "only support uint8 data type.", "results": []}
    if max(image.shape[:2]) > MAX_IMAGE_SIZE_LENGTH or min(image.shape[:2]) < MIN_IMAGE_SIZE_LENGTH:
        return {"status": 1004, "message": "Image size error, "
                "the shorter edge must >= {}px, the longer edge must <= {}px".format(
                MIN_IMAGE_SIZE_LENGTH, MAX_IMAGE_SIZE_LENGTH), "results": []}

    outputs = plant_identifier.predict(image)
    if outputs == -1:
        return {"status": 1005, "message": "Image preprocess error.", "results": []}
    elif outputs == -2:
        return {"status": 1006, "message": "Inference error.", "results": []}
        
    chinese_names = [item['chinese_name'] for item in outputs]
    latin_names = [item['latin_name'] for item in outputs]
    probs = ['{:.5f}'.format(item['probability']) for item in outputs]
    return {"status": 0, "message": "OK", "results": convert_to_dict_list(chinese_names, latin_names, probs)}
    

def allowed_file_type(filename):
    ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp']
    extension = os.path.splitext(os.path.basename(filename))[1]
    return extension.lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST', 'GET'])
def main():
    base_dir = os.path.dirname(__file__)
    raw_image_dir = os.path.join(base_dir, 'static/raw_images')
    tmp_image_dir = os.path.join(base_dir, 'static/images')
    os.makedirs(raw_image_dir, exist_ok=True)
    os.makedirs(tmp_image_dir, exist_ok=True)
    
    if request.method == 'POST':
        f = request.files['image']
        image_filename = os.path.basename(secure_filename(f.filename))

        if not (f and allowed_file_type(image_filename)):
            results = {"status": 1001, "message": "Image file format error, "
                       "only support png, jpg, jpeg, bmp.", "results": []}
            return render_template('upload_error.html', 
                                   status=results['status'], message=results['message'],
                                   image_filename=image_filename, 
                                   timestamp=time.time())
                                   
        time_stamp = int(round(time.time() * 1000))
        new_image_filename = '{}{}'.format(time_stamp, os.path.splitext(image_filename)[-1])
        raw_image_filename = os.path.join(raw_image_dir, new_image_filename)  
        f.save(raw_image_filename)
        
        image = plantid.imread_ex(raw_image_filename, -1)
        results = predict(image)
        if len(results['results']) != 0:
            image = plantid.resize_image_short(image, 512)
            cv2.imwrite(os.path.join(tmp_image_dir, new_image_filename), image)
            labels = ['Chinese Name', 'Latin Name', 'Confidence']
            return render_template('upload_ok.html', 
                                   labels=labels, results=results['results'], 
                                   image_filename=new_image_filename, 
                                   timestamp=time.time())
        else:
            return render_template('upload_error.html', 
                                   status=status, message=message,
                                   image_filename=new_image_filename, 
                                   timestamp=time.time())
    return render_template('upload.html')
    
    
if __name__ == '__main__':
    app.run(port=5000, debug=False)
    
    # http_server = WSGIServer(('0.0.0.0', 5000), app)
    # http_server.serve_forever()
