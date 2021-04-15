import os
import sys
import time
import base64
from io import BytesIO
from datetime import timedelta

import cv2
from PIL import Image
from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

sys.path.insert(0, '..')
import plantid


app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)
plant_identifier = plantid.PlantIdentifier()


def allowed_file_type(filename):
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'bmp', 'JPG', 'PNG', 'BMP'])
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
 

@app.route('/', methods=['POST', 'GET'])
def predict_web():
    if request.method == 'POST':
        f = request.files['image']
        image_filename = os.path.basename(secure_filename(f.filename))

        if not (f and allowed_file_type(image_filename)):
            return jsonify({"error": 1001, "message": "请检查上传的图片类型，仅支持png、PNG、jpg、JPG、bmp、BMP"})
        base_dir = os.path.dirname(__file__)
 
        raw_image_dir = os.path.join(base_dir, 'static/raw_images')
        tmp_image_dir = os.path.join(base_dir, 'static/images')
        os.makedirs(raw_image_dir, exist_ok=True)
        os.makedirs(tmp_image_dir, exist_ok=True)
        
        time_stamp = int(round(time.time() * 1000))
        new_image_filename = '{}{}'.format(time_stamp, os.path.splitext(image_filename)[-1])
        raw_image_filename = os.path.join(raw_image_dir, new_image_filename)  
        f.save(raw_image_filename)
 
        img = cv2.imread(raw_image_filename)
        img = plantid.resize_image_short(img, 512)
        cv2.imwrite(os.path.join(tmp_image_dir, new_image_filename), img)
        
        probs, class_names = plant_identifier.predict(raw_image_filename)
        probs = ['{:.5f}'.format(prob) for prob in probs]
        labels = ['植物物种', '置信度']
        records = zip(class_names, probs)

        return render_template('upload_ok.html', 
                               labels=labels, records=records, 
                               image_filename=new_image_filename, 
                               timestamp=time.time())
    return render_template('upload.html')


def base64_to_pil(image_data):
    """Convert base64 image data to PIL image
    """
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image
    
    
@app.route('/api', methods=['GET', 'POST'])
def predict_json():
    if request.method == 'POST':
        base_dir = os.path.dirname(__file__)
 
        raw_image_dir = os.path.join(base_dir, 'static/raw_images')
        tmp_image_dir = os.path.join(base_dir, 'static/images')
        os.makedirs(raw_image_dir, exist_ok=True)
        os.makedirs(tmp_image_dir, exist_ok=True)
        
        time_stamp = int(round(time.time() * 1000))
        raw_image_filename = os.path.join(raw_image_dir, '{}.jpg'.format(time_stamp))  

        img = base64_to_pil(request.form.get('image'))
        img.save(raw_image_filename)
        
        probs, class_names = plant_identifier.predict(raw_image_filename)
        probs = ['{:.5f}'.format(prob) for prob in probs]

        return jsonify(class_names=class_names, probs=probs)
    return None
    
    
if __name__ == '__main__':
    app.run(port=5000, debug=False)
    
    # http_server = WSGIServer(('0.0.0.0', 5000), app)
    # http_server.serve_forever()
