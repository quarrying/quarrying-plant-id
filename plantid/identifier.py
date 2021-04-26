import os
from collections import OrderedDict

import cv2
import numpy as np
from .utils import *


def get_label_name_dict(filename):
    records = load_list(filename, 'utf8')
    label_name_dict = {}
    for record in records:
        label, chinese_name, latin_name = record.split(',')
        label_name_dict[int(label)] = OrderedDict([('chinese_name', chinese_name), 
                                                   ('latin_name', latin_name)])
    return label_name_dict


class PlantIdentifier(object):
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.net = cv2.dnn.readNetFromONNX(os.path.join(current_dir, 'models/quarrying_plantid_model.oonx'))
        self.label_name_dict = get_label_name_dict(os.path.join(current_dir, 'models/quarrying_plantid_label_map.txt'))

    def _preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_image_short(image, 320)
        image = center_crop(image, 299, 299)
        image = image.astype(np.float32)
        image /= 255.0
        image -= np.asarray([0.485, 0.456, 0.406])
        image /= np.asarray([0.229, 0.224, 0.225])
        return image
        
    def predict(self, filename, topk=5):
        image = imread_ex(filename, -1)
        if (image is None) or (image.dtype != np.uint8):
            print('Image file corrupted!')
            return None
        try:
            image = normalize_image_shape(image)
        except:
            return None

        image = self._preprocess(image)
        blob = cv2.dnn.blobFromImage(image)
        self.net.setInput(blob)
        results = self.net.forward()
        probs = softmax(results)
        values, top_indices = find_topk(probs, kth=topk)
        probs = values[0]
        class_names = [self.label_name_dict[ind] for ind in top_indices[0]]
        return probs, class_names


