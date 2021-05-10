import os
from collections import OrderedDict

import cv2
import numpy as np
from .utils import *


class PlantIdentifier(object):
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_filename = os.path.join(current_dir, 'models/quarrying_plantid_model.oonx')
        label_map_filename = os.path.join(current_dir, 'models/quarrying_plantid_label_map.txt')
        self.net = cv2.dnn.readNetFromONNX(model_filename)
        self.label_name_dict = self._get_label_name_dict(label_map_filename)
        
    @staticmethod
    def _get_label_name_dict(filename):
        records = load_list(filename, 'utf-8')
        label_name_dict = {}
        for record in records:
            label, chinese_name, latin_name = record.split(',')
            label_name_dict[int(label)] = OrderedDict([('chinese_name', chinese_name), 
                                                       ('latin_name', latin_name)])
        return label_name_dict
        
    @staticmethod
    def _preprocess(image):
        try:
            image = normalize_image_shape(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = resize_image_short(image, 320)
            image = center_crop(image, 299, 299)
            image = image.astype(np.float32)
            image /= 255.0
            image -= np.asarray([0.485, 0.456, 0.406])
            image /= np.asarray([0.229, 0.224, 0.225])
            return image
        except:
            return None
        
    def predict(self, filename, topk=5):
        image = imread_ex(filename, -1)
        if (image is None) or (image.dtype != np.uint8):
            return -1
        image = self._preprocess(image)
        if image is None:
            return -2
        
        try:
            blob = cv2.dnn.blobFromImage(image)
            self.net.setInput(blob)
            logits = self.net.forward()
            probs = softmax(logits)
            values, top_indices = find_topk(probs, kth=topk)
            results = []
            for ind, prob in zip(top_indices[0], values[0]):
                one_result = self.label_name_dict[ind]
                one_result['probability'] = prob
                results.append(one_result)
            return results
        except:
            return -3

