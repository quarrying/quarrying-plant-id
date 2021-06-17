import os
from collections import OrderedDict

import cv2
import numpy as np
from . import utils


class PlantIdentifier(object):
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_filename = os.path.join(current_dir, 'models/quarrying_plantid_model.onnx')
        label_map_filename = os.path.join(current_dir, 'models/quarrying_plantid_label_map.txt')
        self.net = cv2.dnn.readNetFromONNX(model_filename)
        self.label_name_dict = self._get_label_name_dict(label_map_filename)
        
    @staticmethod
    def _get_label_name_dict(filename):
        records = utils.load_list(filename)
        label_name_dict = {}
        for record in records:
            label, chinese_name, latin_name = record.split(',')
            label_name_dict[int(label)] = OrderedDict([('chinese_name', chinese_name), 
                                                       ('latin_name', latin_name)])
        return label_name_dict
        
    @staticmethod
    def _preprocess(image):
        try:
            image = utils.normalize_image_shape(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = utils.resize_image_short(image, 320)
            image = utils.center_crop(image, 299, 299)
            image = image.astype(np.float32)
            image /= 255.0
            image -= np.asarray([0.485, 0.456, 0.406])
            image /= np.asarray([0.229, 0.224, 0.225])
            image = np.transpose(image, (2,0,1))
            image = np.expand_dims(image, axis=0)
            return image
        except:
            return None
        
    def predict(self, image, topk=5):
        inputs = self._preprocess(image)
        if inputs is None:
            return -1
        
        try:
            self.net.setInput(inputs)
            logits = self.net.forward()
            probs = utils.softmax(logits)
            values, top_indices = utils.find_topk(probs, topk)
            results = []
            for ind, prob in zip(top_indices[0], values[0]):
                one_result = self.label_name_dict[ind]
                one_result['probability'] = prob
                results.append(one_result)
            return results
        except:
            return -2

