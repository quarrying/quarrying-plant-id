import os
from collections import OrderedDict

import cv2
import khandy
import numpy as np
import onnxruntime


class OnnxModel(object):
    def __init__(self, model_path):
        sess_options = onnxruntime.SessionOptions()
        # # Set graph optimization level to ORT_ENABLE_EXTENDED to enable bert optimization.
        # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # # Use OpenMP optimizations. Only useful for CPU, has little impact for GPUs.
        # sess_options.intra_op_num_threads = multiprocessing.cpu_count()
        onnx_gpu = (onnxruntime.get_device() == 'GPU')
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if onnx_gpu else ['CPUExecutionProvider']
        self.sess = onnxruntime.InferenceSession(model_path, sess_options, providers=providers)
        self._input_names = [item.name for item in self.sess.get_inputs()]
        self._output_names = [item.name for item in self.sess.get_outputs()]
        
    @property
    def input_names(self):
        return self._input_names
        
    @property
    def output_names(self):
        return self._output_names
        
    def forward(self, inputs):
        to_list_flag = False
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
            to_list_flag = True
        input_feed = {name: input for name, input in zip(self.input_names, inputs)}
        outputs = self.sess.run(self.output_names, input_feed)
        if (len(self.output_names) == 1) and to_list_flag:
            return outputs[0]
        else:
            return outputs
            

def check_image_dtype_and_shape(image):
    if not isinstance(image, np.ndarray):
        raise Exception(f'image is not np.ndarray!')

    if isinstance(image.dtype, (np.uint8, np.uint16)):
        raise Exception(f'Unsupported image dtype, only support uint8 and uint16, got {image.dtype}!')
    if image.ndim not in {2, 3}:
        raise Exception(f'Unsupported image dimension number, only support 2 and 3, got {image.ndim}!')
    if image.ndim == 3:
        num_channels = image.shape[-1]
        if num_channels not in {1, 3, 4}:
            raise Exception(f'Unsupported image channel number, only support 1, 3 and 4, got {num_channels}!')


class PlantIdentifier(OnnxModel):
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models/quarrying_plantid_model.onnx')
        label_map_path = os.path.join(current_dir, 'models/quarrying_plantid_label_map.txt')
        family_name_map_path = os.path.join(current_dir, 'models/family_name_map.json')
        genus_name_map_path = os.path.join(current_dir, 'models/genus_name_map.json')
        super(PlantIdentifier, self).__init__(model_path)
        
        self.label_name_dict = self._get_label_name_dict(label_map_path)
        self.family_dict, self.genus_dict = self._get_family_and_genus_dict(label_map_path)
        self.family_name_map = khandy.load_json(family_name_map_path)
        self.genus_name_map = khandy.load_json(genus_name_map_path)
        
        self.names = [self.label_name_dict[i]['chinese_name'] for i in range(len(self.label_name_dict))]
        self.family_names = list(self.family_dict.keys())
        self.genus_names = list(self.genus_dict.keys())
        
    @staticmethod
    def _get_label_name_dict(filename):
        records = khandy.load_list(filename)
        label_name_dict = {}
        for record in records:
            label, chinese_name, latin_name = record.split(',')
            label_name_dict[int(label)] = OrderedDict([('chinese_name', chinese_name), 
                                                       ('latin_name', latin_name)])
        return label_name_dict
        
    @staticmethod
    def _get_family_and_genus_dict(filename):
        records = khandy.load_list(filename)
        # genus_dict should be understood as genus_or_above_taxon_dict
        family_dict, genus_dict = {}, {}
        for record in records:
            label, chinese_name, _ = record.split(',')
            underscore_parts = chinese_name.split('_')
            if len(underscore_parts) == 1:
                family_dict.setdefault(underscore_parts[0], []).append(int(label))
                genus_dict.setdefault(underscore_parts[0], []).append(int(label))
            elif len(underscore_parts) > 1:
                family_dict.setdefault(underscore_parts[0], []).append(int(label))
                genus_dict.setdefault('_'.join(underscore_parts[:2]), []).append(int(label))
        return family_dict, genus_dict
        
    @staticmethod
    def _get_collective_probs(probs, collective_dict):
        batch_size = len(probs)
        num_collective = len(collective_dict)
        collective_probs = np.empty((batch_size, num_collective), dtype=probs.dtype)
        for batch_ind in range(batch_size):
            for collective_ind, collective_name in enumerate(collective_dict):
                taxon_indices = collective_dict[collective_name]
                collective_prob = sum(probs[batch_ind, index] for index in taxon_indices)
                collective_probs[batch_ind, collective_ind] = collective_prob
        return collective_probs
        
    @staticmethod
    def _preprocess(image):
        check_image_dtype_and_shape(image)

        # image size normalization
        image = khandy.resize_image_short(image, 224)
        image = khandy.center_crop(image, 224, 224)
        # image channel normalization
        image = khandy.normalize_image_channel(image, swap_rb=True)
        # image dtype and value range normalization
        mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        image = khandy.normalize_image_value(image, mean, stddev, 'auto')
        # to tensor
        image = np.transpose(image, (2,0,1))
        image = np.expand_dims(image, axis=0)
        return image
        
    def get_plant_names(self):
        return self.names, self.family_names, self.genus_names
        
    def predict(self, image):
        try:
            inputs = self._preprocess(image)
        except Exception as e:
            return {"status": -1, "message": "Inference preprocess error.", "results": {}}
        
        try:
            logits = self.forward(inputs)
            probs = khandy.softmax(logits)
        except Exception as e:
            return {"status": -2, "message": "Inference error.", "results": {}}
            
        family_probs = self._get_collective_probs(probs, self.family_dict)
        genus_probs = self._get_collective_probs(probs, self.genus_dict)
        results = {'probs': probs, 'family_probs': family_probs, 'genus_probs': genus_probs,}
        return {"status": 0, "message": "OK", "results": results}
        
    def identify(self, image, topk=5):
        assert isinstance(topk, int)
        if topk <= 0:
            topk = max(len(self.label_name_dict), 
                       len(self.family_dict),
                       len(self.genus_dict))
        results, family_results, genus_results = [], [], []
        
        outputs = self.predict(image)
        status = outputs['status']
        message = outputs['message']
        if outputs['status'] != 0:
            return {"status": status, "message": message, 
                    "results": results, "family_results": family_results,
                    "genus_results": genus_results}
                    
        probs = outputs['results']['probs']
        family_probs = outputs['results']['family_probs']
        genus_probs = outputs['results']['genus_probs']

        taxon_topk = min(probs.shape[-1], topk)
        topk_probs, topk_indices = khandy.top_k(probs, taxon_topk)
        for ind, prob in zip(topk_indices[0], topk_probs[0]):
            one_result = self.label_name_dict[ind]
            one_result['probability'] = prob.item()
            results.append(one_result)

        family_topk = min(family_probs.shape[-1], topk)
        family_topk_probs, family_topk_indices = khandy.top_k(family_probs, family_topk)
        for ind, prob in zip(family_topk_indices[0], family_topk_probs[0]):
            one_result = OrderedDict()
            one_result['chinese_name'] = self.family_names[ind]
            one_result['latin_name'] = self.family_name_map.get(self.family_names[ind], '')
            one_result['probability'] = prob.item()
            family_results.append(one_result)
            
        genus_topk = min(genus_probs.shape[-1], topk)
        genus_topk_probs, genus_topk_indices = khandy.top_k(genus_probs, genus_topk)
        for ind, prob in zip(genus_topk_indices[0], genus_topk_probs[0]):
            one_result = OrderedDict()
            one_result['chinese_name'] = self.genus_names[ind]
            one_result['latin_name'] = self.genus_name_map.get(self.genus_names[ind], '')
            one_result['probability'] = prob.item()
            genus_results.append(one_result)
            
        return {"status": status, "message": message, 
                "results": results, "family_results": family_results,
                "genus_results": genus_results}
                
