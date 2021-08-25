import os
import sys
import time
import glob
import shutil
import argparse

import cv2
import numpy as np

sys.path.insert(0, '..')
import plantid


def imread_ex(filename, flags=-1):
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)
    except Exception as e:
        return None
        
        
def split_images_by_identify(src_dir, dst_dir):
    plant_identifier = plantid.PlantIdentifier()
    
    filenames = glob.glob(os.path.join(src_dir, '*'))
    start_time = time.time()
    for k, filename in enumerate(filenames):
        image = imread_ex(filename)
        outputs = plant_identifier.identify(image, topk=1)
        if outputs['status'] == 0:
            chinese_name = outputs['results'][0]['chinese_name']
            latin_name = outputs['results'][0]['latin_name']
            confidence = outputs['results'][0]['probability']
            if latin_name == '':
                taxon_name = chinese_name
            else:
                taxon_name = '{} {}'.format(chinese_name, latin_name)
            if confidence > 0.1:
                dst_subdir =  os.path.join(dst_dir, taxon_name)
                os.makedirs(dst_subdir, exist_ok=True)
                
                dst_filename = os.path.join(dst_subdir, '{:.3f}_{}'.format(confidence, os.path.basename(filename)))
                shutil.move(filename, dst_filename)
        print('[{}/{}] Time: {:.3f}s  {}'.format(k+1, len(filenames), time.time() - start_time, filename))
        start_time = time.time()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default='E:/test_images')
    parser.add_argument('--dst_dir', type=str, default='E:/test_images_results')
    return parser.parse_args(argv)
    
    
if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    if not os.path.exists(args.src_dir):
        raise ValueError('src_dir does not exist!')
    split_images_by_identify(args.src_dir, dst_dir=args.dst_dir)

    