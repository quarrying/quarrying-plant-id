import os
import sys
import time
import glob
import argparse

import cv2
import khandy
import numpy as np

sys.path.insert(0, '..')
import plantid


def imread_ex(filename, flags=-1):
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)
    except Exception as e:
        return None
        
        
def rename_images_by_predict(src_dir, label=None):
    if label is None:
        label = os.path.basename(src_dir)
    plant_identifier = plantid.PlantIdentifier()
    taxon_names, _, _ = plant_identifier.get_plant_names()
    taxon_index = taxon_names.index(label)

    filenames = khandy.get_all_filenames(src_dir)
    start_time = time.time()
    for k, filename in enumerate(filenames):
        image = imread_ex(filename)
        outputs = plant_identifier.predict(image)
        if outputs['status'] == 0:
            confidence = outputs['results']['probs'][0, taxon_index]
            dst_filename = os.path.join(os.path.dirname(filename), '{:.3f}_{}'.format(confidence, os.path.basename(filename)))
            os.rename(filename, dst_filename)
        print('[{}/{}] Time: {:.3f}s  {}'.format(k+1, len(filenames), time.time() - start_time, filename))
        start_time = time.time()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default='F:/_Data/Plant/_raw/五加科_八角金盘属_八角金盘')
    parser.add_argument('--label', type=str, default=None)
    return parser.parse_args(argv)
    
    
if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    if not os.path.exists(args.src_dir):
        raise ValueError('src_dir does not exist!')
    rename_images_by_predict(args.src_dir, label=args.label)
    
    # src_dirs = glob.glob('F:/_Data/Plant/_raw/*')
    # for src_dir in src_dirs:
    #     rename_images_by_predict(src_dir)