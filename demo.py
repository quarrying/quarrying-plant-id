import os
import time
import glob

import cv2
import khandy
import numpy as np

import plantid


def imread_ex(filename, flags=-1):
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)
    except Exception as e:
        print('Image decode error!', e)
        return None
        
        
if __name__ == '__main__':
    src_dirs = [r'images']
    src_filenames = sum([khandy.get_all_filenames(src_dir) for src_dir in src_dirs], [])
    src_filenames = sorted(src_filenames, key=lambda t: os.stat(t).st_mtime, reverse=True)
    
    plant_identifier = plantid.PlantIdentifier()
    start_time = time.time()
    for k, name in enumerate(src_filenames):
        image = imread_ex(name)
        if image is None:
            continue
        outputs = plant_identifier.identify(image, topk=5)
        print('[{}/{}] Time: {:.3f}s  {}'.format(k+1, len(src_filenames), time.time() - start_time, name))
        start_time = time.time()
        
        if max(image.shape[:2]) > 1080:
            image = khandy.resize_image_long(image, 1080)
        if outputs['status'] == 0:
            print(outputs['results'][0])
            print(outputs['results'][1])
            print(outputs['results'][2])
            text = '{}: {:.3f}'.format(outputs['results'][0]['chinese_name'], 
                                       outputs['results'][0]['probability'])
            image = khandy.draw_text(image, text, (10, 10), font='data/simsun.ttc', font_size=15)
        else:
            print(outputs)
        cv2.imshow('image', image)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break
            
