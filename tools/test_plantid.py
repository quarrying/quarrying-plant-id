import os
import sys
import time
import glob

import cv2
import numpy as np

sys.path.insert(0, '..')
import plantid


def imread_ex(filename, flags=-1):
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)
    except Exception as e:
        return None
        
        
if __name__ == '__main__':
    src_dir = r'F:\_Data\Plant\植物之整理\阿福花科 Asphodelaceae\阿福花科_火把莲属 Kniphofia uvaria #火炬花'
    filenames = glob.glob(os.path.join(src_dir, '*.jpg'))
    
    plant_identifier = plantid.PlantIdentifier()
    start_time = time.time()
    for k, name in enumerate(filenames):
        image = imread_ex(name)
        if image is None:
            continue
        outputs = plant_identifier.identify(image, topk=5)
        print('[{}/{}] Time: {:.3f}s  {}'.format(k+1, len(filenames), time.time() - start_time, name))
        start_time = time.time()
        if outputs['status'] == 0:
            print(outputs['results'][0])
            print(outputs['genus_results'][0])
            print(outputs['family_results'][0])
        else:
            print(outputs)
            
