import os
import sys
import time
import glob

import cv2

sys.path.insert(0, '..')
import plantid


if __name__ == '__main__':
    src_dir = r'F:\_Data\Plant\植物之整理\阿福花科 Asphodelaceae\阿福花科_火把莲属_火炬花 Kniphofia uvaria'
    filenames = glob.glob(os.path.join(src_dir, '*.jpg'))
    
    plant_identifier = plantid.PlantIdentifier()
    start_time = time.time()
    for k, name in enumerate(filenames):
        image = plantid.imread_ex(name)
        outputs = plant_identifier.predict(image, topk=5)
        print('[{}/{}] Time: {:.3}s  {}'.format(k+1, len(filenames), time.time() - start_time, name))
        start_time = time.time()
        if outputs['status'] == 0:
            print(outputs['results'][0])
            print(outputs['family_results'][0])
            print(outputs['genus_results'][0])
        else:
            print(outputs)
            
