import os
import sys
import time

import cv2

sys.path.insert(0, '..')
import plantid


def get_all_filenames(dirname):
    all_filenames = []
    dirname = os.path.expanduser(dirname)
    for root, _, filenames in os.walk(dirname):
        all_filenames += [os.path.join(root, filename) for filename in filenames]
    return all_filenames


if __name__ == '__main__':
    src_dir = r'F:\_Data\Plant\植物之整理\阿福花科 Asphodelaceae\阿福花科_火把莲属_火炬花 Kniphofia uvaria'
    filenames = get_all_filenames(src_dir)
    
    plant_identifier = plantid.PlantIdentifier()
    start_time = time.time()
    for k, name in enumerate(filenames):
        image = plantid.imread_ex(name)
        outputs = plant_identifier.predict(image)
        print('[{}/{}] Time: {:.3}s  {}'.format(k+1, len(filenames), time.time() - start_time, name))
        start_time = time.time()
        if outputs['status'] == 0:
            print(outputs['results'][0])
            print(outputs['family_results'][0])
            print(outputs['genus_results'][0])
        else:
            print(outputs)
            
