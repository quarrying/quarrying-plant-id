import os
import time
import glob

import cv2
import khandy
import numpy as np
import PIL
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import plantid


def imread_ex(filename, flags=-1):
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)
    except Exception as e:
        return None
        
        
def draw_text(image, text, position, font_size=15, color=(255,0,0),
              font_filename='data/simsun.ttc'):
    assert isinstance(color, (tuple, list)) and len(color) == 3
    gray = color[0]
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
        color = (color[2], color[1], color[0])
    elif isinstance(image, PIL.Image.Image):
        pil_image = image
    else:
        raise ValueError('Unsupported image type!')
    assert pil_image.mode in ['L', 'RGB', 'RGBA']
    if pil_image.mode == 'L':
        color = gray
    
    font_object = ImageFont.truetype(font_filename, size=font_size)
    drawable = ImageDraw.Draw(pil_image)
    drawable.text((position[0], position[1]), text, 
                  fill=color, font=font_object)

    if isinstance(image, np.ndarray):
        return np.asarray(pil_image)
    return pil_image


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
        if max(image.shape[:2]) > 1280:
            image = khandy.resize_image_long(image, 1280)
            
        outputs = plant_identifier.identify(image, topk=5)
        print('[{}/{}] Time: {:.3f}s  {}'.format(k+1, len(src_filenames), time.time() - start_time, name))
        start_time = time.time()
        if outputs['status'] == 0:
            print(outputs['results'][0])
            print(outputs['results'][1])
            print(outputs['results'][2])
            text = '{}: {:.3f}'.format(outputs['results'][0]['chinese_name'], 
                                       outputs['results'][0]['probability'])
            image = draw_text(image, text, (0, 10))
        else:
            print(outputs)
        cv2.imshow('image', image)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break
            
