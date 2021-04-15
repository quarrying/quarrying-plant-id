import os

import cv2
import numpy as np


def load_list(filename, encoding=None, start=0, stop=None, step=1):
    assert isinstance(start, int) and start >= 0
    assert stop is None or (isinstance(stop, int) and stop > start)
    assert isinstance(step, int) and step >= 1
    
    lines = []
    with open(filename, 'r', encoding=encoding) as f:
        for _ in range(start):
            f.readline()
        for k, line in enumerate(f):
            if (stop is not None) and (k + start > stop):
                break
            if k % step == 0:
                lines.append(line.rstrip())
    return lines


def imread_ex(filename, flags=-1):
    """cv2.imread 的扩展, 使支持中文路径.
    """
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)
    except Exception as e:
        return None


interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def resize_image_short(image, dst_size, return_scale=False, interpolation='bilinear'):
    """Resize an image so that the length of shorter side is dst_size while 
    preserving the original aspect ratio.
    
    References:
        `resize_min` in `https://github.com/pjreddie/darknet/blob/master/src/image.c`
    """
    src_height, src_width = image.shape[:2]
    scale = max(dst_size / float(src_width), dst_size / float(src_height))
    dst_width = int(round(scale * src_width))
    dst_height = int(round(scale * src_height))

    resized_image = cv2.resize(image, (dst_width, dst_height), 
                               interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_image
    else:
        return resized_image, scale


def center_crop(image, dst_height, dst_width):
    assert (image.ndim == 2) or (image.ndim == 3)
    assert (image.shape[0] >= dst_height) and (image.shape[1] >= dst_width)
    crop_top = (image.shape[0] - dst_height) // 2
    crop_left = (image.shape[1] - dst_width) // 2
    cropped = image[crop_top: dst_height + crop_top, 
                    crop_left: dst_width + crop_left, ...]
    return cropped


def softmax(x, axis=-1, copy=True):
    if copy:
        x = np.copy(x)
    max_val = np.max(x, axis=axis, keepdims=True)
    x -= max_val
    np.exp(x, x)
    sum_exp = np.sum(x, axis=axis, keepdims=True)
    x /= sum_exp
    return x
    
    
def normalize_image_shape(image):
    """归一化到三通道二维图像
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3:
        num_channels = image.shape[-1]
        if num_channels == 1:
            gray = np.squeeze(image, -1)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif num_channels == 3:
            pass
        elif num_channels == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError('Unsupported!')
    else:
        raise ValueError('Unsupported!')
    return image


def _convert_indices(indices, num_cols):
    num_rows = len(indices)
    row_indices = np.arange(num_rows).reshape((-1, 1))
    row_indices *= num_cols
    indices = indices + row_indices
    return indices
    

def find_topk(a, kth, order=None):
    num_rows, num_cols = a.shape
    top_indices = np.argsort(a, axis=-1, order=order)[:,-kth:][:, ::-1]
    top_indices_take = _convert_indices(top_indices, num_cols)
    values = a.take(top_indices_take)
    return values, top_indices
    
    
