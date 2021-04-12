import os
import glob
import time

import cv2
import numpy as np


def get_all_filenames(dirname):
    all_filenames = []
    dirname = os.path.expanduser(dirname)
    for root, _, filenames in os.walk(dirname):
        all_filenames += [os.path.join(root, filename) for filename in filenames]
    return all_filenames


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


def get_label_name_dict(filename):
    records = load_list(filename, 'utf8')
    label_name_dict = {}
    for record in records:
        name, label = record.split(',')
        label_name_dict[int(label)] = name
    return label_name_dict


def _preprocess(image):
    image = image.astype(np.float32)
    image = image / 255.0
    image -= np.asarray([0.485, 0.456, 0.406])
    image /= np.asarray([0.229, 0.224, 0.225])
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
    
    
def predict(net, filename, topk=5):
    image = imread_ex(filename, -1)
    if (image is None) or (image.dtype != np.uint8):
        print('Image file corrupted!')
        return None
    try:
        image = normalize_image_shape(image)
    except:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_image_short(image, 320)
    image = center_crop(image, 299, 299)
    image = _preprocess(image)

    blob = cv2.dnn.blobFromImage(image)
    net.setInput(blob)
    results = net.forward()
    probs = softmax(results)
    # index = np.argmax(probs)
    # return index, probs[0, index]
    values, top_indices = find_topk(probs, kth=topk)
    return values[0], top_indices[0]


if __name__ == '__main__':
    src_dir = 'F:/_Data/Plant/植物之其他/_raw/牵牛花'
    class_names_filename = 'models/quarrying_plantid_label_map.txt'
    onnx_filename = 'models/quarrying_plantid_model.oonx'

    net = cv2.dnn.readNetFromONNX(onnx_filename)
    label_name_dict = get_label_name_dict(class_names_filename)

    filenames = get_all_filenames(src_dir)
    start_time = time.time()
    for k, name in enumerate(filenames):
        index, prob = predict(net, name)
        print('[{}/{}] Time: {:.3}s  {}'.format(k+1, len(filenames), time.time() - start_time, name))
        start_time = time.time()
        print(label_name_dict[index], prob)

