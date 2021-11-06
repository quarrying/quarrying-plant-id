# Features
- Identify 4,066 plant taxa (maybe genus, species, subspecies, variety, etc.).
- Model size is 29.5M; top1/top5 accuracy is 0.848/0.959.
- Open-source model, easy-to-use interface.
- Update continuously.

# Installation
You need install Anaconda, then run below:
```
git clone https://github.com/quarrying/quarrying-plant-id.git
cd quarrying-plant-id
conda create -n plantid python=3.6 -y
conda activate plantid
pip install -r requirements.txt
```

# Usage 

## Method I: Python Interface
```python
import cv2
import plantid

plant_identifier = plantid.PlantIdentifier()
image = cv2.imread(image_filename)
outputs = plant_identifier.identify(image, topk=5)
if outputs['status'] == 0:
    print(outputs['results'])
else:
    print(outputs)
```
You can also see [demo.py](<demo.py>).

## Method II: Web App
Run below
```
cd webapp
conda activate plantid
python app.py
```
Then open <http://127.0.0.1:5000/>, and upload an image file, or use the following codes:
```python
import base64
import json
import requests

url = 'http://127.0.0.1:5000/api/plant'
filename = 'example.jpg'
with open(filename, 'rb') as f:
    image_data = base64.b64encode(f.read())

data = {'image': image_data}
headers = {'content-type': 'application/x-www-form-urlencoded'}
response = requests.post(url, data=data, headers=headers, timeout=30)
print(response.json())
```

### Examples

![](data/plant_01.png)

![](data/plant_02.png)

![](data/plant_03.png)


# Details
See <https://zhuanlan.zhihu.com/p/364346303>.


# ChangeLog

- 20211024 Release model which supports 4066 plant taxa, top1/top5 accuracy is 0.848/0.959.
- 20210905 Release model which supports 2759 plant taxa, top1/top5 accuracy is 0.890/0.971.
- 20210718 Release model which supports 2002 plant taxa.
- 20210609 Release model which supports 1630 plant taxa.
- 20210413 Release model which supports 1320 plant taxa.

