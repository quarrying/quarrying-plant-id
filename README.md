# Features
- Identify 2002 taxa (maybe genus, species, subspecies, variety, etc.) of the plant.
- Model size: 52.0M; top1 accuracy: 90%+.
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
import plantid

plant_identifier = plantid.PlantIdentifier()
image = plantid.imread_ex(image_filename)
outputs = plant_identifier.identify(image, topk=5)
if outputs['status'] == 0:
    print(outputs['results'])
    print(outputs['family_results'])
    print(outputs['genus_results'])
else:
    print(outputs)
```
You can also see [tools/test.py](<tools/test.py>).

## Method II: Web App
Run below
```
cd webapp
conda activate plantid
python app.py
```
Then open <http://127.0.0.1:5000/>, and upload an image file.


### Examples

![](docs/plant_01.png)

![](docs/plant_02.png)

![](docs/plant_03.png)

# Details
See <https://zhuanlan.zhihu.com/p/364346303>.

