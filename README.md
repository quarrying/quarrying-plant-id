# Features
- Identify 1320 plant species.
- Model size: 51.8M; model accuracy: 90%+.
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

## Method II: Web App
Run below
```
conda activate plantid
python main.py
```
Then open `http://127.0.0.1:5000/`, and upload an image file.


### Examples
![](docs/plant_00.png)

![](docs/plant_01.png)

![](docs/plant_02.png)

![](docs/plant_03.png)
