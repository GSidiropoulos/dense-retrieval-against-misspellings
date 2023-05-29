# Baselines

We use the original [DPR](https://github.com/facebookresearch/DPR) library, to obtain the `DR` and `DR+Augm.` baselines. 

## Installation  
You can clone the original repo and install with pip.
```
git clone https://github.com/facebookresearch/DPR.git
git checkout 49e5838f94ffced8392be750ded2a8fa4a14b5cf
cd DPR
pip install .
```
## DR
Use the original code as it is.

## DR+Augm.
Replace the original biencoder.py file, under `dpr/models/biencoder.py`, with `typo-aware/dpr/models/biencoder.py`.
