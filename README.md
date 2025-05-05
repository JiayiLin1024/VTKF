# VTKF
This repository contains the author's implementation of paper "Towards to Real World Vehicle Privacy Protection: A
New Dataset and Benchmark".


## Requirements
In order to easily run the code, you must have installed the Keras framework with TensorFlow backend. The Darknet framework is self-contained in the "darknet" folder and must be compiled before running the tests. To build Darknet just type "make" in "darknet" folder:
```
cd darknet && make
```


## Dataset
Download LPPD dataset from https://pan.baidu.com/s/1JUcFldqxDAhce0mDAwzFRA?pwd=4v18. Extraction code: 4v18.


## Model
After building the Darknet framework, you need to download the model:
```
wget -c -N http://sergiomsilva.com/data/eccv2018/lp-detector/wpod-net_update1.h5   -P model/lp-detector/
wget -c -N http://sergiomsilva.com/data/eccv2018/lp-detector/wpod-net_update1.json -P model/lp-detector/
```


## To Run
- First go to the `vehicle_tracking` folder and run the vehicle tracking code.
- Then enter the root directory and run `VTKF.py` to protect the privacy of the license plate.