## Environment

|Category|Name|Version|
|-----|-----|-----|
|os|ubuntu|18.04|
||python|3.6|
|Deep Learning Frameworks|tensorflow-gpu|1.11.0|
||Keras|2.1.5|
||opencv|3.4.9|

```angular2
conda create -n VT_sort python=3.6
conda install tensorflow-gpu==1.13.0 Keras==2.3.1
conda install --channel https://conda.anaconda.org/nwani tensorflow-gpu
pip install pillow
pip install matplotlib
pip install numba
pip install scikit-learn==0.19 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install filterpy
pip install pandas
pip install h5py==2.10.0
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## To run
Before running, you need to download the yolo.h5 file and save it to the model_h5 folder.

1. Modify the configuration in the `vehicle_tracking.py` file:

```
DEFAULTS = {
        "model_path": './model_h5/yolo.h5',
        "anchors_path": './model_data/yolo_anchors.txt',
        "classes_path": './model_data/coco_classes.txt',
        "gpu_num": 1,
        "image": False,
        "tracker": True,
        "write_to_file": True,
        "input": './input/Demo2_tiny.mp4',
        "output": './output/Demo2_tiny.mp4',
        "output_path": './output/',
        "score": 0.4,  # threshold
        "iou": 0.4,  # threshold
        "repeat_iou": 0.95,  # threshold
    }
```

2. Run `vehicle_tracking.py`, the results can be viewed in the folder specified in `"output_path`.

```
python vehicle_tracking.py
```