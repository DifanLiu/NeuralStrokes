# Neural Strokes: Stylized Line Drawing of 3D Shapes

This repository contains the PyTorch implementation for [ICCV 2021](http://iccv2021.thecvf.com/) Paper "Neural Strokes: Stylized Line Drawing of 3D Shapes" by [Difan Liu](https://people.cs.umass.edu/~dliu/), [Matthew Fisher](https://techmatt.github.io/), [Aaron Hertzmann](https://www.dgp.toronto.edu/~hertzman/), [Evangelos Kalogerakis](https://people.cs.umass.edu/~kalo/).

### Dependency
- The project is developed on Ubuntu 16.04 with cuda10.0 + cudnn7.6. The code has been tested with PyTorch 1.2.0 (GPU version) and Python 3.7.6. 
- Python packages:
    - [diffvg](https://github.com/BachiLi/diffvg)
    - OpenCV (tested with 4.3.0)
    - PyYAML (tested with 5.3.1)
    - scikit-image (tested with 0.17.2)
    - scipy (tested with 1.4.1)

### Dataset and Weights
- Pre-trained model is available [here](https://www.dropbox.com/s/i5vvs4i0ook1fll/weights.zip?dl=0), please put it in `weights`:
    ```
    cd weights
    unzip weights.zip
    ```

- Dataset is available [here](https://www.dropbox.com/s/c8hlnumiys4b5yf/datasets.zip?dl=0), please put it in `datasets`:
    ```
    cd datasets
    unzip datasets.zip
    ```

### Preprocessing
- Preprocess raw data:
    ```python
    python preprocess.py -s datasets/style_01
    ```
   `-s`: path to the data of a single style.

### Test
- Test:
    ```python
   python test.py -d datasets/style_01/test_2 -g weights/style_01/SG.pth -t weights/style_01/ST.pth -s results/style_01__test_2.png
    ```
   `-d`: path to testing data.
   `-g`: path to Stroke Geometry checkpoint.
   `-t`: path to Stroke Texture checkpoint.
   `-s`: path to save the synthesized image.
### Training
- Start the Stroke Geometry (SG) training:
    ```python
    python train_SG.py -d datasets/style_01/train -n style_01__SG
    ```
- After the Stroke Geometry (SG) training is finished, start the Stroke Texture (ST) training:
    ```python
    python train_ST.py -d datasets/style_01/train -n style_01__ST
    ```
   `-d`: path to training data.
   `-n`: name of the experiment.
   
### Cite:
```
@InProceedings{Liu_2021_ICCV,
author={Liu, Difan and Fisher, Matthew and Hertzmann, Aaron and Kalogerakis, Evangelos},
title={Neural Strokes: Stylized Line Drawing of 3D Shapes},
booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
year = {2021}
}
```

### Contact
To ask questions, please [email](mailto:dliu@cs.umass.edu).
