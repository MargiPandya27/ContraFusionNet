# Contrastive Fusion of Transformer and CNN based Deep Features for Semantic Segmentation

![MA-14 drawio](https://github.com/MargiPandya27/ContraFusionNet/assets/117746681/8237bd41-356f-46f1-a19e-6fb45f2ec50f)


## Experiment Setup

* Ubuntu 20.04.5 LTS
* Python Version: Python 3.8.10
* Conda Version: conda 22.9.0
* Torch Version: 2.0.1+cu117
* Torchvision Version: 0.15.2+cu117
* GPU: 2 x NVIDIA GeForce RTX 3060 (12 GB)
    NVIDIA-SMI 515.105.01, Driver Version: 515.105.01, 
* CUDA Version: 11.7, 12GB


## Dataset Preparation
### Data preparation
You need to download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset.

Your directory tree should be look like this:
````bash
$SEG_ROOT/data
├── cityscapes
│   ├── gtFine
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── leftImg8bit
│       ├── test
│       ├── train
│       └── val
````

## Results
Performance on the Cityscapes dataset. The models are trained and tested with the input size of 512x512.
<div align="center">
  
  | Model                             | BackBone    | Params | Crop Size | mIOU  | Dice Co-efficient | Pixel Accuracy | Testing Time (per image) |
  |-----------------------------------|-------------|--------|-----------|-------|-------------------|-----------------|---------------------------|
  | U-net \cite{ronneberger2015u}      | -           | 34.5M  | 512x512   | 0.226 | 0.7822            | 0.978           | 0.18s                     |
  | DeepLabv3 \cite{chen2017deeplab}   | ResNet50    | 65.92M | 512x512   | 0.198 | 0.728             | 0.976           | 0.272s                    |
  | PSPNet \cite{zhao2017pyramid}      | ResNet50    | 48.7M  | 512x512   | 0.213 | 0.751             | 0.911           | 0.24s                     |
  | HRNet \cite{sun2019high}          | -           | 1.5M   | 512x512   | 0.228 | 0.692             | 0.976           | 0.174s                    |
  | SegFormer \cite{xie2021segformer} | -           | 17.8M  | 512x512   | 0.121 | 0.317             | 0.786           | 0.15s                     |
  | **ContraFusionNet**               | -           | 17.8M  | 512x512   | 0.351 | 0.833             | 0.9759          | 0.24s                     |

</div>

