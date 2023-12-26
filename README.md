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
You need to download the [Cityscapes](https://www.cityscapes-dataset.com/) datasets.

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

## Loss Function
![image](https://github.com/MargiPandya27/ContraFusionNet/assets/117746681/42dceb2d-cfed-42e8-b474-4474f4ff85d9)

![image](https://github.com/MargiPandya27/ContraFusionNet/assets/117746681/2ab4a5d5-ed4b-4fee-8276-59eec32485d5)


## Results

![image](https://github.com/MargiPandya27/ContraFusionNet/assets/117746681/1ded4f57-f432-4bc7-9dc8-80de214a6289)

![image](https://github.com/MargiPandya27/ContraFusionNet/assets/117746681/a1e862fd-9260-4214-b6a8-95f9b8938117)

