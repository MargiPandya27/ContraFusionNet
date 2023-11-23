# DeepFeatureNet


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

We experiment on major benchmark dataset: Cityscapes 


### Expected dataset structure for [Cityscapes](https://www.cityscapes-dataset.com/downloads/)

```text
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
    # below are generated Cityscapes panoptic annotation
    cityscapes_panoptic_train.json
    cityscapes_panoptic_train/
    cityscapes_panoptic_val.json
    cityscapes_panoptic_val/
    cityscapes_panoptic_test.json
    cityscapes_panoptic_test/
  leftImg8bit/
    train/
    val/
    test/
```

- Login and download the dataset

  ```bash
  wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=myusername&password=mypassword&submit=Login' https://www.cityscapes-dataset.com/login/
  ######## gtFine
  wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
  ######## leftImg8bit
  wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
  ```

- Install cityscapes scripts by:

  ```bash
  pip install git+https://github.com/mcordts/cityscapesScripts.git
  ```

- To create labelTrainIds.png, first prepare the above structure, then run cityscapesescript with:

  ```bash
  git clone https://github.com/mcordts/cityscapesScripts.git
  ```

  ```bash
  CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesScripts/cityscapesscripts/preparation/createTrainIdLabelImgs.py
  ```

  These files are not needed for instance segmentation.

- To generate Cityscapes panoptic dataset, run cityscapesescript with:

  ```bash
  CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesScripts/cityscapesscripts/preparation/createPanopticImgs.py
  ```

  These files are not needed for semantic and instance segmentation.
## Results
## Citation
## Acknowledgement
