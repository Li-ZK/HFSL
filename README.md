## Heterogeneous Few-shot Learning for Hyperspectral Image Classification
This is a code demo for the paper "Heterogeneous Few-shot Learning for Hyperspectral Image Classification"
Yan Wang, Ming Liu, Zhaokui Li, Qian Du, Yushi Chen, Fei Li, and Haibo Yang, Heterogeneous Few-shot Learning for Hyperspectral Image Classification, IEEE Geoscience and Remote Sensing Letters, in press.

## Requirements
CUDA = 10.2
Python = 3.7 
Pytorch = 1.5 
sklearn = 0.24.0
numpy = 1.19.2

## dataset
You can download the hyperspectral datasets in mat format at: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes, and move the files to `./datasets` folder.
You can also download the hyperspectral datasets from the following link.
Link: https://pan.baidu.com/s/1k8by5CiyabXRJdD_MVOL1w 
Extract code: wjvv

The mini-ImageNet data sets can be downloaded from the following link:
Link: https://pan.baidu.com/s/1Mn1en9EhfFvE-i62YnbwhQ
Extract code: 54DO

An example dataset folder has the following structure:
```
datasets
├── IP
│   ├── indian_pines_corrected.mat
│   ├── indian_pines_gt.mat
└── paviaU
│   ├── paviaU_gt.mat
│   ├── paviaU.mat
└──Houston
│   ├── mask_train.mat
│   ├── mask_train.mat
│   ├── data.mat
└──miniImagenet
│   ├── 
│   ├── 

## Usage:
Take HFSL method on the PU dataset as an example: 
1. Download the required data set and move to folder`./datasets`.
2. To run the file, you need to download the VGG pre-training weights file (vgg16_bn-6c64b313.pth).
   The VGG pre-training weight file can be downloadfrom the following link:
   Link: https://pan.baidu.com/s/1af--So40MKjhWdFuIcVyKg 
   Extract code：0tdu
3. Taking 5 labeled samples per class as an example, run `mini2hsi-SS-5-PU.py --nlabel 5 `. 

 * `--nlabel` denotes the number of labeled samples per class for the HSI data set.
