# CNSPN

This is code of this IEEE TGRS paper "Improving Few-Shot Remote Sensing Scene Classification with Class Name Semantics"

Once the paper is accepted,we will released all the code.

# paper
The paper link is [CNSPN](http:///.....)  待填写


With ResNet-18 backbone on NWPU RESISC45 and RSD46 WHU dataset.

***For other backbones (Conv4), Please implement it yourself.***


# Results

1-shot on the NWPU RESISC45 dataset: 70.35%

1-shot on the RSD46 WHU dataset: 59.73%



# How to train and test CNSPN model

## Environment installation
请安装python 3.6 和pytorch1.8
* python 3
* pytorch 1.8
## 数据准备

1. Download the images: https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE

2. Make a folder `materials/images` and put those images into it.

3. 先下载数据
4. 放到data文件夹下
## train and test
5. 从头开始训练，或者使用训练好的模型进行测试。

### for  NWPU RESISC45 dataset
`python train.py`
`python test.py` 

### for RSD46 WHU dataset
`python train.py --shot 5 --train-way 20 --save-path ./save/proto-5`

`python test.py --load ./save/proto-5/max-acc.pth --shot 5`

# Acknowledgment
This code is heavily borrowed from XX .

# Note
- 10/10/2022: code released
