# MIMO is All You Need

This is a Pytorch implementation of [MIMO is All You Need](https://arxiv.org/abs/2212.04655) , a Transformer-based architexture for video prediction as described in the following paper: 

**MIMO Is All You Need : A Strong Multi-In-Multi-Out Baseline for Video Prediction**, by Shuliang Ning, Mengcheng Lan, Yanran Li, Chaofeng Chen, Qian Chen, Xunlai Chen, Xiaoguang Han and Shuguang Cui. 

### Install

~~~bash
pip install -r requirment.txt
~~~

### Datasets

We conduct experiments on four video datasets: [MNIST](https://pan.baidu.com/s/1n-r_0BuBa1XjDpEAX-nm1Q?pwd=lnnj) (passwd:lnnj), [Human3.6M](http://vision.imar.ro/human3.6m/description.php), Weather, and [KITTI](https://pan.baidu.com/s/17xhuN8Ad3IjGSCeaGH7wVw?pwd=bfar ) (passwd:bfar). 

For video format datasets, we extract frames from original video clips. 

### Training

Use the train.py scipt to train the model. To train the default model on Moving MNIST dataset, you need to download the MNIST dataset, and change data directory in `--root`, then just run:

~~~bash
python train.py
~~~

To train on your own dataset, just change the dataloader. 

The check point will be saved in `--save_dir` and the generated frames will be saved in the `--gen_frm_dir` folder. 

### Checkpoints

The checkpoint for MNIST is [Here](https://pan.baidu.com/s/1h548ndTYbYpHThTT7ed5vQ) (passwd:chpo)

### Prediction samples

The comparison between MIMO-VP and other two methods. 

30 frames are predicted given the last 10 frames. 

![](https://s21.aconvert.com/convert/p3r68-cdx67/wy8gc-cmku7.gif)





### Citation

@inproceedings{ning2023mimo,
  title={MIMO is all you need: a strong multi-in-multi-out baseline for video prediction},
  author={Ning, Shuliang and Lan, Mengcheng and Li, Yanran and Chen, Chaofeng and Chen, Qian and Chen, Xunlai and Han, Xiaoguang and Cui, Shuguang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={2},
  pages={1975--1983},
  year={2023}
}







 





