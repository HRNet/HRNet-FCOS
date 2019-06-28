# High-resolution Networks for FCOS

## Introduction
This project contains the code of HRNet-FCOS, i.e., using [High-resolution Networks (HRNets)](https://arxiv.org/pdf/1904.04514.pdf) as the backbones for the [Fully Convolutional One-Stage Object Detection (FCOS)](https://arxiv.org/abs/1904.01355) algorithm, which achieves much better object detection performance compared with the ResNet-FCOS counterparts while keeping a similar computation complexity. For more projects using HRNets, please go to our [website](https://github.com/HRNet).

## Quick start
### Installation

Please check [INSTALL.md](INSTALL.md) for installation instructions.
You may also want to see the original [README.md](FCOS_README.md) of FCOS.

### Inference
The inference command line on coco minival split:

    python tools/test_net.py \
        --config-file configs/fcos/fcos_hrnet_w32_5l_2x.yaml \
        MODEL.WEIGHT models/FCOS_hrnet_w32_5l_2x.pth \
        TEST.IMS_PER_BATCH 8

Please note that:
1) If your model's name is different, please replace `models/FCOS_hrnet_w32_5l_2x.pth` with your own.
2) If you enounter out-of-memory error, please try to reduce `TEST.IMS_PER_BATCH` to 1.
3) If you want to evaluate a different model, please change `--config-file` to its config file (in [configs/fcos](configs/fcos)) and `MODEL.WEIGHT` to its weights file.

For your convenience, we provide the following trained models.

FCOS Model | Training mem (GB) | Multi-scale training | SyncBN| Testing time / im | # params |GFLOPs| AP (minival) | Link
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
ResNet_50_5l_2x           | 29.3 | No  |No | 71ms  |32.0M |190.0| 37.1 | [-]()
HRNet_W18_5l_2x           | 54.4 | No  |No | 72ms  |17.5M |180.3| 37.7 | [model](https://1drv.ms/u/s!Av9x_1oQAAoqeRND03CfS4HBONM?e=wT0G0O)
||
ResNet_50_6l_2x           | 58.2 | No  |No | 98ms  |32.7M |529.0| 37.1 | [-]()
HRNet_W18_6l_2x           | 88.1 | No  |No | 106ms |18.1M |515.1| 37.8 | [model](https://1drv.ms/u/s!Av9x_1oQAAoqeumBsKzXzZjE8Qs?e=fT1edk)
||
ResNet_101_5l_2x          | 44.1 | Yes |No | 74ms  |51.0M |261.2| 41.4 | [model](https://cloudstor.aarnet.edu.au/plus/s/vjL3L0AW7vnhRTo/download)
HRNet_W32_5l_2x           | 78.9 | Yes |No | 87ms  |37.3M |273.3| 41.9 | [model](https://1drv.ms/u/s!Av9x_1oQAAoqfPuN69wCHx26k0o?e=L7c5FX)
||
ResNet_101_6l_2x          | 71.0 | Yes |No | 121ms |51.6M |601.0| 41.5 | [model](https://1drv.ms/u/s!Av9x_1oQAAoqe7UH3Bh-kB8JuKA?e=EF9K0B)
HRNet_W32_6l_2x           | 108.6| Yes |No | 125ms |37.9M |608.0| 42.1 | [model](https://1drv.ms/u/s!Av9x_1oQAAoqfZn3Xt2CrKUI2rk?e=ZdJSPG)
||
HRNet_W40_6l_3x           | 128.0| Yes |No | 142ms |54.1M |682.9| 42.6 | [model](https://1drv.ms/u/s!Av9x_1oQAAoqfu-2x6aOIsGxSsg?e=OBbs5Z)

[1] *1x and 2x mean the model is trained for 90K and 180K iterations, respectively.*\
[2] *5l and 6l denote that we use feature pyramid with 5 levels and 6 levels, respectively.*\
[3] *We report total training memory footprint on all GPUs instead of the memory footprint per GPU as in maskrcnn-benchmark.*\
[4] *The inference speed of HRNet can get improved if the branches in the HRNet model can run in parallel.*\
[5] *All results are obtained with a single model and without any test time data augmentation.*

### Training

The following command line will trains a fcos_hrnet_w32_5l_2x model on 8 GPUs with Synchronous Stochastic Gradient Descent (SGD):

    python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=$((RANDOM + 10000)) \
        tools/train_net.py \
        --config-file configs/fcos/fcos_hrnet_w32_5l_2x.yaml \
        MODEL.WEIGHT hrnetv2_w32_imagenet_pretrained.pth \
        MODEL.SYNCBN False \
        DATALOADER.NUM_WORKERS 4 \
        OUTPUT_DIR training_dir/fcos_hrnet_w32_5l_2x
        
Note that:
1) If you want to use fewer GPUs, please change `--nproc_per_node` to the number of GPUs. No other settings need to be changed. The total batch size does not depends on `nproc_per_node`. If you want to change the total batch size, please change `SOLVER.IMS_PER_BATCH` in [configs/fcos/fcos_hrnet_w32_5l_2x.yaml](configs/fcos/fcos_hrnet_w32_5l_2x.yaml).
2) If you want to use Synchronous Batch-Normalization (SyncBN), please change `MODEL.SYNCBN` to `True`. Note that this will lead to ~2x slower training speed when training on mulitple machines. You also need to fix the image padding size when using SyncBN, see [here](maskrcnn_benchmark/structures/image_list.py#L62).
3) The imagenet pre-trained model can be found [here](https://github.com/HRNet/HRNet-Object-Detection#faster-r-cnn).
4) The models will be saved into `OUTPUT_DIR`.
5) If you want to train FCOS on your own dataset, please follow this instruction [#54](https://github.com/tianzhi0549/FCOS/issues/54#issuecomment-497558687).
### Contributing to the project

Any pull requests or issues are welcome.

### Citations
Please consider citing the following papers in your publications if the project helps your research. 
```
@article{sun2019deep,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  journal={arXiv preprint arXiv:1902.09212},
  year={2019}
}

@article{tian2019fcos,
  title   =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author  =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  journal =  {arXiv preprint arXiv:1904.01355},
  year    =  {2019}
}
```


### License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors. 
