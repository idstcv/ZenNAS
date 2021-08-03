[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](http://img.shields.io/badge/cs.CV-arXiv%3A2004.08955-B31B1B.svg)](https://arxiv.org/abs/2102.01063)

# ZenNAS: A Zero-Shot NAS for High-Performance Deep Image Recognition

ZenNAS is a lighting fast Neural Architecture Searching (NAS) algorithm for automatically designing deep neural networks with high prediction accuracy and high inference speed on GPU and mobile device.

Our paper is available here: [arXiv link](https://arxiv.org/abs/2102.01063)

## Update

This work is accepted by ICCV 2021. Will release the searching and training code by the end of August.

## How Fast It IS

Using 1 GPU searching for 12 hours, ZenNAS is able to design networks of ImageNet top-1 accuracy comparable to EfficientNet-B5 (\~83.6%) while inference speed 4.9x times faster on V100, 10x times faster on NVIDIA T4, 1.6x times faster on Google Pixel2.

![Inference Speed](./misc/ZenNet_speed.png)

## Examples

To evaluate the pre-trained model on ImageNet using GPU 0:

``` bash
python val.py --fp16 --gpu 0 --arch ${zennet_model_name}
```

where ${zennet\_model\_name} should be replaced by a valid ZenNet model name. The complete list of model names can be found in 'Pre-trained Models' section.

To evaluate the pre-trained model on CIFAR10 or CIFAR100 using GPU 0:

``` bash
python val_cifar.py --dataset cifar10 --gpu 0 --arch ${zennet_model_name}
```

To create a ZenNet in your python code:

``` python
gpu=0
model = ZenNet.get_ZenNet(opt.arch, pretrained=True)
torch.cuda.set_device(gpu)
torch.backends.cudnn.benchmark = True
model = model.cuda(gpu)
model = model.half()
model.eval()
```

## System Requirement and Default Paths

* PyTorch >= 1.5, Python >= 3.7
* By default, ImageNet dataset is stored under \~/data/imagenet; CIFAR10/CIFAR100 is stored under \~/data/pytorch\_cifar10 or \~/data/pytorch\_cifar100
* Pre-trained parameters are cached under \~/.cache/pytorch/checkpoints/zennet\_pretrained

## Pre-trained Models

We provided pre-trained models on ImageNet and CIFAR10/CIFAR100.

### ImageNet Models

| model | resolution | \# params | FLOPs | Top-1 Acc | V100 | T4 | Pixel2 |
| ----- | ---------- | -------- | ----- | --------- | ---- | --- | ------ |
| [zennet\_imagenet1k\_flops400M\_SE\_res224](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/iccv2021_zennet_imagenet1k_flops400M_SE_res224/student_best-params_rank0.pth) | 224 | 5.7M | 410M | 78.0% | 0.25 | 0.39 | 87.9 |
| [zennet\_imagenet1k\_flops600M\_SE\_res224](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_flops600M_SE_res224/student_best-params_rank0.pth) | 224 | 7.1M | 611M | 79.1% | 0.36 | 0.52 | 128.6 |
| [zennet\_imagenet1k\_flops900M\_SE\_res224](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/zennet_imagenet1k_flops900M_SE_res224/student_best-params_rank0.pth) | 224 | 19.4M | 934M | 80.8% | 0.55 | 0.55 | 215.7 |
| [zennet\_imagenet1k\_latency01ms\_res224](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_latency01ms_res224/student_best-params_rank0.pth) | 224 | 30.1M | 1.7B | 77.8% | 0.1 | 0.08 | 181.7 |
| [zennet\_imagenet1k\_latency02ms\_res224](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_latency02ms_res224/student_best-params_rank0.pth) | 224 | 49.7M | 3.4B | 80.8% | 0.2 | 0.15 | 357.4 |
| [zennet\_imagenet1k\_latency03ms\_res224](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_latency03ms_res224/student_best-params_rank0.pth) | 224 | 85.4M | 4.8B | 81.5% | 0.3 | 0.20 | 517.0 |
| [zennet\_imagenet1k\_latency05ms\_res224](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_latency05ms_res224/student_best-params_rank0.pth) | 224 | 118M | 8.3B | 82.7% | 0.5 | 0.30 | 798.7 |
| [zennet\_imagenet1k\_latency08ms\_res224](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_latency08ms_res224/student_best-params_rank0.pth) | 224 | 183M | 13.9B | 83.0% | 0.8 | 0.57 | 1365 |
| [zennet\_imagenet1k\_latency12ms\_res224](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_latency12ms_res224/student_best-params_rank0.pth) | 224 | 180M | 22.0B | 83.6% | 1.2 | 0.85 | 2051 |
| EfficientNet-B3 | 300 | 12.0M | 1.8B | 81.1% | 1.12 | 1.86 | 569.3 |
| EfficientNet-B5 | 456 | 30.0M | 9.9B | 83.3% | 4.5 | 7.0 | 2580 |
| EfficientNet-B6 | 528 | 43M | 19.0B | 84.0% | 7.64 | 12.3 | 4288 |

* 'V100' is the inference latency on NVIDIA V100 in milliseconds, benchmarked at batch size 64, float16.
* 'T4' is the inference latency on NVIDIA T4 in milliseconds, benchmarked at batch size 64, TensorRT INT8.
* 'Pixel2' is the inference latency on Google Pixel2 in milliseconds, benchmarked at single image.

### CIFAR10/CIFAR100 Models

| model | resolution | \# params | FLOPs | Top-1 Acc |
| ----- | ---------- | -------- | ----- | --------- |
| [zennet\_cifar10\_model\_size05M\_res32](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/zennet_cifar10_model_size05M_res32/best-params_rank0.pth) | 32 | 0.5M | 140M | 96.2% |
| [zennet\_cifar10\_model\_size1M\_res32](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/zennet_cifar10_model_size1M_res32/best-params_rank0.pth) | 32 | 1.0M | 162M | 96.2% |
| [zennet\_cifar10\_model\_size2M\_res32](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/zennet_cifar10_model_size2M_res32/best-params_rank0.pth) | 32 | 2.0M | 487M | 97.5% |
| [zennet\_cifar100\_model\_size05M\_res32](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/zennet_cifar100_model_size05M_res32/best-params_rank0.pth) | 32 | 0.5M | 140M | 79.9% |
| [zennet\_cifar100\_model\_size1M\_res32](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/zennet_cifar100_model_size1M_res32/best-params_rank0.pth) | 32 | 1.0M | 162M | 80.1% |
| [zennet\_cifar100\_model\_size2M\_res32](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/zennet_cifar100_model_size2M_res32/best-params_rank0.pth) | 32 | 2.0M | 487M | 84.4% |

## Major Contributor

* Ming Lin ([Home Page](https://minglin-home.github.io/), [linming04@gmail.com](mailto:linming04@gmail.com))
* Pichao Wang ([pichao.wang@alibaba-inc.com](mailto:pichao.wang@alibaba-inc.com))
* Zhenhong Sun ([zhenhong.szh@alibaba-inc.com](mailto:zhenhong.szh@alibaba-inc.com))
* Hesen Chen ([hesen.chs@alibaba-inc.com](mailto:hesen.chs@alibaba-inc.com))

## Copyright

Copyright (C) 2010-2021 Alibaba Group Holding Limited.
