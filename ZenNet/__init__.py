'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''


import os, sys
this_script_dir = os.path.dirname(os.path.abspath(__file__))

from . import masternet
import global_utils
import torch
import urllib.request

pretrain_model_pth_dir = os.path.expanduser('~/.cache/pytorch/checkpoints/zennet_pretrained')

zennet_model_zoo = {
    'zennet_cifar10_model_size05M_res32': {
        'plainnet_str_txt': 'zennet_cifar_model_size05M_res32.txt',
        'pth_path': 'zennet_cifar10_model_size05M_res32/best-params_rank0.pth',
        'num_classes': 10,
        'use_SE': False,
        'resolution': 32,
        'crop_image_size': 32,
        'pretrained_pth_url': 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/zennet_cifar10_model_size05M_res32/best-params_rank0.pth',
    },

    'zennet_cifar10_model_size1M_res32': {
        'plainnet_str_txt': 'zennet_cifar_model_size1M_res32.txt',
        'pth_path': 'zennet_cifar10_model_size1M_res32/best-params_rank0.pth',
        'num_classes': 10,
        'use_SE': False,
        'resolution': 32,
        'crop_image_size': 32,
        'pretrained_pth_url': 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/zennet_cifar10_model_size1M_res32/best-params_rank0.pth',
    },

    'zennet_cifar10_model_size2M_res32': {
        'plainnet_str_txt': 'zennet_cifar_model_size2M_res32.txt',
        'pth_path': 'zennet_cifar10_model_size2M_res32/best-params_rank0.pth',
        'num_classes': 10,
        'use_SE': False,
        'resolution': 32,
        'crop_image_size': 32,
        'pretrained_pth_url': 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/zennet_cifar10_model_size2M_res32/best-params_rank0.pth',
    },

    'zennet_cifar100_model_size05M_res32': {
        'plainnet_str_txt': 'zennet_cifar_model_size05M_res32.txt',
        'pth_path': 'zennet_cifar100_model_size05M_res32/best-params_rank0.pth',
        'num_classes': 100,
        'use_SE': False,
        'resolution': 32,
        'crop_image_size': 32,
        'pretrained_pth_url': 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/zennet_cifar100_model_size05M_res32/best-params_rank0.pth',
    },

    'zennet_cifar100_model_size1M_res32': {
        'plainnet_str_txt': 'zennet_cifar_model_size1M_res32.txt',
        'pth_path': 'zennet_cifar100_model_size1M_res32/best-params_rank0.pth',
        'num_classes': 100,
        'use_SE': False,
        'resolution': 32,
        'crop_image_size': 32,
        'pretrained_pth_url': 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/zennet_cifar100_model_size1M_res32/best-params_rank0.pth',
    },

    'zennet_cifar100_model_size2M_res32': {
        'plainnet_str_txt': 'zennet_cifar_model_size2M_res32.txt',
        'pth_path': 'zennet_cifar100_model_size2M_res32/best-params_rank0.pth',
        'num_classes': 100,
        'use_SE': False,
        'resolution': 32,
        'crop_image_size': 32,
        'pretrained_pth_url': 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/zennet_cifar100_model_size2M_res32/best-params_rank0.pth',
    },

    'zennet_imagenet1k_flops400M_SE_res224': {
        'plainnet_str_txt': 'zennet_imagenet1k_flops400M_res224.txt',
        'pth_path': 'iccv2021_zennet_imagenet1k_flops400M_SE_res224/student_best-params_rank0.pth',
        'num_classes': 1000,
        'use_SE': True,
        'resolution': 224,
        'crop_image_size': 320,
        'pretrained_pth_url': 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_flops400M_SE_res224/student_best-params_rank0.pth',
    },

    'zennet_imagenet1k_flops600M_SE_res224': {
        'plainnet_str_txt': 'zennet_imagenet1k_flops600M_res224.txt',
        'pth_path': 'iccv2021_zennet_imagenet1k_flops600M_SE_res224/student_best-params_rank0.pth',
        'num_classes': 1000,
        'use_SE': True,
        'resolution': 224,
        'crop_image_size': 320,
        'pretrained_pth_url': 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_flops600M_SE_res224/student_best-params_rank0.pth',
    },

    'zennet_imagenet1k_flops900M_SE_res224': {
        'plainnet_str_txt': 'zennet_imagenet1k_flops900M_res224.txt',
        'pth_path': 'zennet_imagenet1k_flops900M_SE_res224/student_best-params_rank0.pth',
        'num_classes': 1000,
        'use_SE': True,
        'resolution': 224,
        'crop_image_size': 320,
        'pretrained_pth_url': 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/zennet_imagenet1k_flops900M_SE_res224/student_best-params_rank0.pth',
    },

    'zennet_imagenet1k_latency01ms_res224': {
        'plainnet_str_txt': 'zennet_imagenet1k_latency01ms_res224.txt',
        'pth_path': 'iccv2021_zennet_imagenet1k_latency01ms_res224/student_best-params_rank0.pth',
        'num_classes': 1000,
        'use_SE': False,
        'resolution': 224,
        'crop_image_size': 320,
        'pretrained_pth_url': 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_latency01ms_res224/student_best-params_rank0.pth',
    },

    'zennet_imagenet1k_latency02ms_res224': {
        'plainnet_str_txt': 'zennet_imagenet1k_latency02ms_res224.txt',
        'pth_path': 'iccv2021_zennet_imagenet1k_latency02ms_res224/student_best-params_rank0.pth',
        'num_classes': 1000,
        'use_SE': False,
        'resolution': 224,
        'crop_image_size': 320,
        'pretrained_pth_url': 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_latency02ms_res224/student_best-params_rank0.pth',
    },

    'zennet_imagenet1k_latency03ms_res224': {
        'plainnet_str_txt': 'zennet_imagenet1k_latency03ms_res224.txt',
        'pth_path': 'iccv2021_zennet_imagenet1k_latency03ms_res224/student_best-params_rank0.pth',
        'num_classes': 1000,
        'use_SE': False,
        'resolution': 224,
        'crop_image_size': 320,
        'pretrained_pth_url': 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_latency03ms_res224/student_best-params_rank0.pth',
    },

    'zennet_imagenet1k_latency05ms_res224': {
        'plainnet_str_txt': 'zennet_imagenet1k_latency05ms_res224.txt',
        'pth_path': 'iccv2021_zennet_imagenet1k_latency05ms_res224/student_best-params_rank0.pth',
        'num_classes': 1000,
        'use_SE': False,
        'resolution': 224,
        'crop_image_size': 320,
        'pretrained_pth_url': 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_latency05ms_res224/student_best-params_rank0.pth',
    },

    'zennet_imagenet1k_latency08ms_res224': {
        'plainnet_str_txt': 'zennet_imagenet1k_latency08ms_res224.txt',
        'pth_path': 'iccv2021_zennet_imagenet1k_latency08ms_res224/student_best-params_rank0.pth',
        'num_classes': 1000,
        'use_SE': False,
        'resolution': 224,
        'crop_image_size': 320,
        'pretrained_pth_url': 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_latency08ms_res224/student_best-params_rank0.pth',
    },
'zennet_imagenet1k_latency12ms_res224': {
        'plainnet_str_txt': 'zennet_imagenet1k_latency12ms_res224.txt',
        'pth_path': 'iccv2021_zennet_imagenet1k_latency12ms_res224/student_best-params_rank0.pth',
        'num_classes': 1000,
        'use_SE': False,
        'resolution': 224,
        'crop_image_size': 380,
        'pretrained_pth_url': 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_latency12ms_res224/student_best-params_rank0.pth',
    },
}

def get_ZenNet(model_name, pretrained=False):
    if model_name not in zennet_model_zoo:
        print('Error! Cannot find ZenNet model name! Please choose one in the following list:')

        for key in zennet_model_zoo:
            print(key)
        raise ValueError('ZenNet Model Name not found: ' + model_name)

    model_plainnet_str_txt = os.path.join(this_script_dir, zennet_model_zoo[model_name]['plainnet_str_txt'])
    model_pth_path = os.path.join(pretrain_model_pth_dir, zennet_model_zoo[model_name]['pth_path'])
    pretrained_pth_url = zennet_model_zoo[model_name]['pretrained_pth_url']
    use_SE = zennet_model_zoo[model_name]['use_SE']
    num_classes = zennet_model_zoo[model_name]['num_classes']

    with open(model_plainnet_str_txt, 'r') as fid:
        model_plainnet_str = fid.readline().strip()

    model = masternet.PlainNet(num_classes=num_classes, plainnet_struct=model_plainnet_str, use_se=use_SE)

    if pretrained:

        if not os.path.isfile(model_pth_path):
            print('downloading pretrained parameters from ' + pretrained_pth_url)
            global_utils.mkfilepath(model_pth_path)
            urllib.request.urlretrieve(url=pretrained_pth_url, filename=model_pth_path)
            print('pretrained model parameters cached at ' + model_pth_path)

        print('loading pretrained parameters...')
        checkpoint = torch.load(model_pth_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)

    return model