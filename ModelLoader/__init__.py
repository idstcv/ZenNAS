'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.

The geffnet module is modified from:
https://github.com/rwightman/gen-efficientnet-pytorch
'''


import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from . import geffnet
from . import myresnet

import torchvision.models

torchvision_model_name_list = sorted(name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.__dict__[name]))

def _get_model_(arch, num_classes, pretrained=False, opt=None, argv=None):

    # load torch vision model
    if arch in torchvision_model_name_list:
        if pretrained:
            print('Using pretrained model: {}'.format(arch))
            model = torchvision.models.__dict__[arch](pretrained=True, num_classes=num_classes)
        else:
            print('Create torchvision model: {}'.format(arch))
            model = torchvision.models.__dict__[arch](num_classes=num_classes)

        # my implementation of resnet
    elif arch == 'myresnet18':
        print('Create model: {}'.format(arch))
        model = myresnet.resnet18(pretrained=False, opt=opt, argv=argv)
    elif arch == 'myresnet34':
        print('Create model: {}'.format(arch))
        model = myresnet.resnet34(pretrained=False, opt=opt, argv=argv)
    elif arch == 'myresnet50':
        print('Create model: {}'.format(arch))
        model = myresnet.resnet50(pretrained=False, opt=opt, argv=argv)
    elif arch == 'myresnet101':
        print('Create model: {}'.format(arch))
        model = myresnet.resnet101(pretrained=False, opt=opt, argv=argv)
    elif arch == 'myresnet152':
        print('Create model: {}'.format(arch))
        model = myresnet.resnet152(pretrained=False, opt=opt, argv=argv)

    # geffnet
    elif arch.startswith('geffnet_'):
        model_name = arch[len('geffnet_'):]
        model = geffnet.create_model(model_name, pretrained=pretrained)


    # PlainNet
    elif arch == 'PlainNet':
        import PlainNet
        print('Create model: {}'.format(arch))
        model = PlainNet.PlainNet(num_classes=num_classes, opt=opt, argv=argv)

    # Any PlainNet
    elif arch.find('.py:MasterNet') >= 0:
        module_path = arch.split(':')[0]
        assert arch.split(':')[1] == 'MasterNet'
        my_working_dir = os.path.dirname(os.path.dirname(__file__))
        module_full_path = os.path.join(my_working_dir, module_path)

        import importlib.util
        spec = importlib.util.spec_from_file_location('AnyPlainNet', module_full_path)
        AnyPlainNet = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(AnyPlainNet)
        print('Create model: {}'.format(arch))
        model = AnyPlainNet.MasterNet(num_classes=num_classes, opt=opt, argv=argv)

    else:
        raise ValueError('Unknown model arch: ' + arch)

    return model

def get_model(opt, argv):
    return _get_model_(arch=opt.arch, num_classes=opt.num_classes, pretrained=opt.pretrained, opt=opt, argv=argv)