'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch, argparse
from torch import nn
import torch.nn.functional as F
import PlainNet
from PlainNet import parse_cmd_options, basic_blocks, super_blocks


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_BN', action='store_true')
    parser.add_argument('--no_reslink', action='store_true')
    parser.add_argument('--use_se', action='store_true')
    parser.add_argument('--dropout', type=float, default=None)
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt


class PlainNet(PlainNet.PlainNet):
    def __init__(self, argv=None, opt=None, num_classes=None, plainnet_struct=None, no_create=False,
                 no_reslink=None, no_BN=None, use_se=None, dropout=None,
                 **kwargs):

        if argv is not None:
            module_opt = parse_cmd_options(argv)
        else:
            module_opt = None

        if no_BN is None:
            if module_opt is not None:
                no_BN = module_opt.no_BN
            else:
                no_BN = False

        if no_reslink is None:
            if module_opt is not None:
                no_reslink = module_opt.no_reslink
            else:
                no_reslink = False

        if use_se is None:
            if module_opt is not None:
                use_se = module_opt.use_se
            else:
                use_se = False

        if dropout is None:
            if module_opt is not None:
                self.dropout = module_opt.dropout
            else:
                self.dropout = None
        else:
            self.dropout = dropout

        if self.dropout is not None:
            print('--- using dropout={:4g}'.format(self.dropout))

        super(PlainNet, self).__init__(argv=argv, opt=opt, num_classes=num_classes, plainnet_struct=plainnet_struct,
                                       no_create=no_create, no_reslink=no_reslink, no_BN=no_BN, use_se=use_se, **kwargs)
        self.last_channels = self.block_list[-1].out_channels
        self.fc_linear = basic_blocks.Linear(in_channels=self.last_channels, out_channels=self.num_classes, no_create=no_create)

        self.no_create = no_create
        self.no_reslink = no_reslink
        self.no_BN = no_BN
        self.use_se = use_se

        # bn eps
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eps = 1e-3

    def extract_stage_features_and_logit(self, x, target_downsample_ratio=None):
        stage_features_list = []
        image_size = x.shape[2]
        output = x

        for block_id, the_block in enumerate(self.block_list):
            output = the_block(output)
            if self.dropout is not None:
                dropout_p = float(block_id) / len(self.block_list) * self.dropout
                output = F.dropout(output, dropout_p, training=self.training, inplace=True)
            dowsample_ratio = round(image_size / output.shape[2])
            if dowsample_ratio == target_downsample_ratio:
                stage_features_list.append(output)
                target_downsample_ratio *= 2
            pass
        pass

        output = F.adaptive_avg_pool2d(output, output_size=1)
        # if self.dropout is not None:
        #     output = F.dropout(output, self.dropout, training=self.training, inplace=True)
        output = torch.flatten(output, 1)
        logit = self.fc_linear(output)
        return stage_features_list, logit

    def forward(self, x):
        output = x
        for block_id, the_block in enumerate(self.block_list):
            output = the_block(output)
            if self.dropout is not None:
                dropout_p = float(block_id) / len(self.block_list) * self.dropout
                output = F.dropout(output, dropout_p, training=self.training, inplace=True)

        output = F.adaptive_avg_pool2d(output, output_size=1)
        if self.dropout is not None:
            output = F.dropout(output, self.dropout, training=self.training, inplace=True)
        output = torch.flatten(output, 1)
        output = self.fc_linear(output)
        return output

    def forward_pre_GAP(self, x):
        output = x
        for the_block in self.block_list:
            output = the_block(output)
        return output

    def get_FLOPs(self, input_resolution):
        the_res = input_resolution
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(the_res)
            the_res = the_block.get_output_resolution(the_res)

        the_flops += self.fc_linear.get_FLOPs(the_res)

        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        the_size += self.fc_linear.get_model_size()

        return the_size

    def get_num_layers(self):
        num_layers = 0
        for block in self.block_list:
            assert isinstance(block, super_blocks.PlainNetSuperBlockClass)
            num_layers += block.sub_layers
        return num_layers

    def replace_block(self, block_id, new_block):
        self.block_list[block_id] = new_block

        if block_id < len(self.block_list) - 1:
            if self.block_list[block_id + 1].in_channels != new_block.out_channels:
                self.block_list[block_id + 1].set_in_channels(new_block.out_channels)
        else:
            assert block_id == len(self.block_list) - 1
            self.last_channels = self.block_list[-1].out_channels
            if self.fc_linear.in_channels != self.last_channels:
                self.fc_linear.set_in_channels(self.last_channels)

        self.module_list = nn.ModuleList(self.block_list)

    def split(self, split_layer_threshold):
        new_str = ''
        for block in self.block_list:
            new_str += block.split(split_layer_threshold=split_layer_threshold)
        return new_str

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=3.26033)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 3.26033 * np.sqrt(2 / (m.weight.shape[0] + m.weight.shape[1])))
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                pass

        for superblock in self.block_list:
            if not isinstance(superblock, super_blocks.PlainNetSuperBlockClass):
                continue
            for block in superblock.block_list:
                if not isinstance(block, basic_blocks.ResBlock):
                    continue
                # print('---debug set bn weight zero in resblock {}:{}'.format(superblock, block))
                last_bn_block = None
                for inner_resblock in block.block_list:
                    if isinstance(inner_resblock, basic_blocks.BN):
                        last_bn_block = inner_resblock
                    pass
                pass  # end for
                assert last_bn_block is not None
                # print('-------- last_bn_block={}'.format(last_bn_block))
                nn.init.zeros_(last_bn_block.netblock.weight)