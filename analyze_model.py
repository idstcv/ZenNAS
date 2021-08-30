'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''


import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import ModelLoader
import global_utils
from ptflops import get_model_complexity_info


def main(opt, argv):
    model = ModelLoader.get_model(opt, argv)
    flops, params = get_model_complexity_info(model, (3, opt.input_image_size, opt.input_image_size),
                                              as_strings=False,
                                              print_per_layer_stat=True)
    print('Flops:  {:4g}'.format(flops))
    print('Params: {:4g}'.format(params))

if __name__ == "__main__":
    opt = global_utils.parse_cmd_options(sys.argv)

    main(opt, sys.argv)