'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import os
import distutils.dir_util
import pprint, ast, argparse, logging

import numpy as np

import torch
import uuid

def equally_assign_job_package(job_value_list, num_workers):
    job_package_list = []
    assigned_job_value_list = [0] * num_workers
    for worker_id in range(num_workers):
        job_package_list.append([])
    value_sorted_idx_list = np.argsort(job_value_list).tolist()
    value_sorted_idx_list.reverse()
    for idx in value_sorted_idx_list:
        min_assigned_job_value = min(assigned_job_value_list)
        min_idx = assigned_job_value_list.index(min_assigned_job_value)
        job_package_list[min_idx].append(idx)
        assigned_job_value_list[min_idx] += job_value_list[idx]

    return job_package_list

def filter_dict_list(dict_list, **kwargs):
    new_list = dict_list

    for key, value in kwargs.items():
        if len(new_list) == 0:
            return []
        new_list = [x for x in new_list if (isinstance(x[key], float) and abs(x[key] - value) < 1e-6) or  x[key] == value]

    return new_list

def load_py_module_from_path(module_path, module_name=None):
    if module_path.find(':') > 0:
        split_path = module_path.split(':')
        module_path = split_path[0]
        function_name = split_path[1]
    else:
        function_name = None

    if module_name is None:
        module_name = module_path.replace('/', '_').replace('.', '_')

    assert os.path.isfile(module_path)

    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    any_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(any_module)
    if function_name is None:
        return any_module
    else:
        return getattr(any_module, function_name)


def smart_round(x, base=None):
    if base is None:
        if x > 32 * 8:
            round_base = 32
        elif x > 16 * 8:
            round_base = 16
        else:
            round_base = 8
    else:
        round_base = base

    return max(round_base, round(x / float(round_base)) * round_base)

def merge_object_attr(obj1, obj2):
    new_dict = {}
    for k, v in obj1.__dict__.items():
        if v is None and k in obj2.__dict__:
            new_v = obj2.__dict__[k]
        else:
            new_v = v
        new_dict[k] = new_v

    for k, v in obj2.__dict__.items():
        if k not in new_dict:
            new_dict[k] = v

    obj1.__dict__.update(new_dict)
    return obj1

def smart_float(str1):
    if str1 is None:
        return None
    the_base = 1
    if str1[-1] == 'k':
        the_base = 1000
        str1 = str1[0:-1]
    elif str1[-1] == 'm':
        the_base = 1000000
        str1 = str1[0:-1]
    elif str1[-1] == 'g':
        the_base = 1000000000
        str1 = str1[0:-1]
    pass
    the_x = float(str1) * the_base
    return the_x

def split_str_to_list(str_to_split):
    group_str = str_to_split.split(',')
    the_list = []
    for s in group_str:
        t = s.split('*')
        if len(t) == 1:
            the_list.append(s)
        else:
            the_list += [t[0]] * int(t[1])
    return the_list

def mkfilepath(filename):
    distutils.dir_util.mkpath(os.path.dirname(filename))


def mkdir(dirname):
    distutils.dir_util.mkpath(dirname)

def robust_save(filename, save_function):
    mkfilepath(filename)
    backup_filename = filename + '.robust_save_temp'
    save_function(backup_filename)
    if os.path.isfile(filename):
        os.remove(filename)
    os.rename(backup_filename, filename)

def save_pyobj(filename, pyobj):
    mkfilepath(filename)
    the_s = pprint.pformat(pyobj, indent=2, width=120, compact=True)
    with open(filename, 'w') as fid:
        fid.write(the_s)


def load_pyobj(filename):
    with open(filename, 'r') as fid:
        the_s = fid.readlines()

    if isinstance(the_s, list):
        the_s = ''.join(the_s)

    the_s = the_s.replace('inf', '1e20')
    pyobj = ast.literal_eval(the_s)
    return pyobj

def create_logging(log_filename=None, level=logging.INFO):
    if log_filename is not None:
        mkfilepath(log_filename)
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[
                logging.StreamHandler()
            ]
        )