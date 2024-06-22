# -*- coding:utf-8 -*-
import os
import sys
import shutil
import logging
import colorlog
from tqdm import tqdm
import time
import yaml
import random
import importlib
from PIL import Image
from warnings import simplefilter
import imageio
import math
import collections
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from einops import rearrange, repeat
import torch.distributed as dist
from torchvision import datasets, transforms, utils

logging.getLogger().setLevel(logging.WARNING)
simplefilter(action='ignore', category=FutureWarning)

def get_logger(filename=None):
    """
    examples:
        logger = get_logger('try_logging.txt')

        logger.debug("Do something.")
        logger.info("Start print log.")
        logger.warning("Something maybe fail.")
        try:
            raise ValueError()
        except ValueError:
            logger.error("Error", exc_info=True)

        tips:
        DO NOT logger.inf(some big tensors since color may not helpful.)
    """
    logger = logging.getLogger('utils')
    level = logging.DEBUG
    logger.setLevel(level=level)
    # Use propagate to avoid multiple loggings.
    logger.propagate = False
    # Remove %(levelname)s since we have colorlog to represent levelname.
    format_str = '[%(asctime)s <%(filename)s:%(lineno)d> %(funcName)s] %(message)s'

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(level)
    coloredFormatter = colorlog.ColoredFormatter(
        '%(log_color)s' + format_str,
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            # 'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'reg,bg_white',
        }
    )

    streamHandler.setFormatter(coloredFormatter)
    logger.addHandler(streamHandler)

    if filename:
        fileHandler = logging.FileHandler(filename)
        fileHandler.setLevel(level)
        formatter = logging.Formatter(format_str)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    # Fix multiple logging for torch.distributed
    try:
        class UniqueLogger:
            def __init__(self, logger):
                self.logger = logger
                self.local_rank = torch.distributed.get_rank()

            def info(self, msg, *args, **kwargs):
                if self.local_rank == 0:
                    return self.logger.info(msg, *args, **kwargs)

            def warning(self, msg, *args, **kwargs):
                if self.local_rank == 0:
                    return self.logger.warning(msg, *args, **kwargs)

        logger = UniqueLogger(logger)
    # AssertionError for gpu with no distributed
    # AttributeError for no gpu.
    except Exception:
        pass
    return logger


logger = get_logger()

def split_filename(filename):
    absname = os.path.abspath(filename)
    dirname, basename = os.path.split(absname)
    split_tmp = basename.rsplit('.', maxsplit=1)
    if len(split_tmp) == 2:
        rootname, extname = split_tmp
    elif len(split_tmp) == 1:
        rootname = split_tmp[0]
        extname = None
    else:
        raise ValueError("programming error!")
    return dirname, rootname, extname

def data2file(data, filename, type=None, override=False, printable=False, **kwargs):
    dirname, rootname, extname = split_filename(filename)
    print_did_not_save_flag = True
    if type:
        extname = type
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    if not os.path.exists(filename) or override:
        if extname in ['jpg', 'png', 'jpeg']:
            utils.save_image(data, filename, **kwargs)
        elif extname == 'gif':
            imageio.mimsave(filename, data, format='GIF', duration=kwargs.get('duration'), loop=0)
        elif extname == 'txt':
            if kwargs is None:
                kwargs = {}
            max_step = kwargs.get('max_step')
            if max_step is None:
                max_step = np.Infinity

            with open(filename, 'w', encoding='utf-8') as f:
                for i, e in enumerate(data):
                    if i < max_step:
                        f.write(str(e) + '\n')
                    else:
                        break
        else:
            raise ValueError('Do not support this type')
        if printable: logger.info('Saved data to %s' % os.path.abspath(filename))
    else:
        if print_did_not_save_flag: logger.info(
            'Did not save data to %s because file exists and override is False' % os.path.abspath(
                filename))


def file2data(filename, type=None, printable=True, **kwargs):
    dirname, rootname, extname = split_filename(filename)
    print_load_flag = True
    if type:
        extname = type
    
    if extname in ['pth', 'ckpt']:
        data = torch.load(filename, map_location=kwargs.get('map_location'))
    elif extname == 'txt':
        top = kwargs.get('top', None)
        with open(filename, encoding='utf-8') as f:
            if top:
                data = [f.readline() for _ in range(top)]
            else:
                data = [e for e in f.read().split('\n') if e]
    elif extname == 'yaml':
        with open(filename, 'r') as f:
            data = yaml.load(f)
    else:
        raise ValueError('type can only support h5, npy, json, txt')
    if printable:
        if print_load_flag:
            logger.info('Loaded data from %s' % os.path.abspath(filename))
    return data


def ensure_dirname(dirname, override=False):
    if os.path.exists(dirname) and override:
        logger.info('Removing dirname: %s' % os.path.abspath(dirname))
        try:
            shutil.rmtree(dirname)
        except OSError as e:
            raise ValueError('Failed to delete %s because %s' % (dirname, e))

    if not os.path.exists(dirname):
        logger.info('Making dirname: %s' % os.path.abspath(dirname))
        os.makedirs(dirname, exist_ok=True)


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def adaptively_load_state_dict(target, state_dict):
    target_dict = target.state_dict()

    try:
        common_dict = {k: v for k, v in state_dict.items() if k in target_dict and v.size() == target_dict[k].size()}
    except Exception as e:
        logger.warning('load error %s', e)
        common_dict = {k: v for k, v in state_dict.items() if k in target_dict}

    if 'param_groups' in common_dict and common_dict['param_groups'][0]['params'] != \
            target.state_dict()['param_groups'][0]['params']:
        logger.warning('Detected mismatch params, auto adapte state_dict to current')
        common_dict['param_groups'][0]['params'] = target.state_dict()['param_groups'][0]['params']
    target_dict.update(common_dict)
    target.load_state_dict(target_dict)

    missing_keys = [k for k in target_dict.keys() if k not in common_dict]
    unexpected_keys = [k for k in state_dict.keys() if k not in common_dict]

    if len(unexpected_keys) != 0:
        logger.warning(
            f"Some weights of state_dict were not used in target: {unexpected_keys}"
        )
    if len(missing_keys) != 0:
        logger.warning(
            f"Some weights of state_dict are missing used in target {missing_keys}"
        )
    if len(unexpected_keys) == 0 and len(missing_keys) == 0:
        logger.warning("Strictly Loaded state_dict.")

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def image2pil(filename):
    return Image.open(filename)


def image2arr(filename):
    pil = image2pil(filename)
    return pil2arr(pil)


# 格式转换
def pil2arr(pil):
    if isinstance(pil, list):
        arr = np.array(
            [np.array(e.convert('RGB').getdata(), dtype=np.uint8).reshape(e.size[1], e.size[0], 3) for e in pil])
    else:
        arr = np.array(pil)
    return arr


def arr2pil(arr):
    if arr.ndim == 3:
        return Image.fromarray(arr.astype('uint8'), 'RGB')
    elif arr.ndim == 4:
        return [Image.fromarray(e.astype('uint8'), 'RGB') for e in list(arr)]
    else:
        raise ValueError('arr must has ndim of 3 or 4, but got %s' % arr.ndim)

def notebook_show(*images):
    from IPython.display import Image
    from IPython.display import display
    display(*[Image(e) for e in images])