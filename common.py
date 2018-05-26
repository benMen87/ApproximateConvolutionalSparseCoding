from __future__ import division
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


def I(_x): return _x

def normilize(_x, _val=255): return _x / _val

def nhwc_to_nchw(_x): 
    if len(_x.shape) == 3 and (_x.shape[-1] == 1 or _x.shape[-1] == 3): #unsqueeze N dim
        _x = _x[None, ...]
    elif len(_x.shape) == 3: #unsqueezed C dim
        _x = _x[...,None]
    elif len(_x.shape) == 2:  #unsqueeze N and C dim
        _x = _x[None,:,:,None]
    return np.transpose(_x, (0, 3, 1, 2))

def logdictargs(fullpath, args):
    import json
    with open(fullpath, 'w') as fp:
        fp.write(json.dumps(args))

def get_unique_name(path):
    idx = 1
    _path = path
    while os.path.isdir(_path):
        _path = '{}_{}'.format(path, idx)
        idx += 1
    return _path

def init_model_dir(path, name):
    full_path = os.path.join(path, name)
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        full_path = get_unique_name(full_path)
    os.mkdir(full_path)
    return full_path

    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
             else x.new(torch.arange(x.size(i)-1, -1, -1).tolist()).long()
             for i in range(x.dim()))
    return x[inds]

def bn(num_features):
    return nn.BatchNorm2d(num_features)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None

    if stride != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        else:
            assert False
        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection' and False:
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)


    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)

def gaussian(ins, is_training, mean, stddev):
    if is_training:
   #     noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        noise = stddev * torch.randn_like(ins) + mean
        return ins + noise
    return ins


def reconsturction_loss(factor=1.0, use_cuda=True):
    from pytorch_msssim import MSSSIM

    msssim = MSSSIM()
    l1 = nn.L1Loss()
    if use_cuda:
        msssim = msssim.cuda()
        l1 = l1.cuda()
    return l1#lambda x, xn: factor * l1(x, xn)  + (1 - factor) * (1 - msssim(x, xn))

def psnr(im, recon, verbose=True):
    im.shape
    im = np.squeeze(im)
    recon = np.squeeze(recon)
    MSE = np.sum((im - recon)**2) / (im.shape[0] * im.shape[1])
    MAX = np.max(im)
    PSNR = 10 * np.log10(MAX ** 2 / MSE)
    if verbose:
        print('PSNR %f'%PSNR)
    return PSNR
