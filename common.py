from __future__ import division
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from pytorch_msssim import msssim

def project_dir():
    return os.path.dirname(os.path.realpath(__file__))

def to_np(_x): return _x.data.cpu().numpy()

def I(_x): return _x

def normalize(_x, _val=255, shift=0):
    return (_x - shift)/ _val

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def nhwc_to_nchw(_x):
    if len(_x.shape) == 3 and (_x.shape[-1] == 1 or _x.shape[-1] == 3): #unsqueeze N dim
        _x = _x[None, ...]
    elif len(_x.shape) == 3: #unsqueezed C dim
        _x = _x[..., None]
    elif len(_x.shape) == 2:  #unsqueeze N and C dim
        _x = _x[None, :, :, None]
    return np.transpose(_x, (0, 3, 1, 2))

def remove_img_boarder(border, x):
    return x[0, 0, border:-border, border:-border]

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
        noise = stddev * torch.randn_like(ins) + mean
        return ins + noise
    return ins

def delete_pixels(ins, is_training, sample_prob=0.3):
    if is_training:
        _sample_prob = torch.Tensor(1)
        prob_mask = _sample_prob.uniform_(sample_prob) * torch.ones_like(ins)
        mask = torch.bernoulli(prob_mask)
        return ins * mask  + (1 - mask)
    return ins

def reconstruction_loss(distance='l1', use_cuda=True):

    if distance == 'l1':
        dist = nn.L1Loss()
    elif distance == 'l2':
        dist  = nn.MSELoss()
    elif distance == 'msssim':
        dist = lambda res, tar: 1 - msssim(tar, res.clamp(0, 1))
    else:
        raise ValueError(f"unidentified value {distance}")

    return dist

def get_criterion(losses_types, factors, use_cuda=True):
    """
    Build Loss
        total_loss = sum_i factor_i * loss_i(results, targets)
    Args:
        factors(list): scales for each loss.
        losses(list): loss to apply to each result, target element
    """
    losses = []
    for loss_type in losses_types:
        losses.append(reconstruction_loss(loss_type))

    def total_loss(results, targets):
        """Calculate total loss
            total_loss = sum_i losses_i(results_i, targets_i)
        Args:
            results(tensor): nn outputs.
            targets(tensor): targets of resluts.

        """
        loss_acc = 0
        for fac, loss  in zip(factors, losses):
            _loss = loss(results, targets)
            loss_acc += _loss * fac
        return loss_acc

    return total_loss

def psnr(im, recon, verbose=False):
    im = np.squeeze(im)
    recon = np.squeeze(recon)
    MSE = np.sum((im - recon)**2) / np.prod(im.shape)
    MAX = 1.0#np.max(im)
    PSNR = 10 * np.log10(MAX ** 2 / MSE)
    if verbose:
        print('PSNR %f'%PSNR)
    return PSNR

def clean(save_path, save_count=10):
    import glob

    l = glob.glob(save_path)

    if len(l) < save_count:
        return 
    l.sort(key=os.path.getmtime) 
    for f in l[:-save_count]:
        print('removing', f)
        os.remove(f)

def save_train(path, model, optimizer, scheduler=None, epoch=None):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    if epoch is not None:
        state['epoch'] = epoch
    torch.save(state, os.path.join(path, 'epoch_{}'.format(epoch)))
    return os.path.join(path, 'epoch_{}'.format(epoch))

def load_train(path, model, optimizer, scheduler=None):
    state = torch.load(path)

    pretrained = state['model']
    model.load_state_dict(pretrained, strict=False)
    if 'optimizer' in state:
        try:
            optimizer.load_state_dict(state['optimizer'])
        except Exception as e:
            print(f'did not restore optimizer due to error {e}')
    else:
        print('Optimizer not initialized since no data for it exists in supplied path')
    if scheduler is not None:
        if 'scheduler' in state:
            scheduler.load_state_dict(state['scheduler'])
        else:
            print('Scheduler not initialized since no data for it exists in supplied path')
    if 'epoch' in state:
        e = state['epoch']
    else:
        e = 0
    return e

def save_eval(path, model):
    torch.save(model.state_dict(), path)

def load_eval(path, model):

    state = torch.load(path, map_location='cpu')
    pretrained = state['model']

    # Load state dict with strict=False to allow for model architecture changes
    # This will warn about missing or unexpected keys
    model.load_state_dict(pretrained, strict=False)
    model.eval()

