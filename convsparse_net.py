from __future__ import print_function
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.init
from torch.nn import Parameter
import numpy as np
from .common import conv as dp_conv
from .common import flip



class SparseConvAE(nn.Module):
    
    def __init__(self, num_input_channels=3, num_output_channels=3,
                 kc = 64, ks=3, ista_iters=3, iter_wieght_share=True,
                 pad='reflection', norm_wieghts=True, last=False):

        super(SparseConvAE, self).__init__()

        self.lista_encode = LISTAConvDictADMM(
            num_input_channels=num_input_channels, num_output_channels=num_output_channels,
            kc =kc, ks=ks, ista_iters=ista_iters, iter_wieght_share=iter_wieght_share,
            pad=pad, norm_wieghts=norm_wieghts
        )
        self.lista_decode = dp_conv(
            kc,
            num_input_channels,
            ks,
            stride=1,
            bias=False,
            pad=pad
        )
        self.last = last

    def forward(self, inputs):
#        print('thrsh mean %f'%np.mean(self.lista_encode.softthrsh._lambd.clamp(0,1).cpu().data.numpy()))
        non_zero_cnt = np.count_nonzero(inputs.cpu().data.numpy())
#        print('non zero count: {}'.format(non_zero_cnt))
        sparse_code = self.lista_encode(inputs)
#        non_zero_cnt = np.count_nonzero(sparse_code.cpu().data.numpy())
        result = self.lista_decode(sparse_code)
#        non_zero_cnt = np.count_nonzero(result.cpu().data.numpy())
        if self.last:
            return result
        else:
            return result + inputs
    
#TODO: 
#      1. Add mask as input.
#      2. Add unit norm to filters?

class LISTAConvDictADMM(nn.Module):
    """
    LISTA ConvDict encoder based on paper:
    https://arxiv.org/pdf/1711.00328.pdf
    """
    def __init__(self, num_input_channels=3, num_output_channels=3,
                 kc=64, ks=7, ista_iters=3, iter_wieght_share=True,
                 pad='reflection', norm_wieghts=True):
        super(LISTAConvDict, self).__init__()


        if iter_wieght_share == False:
            raise NotImplementedError('untied weights is not implemented yet...')
        self._ista_iters = ista_iters
        self.softthrsh = SoftshrinkTrainable(Parameter(0.1 * torch.ones(1, kc), requires_grad=True))

        self.encode_conv = dp_conv(
            num_input_channels,
            kc,
            ks,
            stride=1,
            bias=False,
            pad=pad
        )

        self.decode_conv0 = dp_conv(
            kc,
            num_input_channels,
            ks,
            stride=1,
            bias=False,
            pad=pad
        )


        self.decode_conv1 = dp_conv(
            kc,
            num_input_channels,
            ks,
            stride=1,
            bias=False,
            pad=pad
        )

        self.mu = Variable(0.6, requires_grad=True)

       # self._init_vars()

    def _init_vars(self): 
        ###################################
        # Better  Results without this inilization.
        ##################################
        wd = self.decode_conv[1].weight.data
        wd = F.normalize(F.normalize(wd, p=2, dim=2), p=2, dim=3)
        self.decode_conv[1].weight.data = wd
        self.encode_conv[1].weight.data = we

    def forward_enc(self, inputs):
        #print('thersh max: {}\n'.format(np.max(self.softthrsh._lambd.cpu().data.numpy())))
        sc = self.softthrsh(self.encode_conv(inputs))

        for step in range(self._ista_iters):
             _inputs = self.mu * inputs + (1 - self.mu) * self.decode_conv0(sc)

            sc_residual = self.encode_conv(
                _inputs - self.decode_conv1(sc)
                )
            sc = self.softthrsh(sc + sc_residual)
        return sc

    def forward_dec(self, sc):
        return self.decode_conv0(sc)

    def forward(self, inputs):
        sc = self.forward_enc(inputs)
        outputs = self.forward_dec(sc)
        return outputs

class SoftshrinkTrainable(nn.Module):
    """
    Learn threshold (lambda)
    """
    grads = {'thrsh': 0}
    def save_grad(self, name):
        def hook(grad):
            grads[name] = grad
        return hook

    def __init__(self, _lambd):
        super(SoftshrinkTrainable, self).__init__()
        self._lambd = _lambd
#        self._lambd.register_hook(print)

    def forward(self, inputs):
        _lambd = self._lambd.clamp(0)
        pos = inputs - _lambd.unsqueeze(2).unsqueeze(3).expand_as(inputs)
        neg = (-1) * inputs - _lambd.unsqueeze(2).unsqueeze(3).expand_as(inputs)
        return pos.clamp(min=0) - neg.clamp(min=0)
