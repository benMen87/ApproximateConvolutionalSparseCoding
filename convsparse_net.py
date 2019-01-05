from __future__ import print_function
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
from itertools import cycle
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.init
from torch.nn import Parameter
import numpy as np

from common import conv as dp_conv
from common import flip, I


class LISTAConvDictADMM(nn.Module):
    """
    LISTA ConvDict encoder based on paper:
    https://arxiv.org/pdf/1711.00328.pdf
    """
    def __init__(self, num_input_channels=3, num_output_channels=3,
                 kc=64, ks=7, ista_iters=3, iter_weight_share=True,
                 pad='reflection', norm_weights=True, use_sigmoid=False):

        super(LISTAConvDictADMM, self).__init__()

        self._ista_iters = ista_iters
        self._layers = 1 if iter_weight_share else ista_iters

        self.softthrsh = nn.ModuleList([
            SoftshrinkTrainable(
                Parameter(0.1 * torch.ones(1, kc), requires_grad=True)
            ) for _ in range(self._layers + 1)])

        def build_conv_layers(in_ch, out_ch, count):
            """Conv layer wrapper
            """
            return nn.ModuleList(
                [dp_conv(in_f=in_ch, out_f=out_ch, kernel_size=ks,
                         stride=1, bias=False, pad=pad) for _ in
                 range(count)])

        self.encode_conv = build_conv_layers(num_input_channels, kc,
                                             self._layers + 1)
        self.decode_conv0 = build_conv_layers(kc, num_input_channels,
                                              self._layers)
        self.decode_conv1 = build_conv_layers(kc, num_input_channels,
                                              1)[0]


    def forward_enc(self, inputs):
        """Conv LISTA forwrd pass
        """
        #print('thersh max: {}\n'.format(np.max(self.softthrsh._lambd.cpu().data.numpy())))
        csc = self.softthrsh[0](self.encode_conv[0](inputs))

        for _itr, lyr in\
            zip(range(self._ista_iters),
                    cycle(range(self._layers))):

            _inputs = inputs

            sc_residual = self.encode_conv[lyr + 1](
                _inputs - self.decode_conv0[lyr](csc)
                )
            csc = self.softthrsh[lyr + 1](csc + sc_residual)
        return csc

    def forward_dec(self, csc):
        """
        Decoder foward  csc --> input
        """
        return self.decode_conv1(csc)

    #pylint: disable=arguments-differ
    def forward(self, inputs):
        csc = self.forward_enc(inputs)
        outputs = self.forward_dec(csc)
        return outputs, csc

class SoftshrinkTrainable(nn.Module):
    """
    Learn threshold (lambda)
    """

    def __init__(self, _lambd):
        super(SoftshrinkTrainable, self).__init__()
        self._lambd = _lambd
#        self._lambd.register_hook(print)

    def forward(self, inputs):
        _lambd = self._lambd.clamp(0)
        pos = inputs - _lambd.unsqueeze(2).unsqueeze(3).expand_as(inputs)
        neg = (-1) * inputs - _lambd.unsqueeze(2).unsqueeze(3).expand_as(inputs)
        return pos.clamp(min=0) - neg.clamp(min=0)
