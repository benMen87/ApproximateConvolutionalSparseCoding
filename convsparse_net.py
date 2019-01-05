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

        if iter_weight_share:
            self.softthrsh = [
                SoftshrinkTrainable(
                    Parameter(0.1 * torch.ones(1,kc), requires_grad=True)
                )] * ista_iters
        else:
            self.softthrsh = [
                SoftshrinkTrainable(
                    Parameter(0.1 * torch.ones(1,kc), requires_grad=True)
                ) for _ in ista_iters]

        def build_conv_layer(in_ch, out_ch):
            if iter_weight_share:
                return [dp_conv(num_input_channels, in_ch, out_ch,
                                stride=1, bias=False, pad=pad)] * ista_iters
            else:
                return [dp_conv(num_input_channels, in_ch, out_ch,
                                stride=1, bias=False, pad=pad) for _ in ista_iters]

        self.encode_conv = build_conv_layer(num_input_channels, kc)
        self.decode_conv0 = build_conv_layer(kc, num_input_channels)
        self.decode_conv1 = build_conv_layer(kc, num_input_channels)

        self.output_act = (I if not use_sigmoid else
                           torch.nn.Sigmoid())

    def forward_enc(self, inputs):
        #print('thersh max: {}\n'.format(np.max(self.softthrsh._lambd.cpu().data.numpy())))
        csc = self.softthrsh[itr](self.encode_conv[itr](inputs))
        for itr in range(self._ista_iters):
            _inputs = inputs

            sc_residual = self.encode_conv[itr](
                _inputs - self.decode_conv1[itr](csc)
            )
            csc = self.softthrsh[itr](csc + sc_residual)
        return csc

    def forward_dec(self, csc):
        """
        Decoder foward  csc --> input
        """
        return self.decode_conv0(csc)

    #pylint: disable=arguments-differ
    def forward(self, inputs):
        csc = self.forward_enc(inputs)
        outputs = self.output_act(self.forward_dec(csc))
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
