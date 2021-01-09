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

from common import flip

class LISTAConvDict(nn.Module):
    """
    LISTA ConvDict encoder based on paper:
    https://arxiv.org/pdf/1711.00328.pdf
    """
    def __init__(self, num_input_channels=3, num_output_channels=3,
                 kc=64, ks=7, ista_iters=3, iter_weight_share=True,
                 share_decoder=False):

        super(LISTAConvDict, self).__init__()
        self._ista_iters = ista_iters
        self._layers = 1 if iter_weight_share else ista_iters

        def build_softthrsh():
            return SoftshrinkTrainable(
                Parameter(0.1 * torch.ones(1, kc), requires_grad=True)
            )

        self.softthrsh0 = build_softthrsh()
        if iter_weight_share:
            self.softthrsh1 = nn.ModuleList([self.softthrsh0
                                             for _ in range(self._layers)])
        else:
            self.softthrsh1 = nn.ModuleList([build_softthrsh()
                                             for _ in range(self._layers)])

        def build_conv_layers(in_ch, out_ch, count, stride=1):
            """Conv layer wrapper
            """
            return nn.ModuleList(
                [nn.Conv2d(in_ch, out_ch, ks,
                           stride=stride, padding=ks//2, bias=False) for _ in
                 range(count)])

        def build_deconv_layers(in_ch, out_ch, count, stride=1):
            """decConv layer wrapper
            """
            if stdride = 1:
                return build_conv_layers(in_ch, out_ch, count, stride=1)

            return nn.ModuleList(
                [nn.ConvTranspose2d(in_ch, out_ch, ks,
                           stride=stride, padding=ks//2, bias=False) for _ in
                 range(count)])


        self.encode_conv0 = build_conv_layers(num_input_channels, kc, 1)[0]
        if iter_weight_share:
            self.encode_conv1 = nn.ModuleList(self.encode_conv0 for _ in
                                              range(self._layers))
        else:
            self.encode_conv1 = build_conv_layers(num_input_channels, kc,
                                                  self._layers)

        self.decode_conv0 = build_conv_layers(kc, num_input_channels,
                                              self._layers if not share_decoder
                                              else 1)
        if share_decoder:
            self.decode_conv1 = self.decode_conv0[0]
            self.decode_conv0 = nn.ModuleList([self.decode_conv0[0] for _ in
                                               range(self._layers)])
        else:
            self.decode_conv1 = build_conv_layers(kc, num_output_channels, 1)[0]

    @property
    def ista_iters(self):
        """Amount of ista iterations
        """
        return self._ista_iters

    @property
    def layers(self):
        """Amount of layers with free parameters.
        """
        return self._layers

    @property
    def conv_dictionary(self):
        """Get the weights of convolutoinal dictionary
        """
        return self.decode_conv1.weight.data

    def forward_enc(self, inputs):
        """Conv LISTA forwrd pass
        """
        csc = self.softthrsh0(self.encode_conv0(inputs))

        for _itr, lyr in\
            zip(range(self._ista_iters),
                    cycle(range(self._layers))):

            sc_residual = self.encode_conv1[lyr](
                inputs - self.decode_conv0[lyr](csc)
            )
            csc = self.softthrsh1[lyr](csc + sc_residual)
        return csc

    def forward_enc_generataor(self, inputs):
        """forwar encoder generator
        Use for debug and anylize model.
        """
        csc = self.softthrsh0(self.encode_conv0(inputs))

        for itr, lyr in\
            zip(range(self._ista_iters),
                    cycle(range(self._layers))):

            sc_residual = self.encode_conv1[lyr](
                inputs - self.decode_conv0[lyr](csc)
            )
            csc = self.softthrsh1[lyr](csc + sc_residual)
            yield csc, sc_residual, itr

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

    @property
    def thrshold(self):
        return self._lambd
#        self._lambd.register_hook(print)

    def forward(self, inputs):
        """ sign(inputs) * (abs(inputs)  - thrshold)"""
        _inputs = inputs
        _lambd = self._lambd.clamp(0).unsqueeze(-1).unsqueeze(-1)
        result = torch.sign(_inputs) * (F.relu(torch.abs(_inputs) - _lambd))
        return result

    def _forward(self, inputs):
        """ sign(inputs) * (abs(inputs)  - thrshold)"""
        _lambd = self._lambd.clamp(0)
        pos = (inputs - _lambd.unsqueeze(-1).unsqueeze(-1))
        neg = ((-1) * inputs - _lambd.unsqueeze(-1).unsqueeze(-1))
        return (pos.clamp(min=0) - neg.clamp(min=0))
