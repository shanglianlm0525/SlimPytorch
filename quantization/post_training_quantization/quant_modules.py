# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2023/3/2 16:30
# @Author : liumin
# @File : quant_modules.py

import torch
import torch.nn as nn


class Quantizers(nn.Module):
    def __init__(self, n, quant_mode='mse', act_q=True,
                 is_quantize=False, is_calibration = False, is_symmetric = False, per_channel=False):
        super(Quantizers, self).__init__()
        self.quant_mode = quant_mode
        self.n = n

        self.act_q = act_q
        self.is_quantize = is_quantize
        self.is_calibration = is_calibration
        self.is_symmetric = is_symmetric
        self.per_channel = per_channel

        self.min = torch.Tensor([float('inf')])[0].cuda()
        self.max = torch.Tensor([float('-inf')])[0].cuda()
        self.scale = None
        self.min_mse = float('inf')

    def set_quantize(self, is_quantize):
        self.is_quantize = is_quantize

    def estimate_range(self, is_calibration):
        self.is_calibration = is_calibration

    def init_params(self, x_f):
        '''
        https://heartbeat.fritz.ai/quantization-arithmetic-421e66afd842

        There exist two modes
        1) Symmetric:
            Symmetric quantization uses absolute max value as its min/max meaning symmetric with respect to zero
        2) Asymmetric
            Asymmetric Quantization uses actual min/max, meaning it is asymmetric with respect to zero

        Scale factor uses full range [-2**n / 2, 2**n - 1]
        '''

        if self.is_symmetric:
            x_min, x_max = -torch.max(torch.abs(x_f)), torch.max(torch.abs(x_f))
        else:
            x_min, x_max = torch.min(x_f), torch.max(x_f)

        self.min = torch.min(x_min, self.min)
        self.max = torch.max(x_max, self.max)
        max_range = self.max - self.min

        if self.quant_mode == 'mse':
            if not self.init or self.act_q:
                self.init = True
                for i in range(80):
                    scale = (max_range - (0.01 * i)) / float(2 ** self.n - 1)
                    offset = torch.round(-self.min / scale)
                    x_fp_q = self.quant_dequant(x_f, scale, offset)
                    curr_mse = torch.pow(x_fp_q - x_f, 2).mean().cpu().numpy()
                    if self.min_mse > curr_mse:
                        self.min_mse = curr_mse
                        self.scale = scale
                        self.offset = offset

        elif self.quant_mode == 'minmax':
            self.scale = max_range / float(2 ** self.n - 1)
            if not self.is_symmetric:
                self.offset = torch.round(-x_min / self.scale)
        self.init = True

    def quant_dequant(self, x_f, scale, offset):
        '''
        Quantizing
        Formula is derived from below:
        https://medium.com/ai-innovation/quantization-on-pytorch-59dea10851e1
        '''
        x_int = torch.round(x_f / scale)
        if not self.is_symmetric:
            x_int += offset

        if self.is_symmetric:
            l_bound, u_bound = -2 ** (self.n - 1), 2 ** (self.n - 1) - 1
        else:
            l_bound, u_bound = 0, 2 ** (self.n) - 1
        x_q = torch.clamp(x_int, min=l_bound, max=u_bound)

        '''
        De-quantizing
        '''
        if not self.is_symmetric:
            x_q -= offset
        x_float_q = x_q * scale
        return x_float_q

    def forward(self, x_f):
        if (self.calibration and self.act_q) or not self.init:
            self.init_params(x_f)
        return self.quant_dequant(x_f, self.scale, self.offset) if self.is_quantize else x_f