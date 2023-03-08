# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2023/3/2 16:30
# @Author : liumin
# @File : quant_module.py

import torch
import torch.nn as nn
import numpy as np


class Quantizer(nn.Module):
    def __init__(self, n, quant_mode='mse', is_quantize=False, act_q=True,
                 is_calibration = False, is_symmetric = False, per_channel=False,
                 calibration_data=None):
        super(Quantizer, self).__init__()
        self.is_quantize = is_quantize
        self.quant_mode = quant_mode
        self.n = n

        # weight and activation quantizer config
        # Weight Quantization : per-channel asymmetric quantization
        # Activation Quantization : per-layer asymmetric quantization
        self.is_symmetric = is_symmetric
        self.per_channel = per_channel
        # self.act_q = act_q
        self.is_calibration = is_calibration
        self.calibration_data = calibration_data

        # quantizer parameters
        self.min = torch.Tensor([float('inf')])[0].cuda()
        self.max = torch.Tensor([float('-inf')])[0].cuda()
        self.scale = None
        self.offset = None
        self.min_mse = float('inf')

    def set_quantize(self, flag):
        self.is_quantize = flag

    def estimate_range(self, flag):
        self.is_calibration = flag

    def set_symmetric(self, flag):
        self.is_symmetric = flag

    def set_per_channel(self, flag):
        self.per_channel = flag

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
            if self.per_channel:
                x_f_tmp = x_f.view(x_f.shape[0], -1)
                x_min, x_max = -torch.max(torch.abs(x_f_tmp), 1)[0], torch.max(torch.abs(x_f_tmp), 1)[0]
            else:
                x_min, x_max = -torch.max(torch.abs(x_f)), torch.max(torch.abs(x_f))
        else:
            if self.per_channel:
                x_f_tmp = x_f.view(x_f.shape[0], -1)
                x_min, x_max = torch.min(x_f_tmp, 1)[0], torch.max(x_f_tmp, 1)[0]
            else:
                x_min, x_max = torch.min(x_f), torch.max(x_f)

        if self.per_channel:
            self.min = torch.min(x_min, self.min)
            self.max = torch.max(x_max, self.max)
            max_range = self.max - self.min
        else:
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
            if self.per_channel:
                self.scale = max_range / float(2 ** self.n - 1)
                if not self.is_symmetric:
                    self.offset = torch.round(-x_min / self.scale)
            else:
                self.scale = max_range / float(2 ** self.n - 1)
                if not self.is_symmetric:
                    self.offset = torch.round(-x_min / self.scale)

        elif self.quant_mode == 'kl_divergence':
            pass
            # 1 count the absmax

            # 2 initialize histogram

            # 3 build histogram

            # 4 using kld to find the best threshold value
        else:
            raise NotImplementedError(self.quant_mode + ' is not Implemented! ')
        self.init = True

    def compute_kl_divergence(self, dist_a, dist_b):
        dist_a = np.array(dist_a)
        dist_b = np.array(dist_b)
        nonzero_inds = dist_a != 0
        return np.sum(dist_a[nonzero_inds] *
                      np.log(dist_a[nonzero_inds] / (dist_b[nonzero_inds] + 1e-12) + 1e-12))

    def quant_dequant(self, x_f, scale, offset):
        '''
        Quantizing
        Formula is derived from below:
        https://medium.com/ai-innovation/quantization-on-pytorch-59dea10851e1
        '''
        if self.per_channel:
            if x_f.ndim == 4:
                scale = scale.view(x_f.shape[0], 1, 1, 1)
                if not self.is_symmetric:
                    offset = offset.view(x_f.shape[0], 1, 1, 1)
            elif x_f.ndim == 2:
                scale = scale.view(x_f.shape[0], 1)
                if not self.is_symmetric:
                    offset = offset.view(x_f.shape[0], 1)
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
        else:
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