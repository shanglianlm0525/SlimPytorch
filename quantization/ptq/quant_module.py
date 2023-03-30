# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2023/3/19 18:11
# @Author : liumin
# @File : quant_module.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Quantizer(nn.Module):
    def __init__(self, scheme='mse', bit=8, b_bit=None,  is_quantize=False,
                 is_symmetric = False, per_channel=False, is_calibration=False):
        super(Quantizer, self).__init__()
        self.scheme = scheme
        self.bit = bit
        self.b_bit = b_bit

        # weight and activation quantizer config
        # Weight Quantization : per-channel asymmetric quantization
        # Activation Quantization : per-layer asymmetric quantization
        self.is_quantize = is_quantize
        self.is_symmetric = is_symmetric
        self.per_channel = per_channel
        self.is_calibration = is_calibration

        # quantizer parameters
        self.init = False
        self.min = torch.Tensor([float('inf')])[0]
        self.max = torch.Tensor([float('-inf')])[0]
        self.scale = None
        self.offset = None

        # mse
        self.min_mse = float('inf')
        # kl
        self.num_histogram_bins = 2048
        # self.histogram = torch.zeros(self.num_histogram_bins)


    def set_quantize(self, flag):
        self.is_quantize = flag

    def set_symmetric(self, flag):
        self.is_symmetric = flag

    def set_per_channel(self, flag):
        self.per_channel = flag

    def set_is_calibration(self, flag):
        self.is_calibration = flag

    def get_data_range(self, x_f, device=None):
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

        self.min = torch.min(x_min, self.min)
        self.max = torch.max(x_max, self.max)
        data_range = self.max - self.min
        return data_range

    def quant_params(self, x_f):
        data_range = self.get_data_range(x_f)

        if self.scheme == 'mse':
            if not self.init or self.act_q:
                self.init = True
                for i in range(80):
                    scale = (data_range - (0.01 * i)) / float(2 ** self.bit - 1)
                    offset = torch.round(-self.min / scale)
                    x_fp_q = self.quant_dequant(x_f, scale, offset)
                    curr_mse = torch.pow(x_fp_q - x_f, 2).mean().cpu().numpy()
                    if self.min_mse > curr_mse:
                        self.min_mse = curr_mse
                        self.scale = scale
                        self.offset = offset

        elif self.scheme == 'minmax':
            if self.per_channel:
                self.scale = data_range / float(2 ** self.bit - 1)
                if not self.is_symmetric:
                    self.offset = torch.round(-self.min / self.scale)
            else:
                self.scale = data_range / float(2 ** self.bit - 1)
                if not self.is_symmetric:
                    self.offset = torch.round(-self.min / self.scale)

        elif self.scheme == 'kl':
            x_f = x_f.to(torch.device("cpu"))
            self.max = self.max.to(torch.device("cpu"))
            # 1 count the absmax
            # 2 build histogram
            indexs = torch.abs(x_f) / self.max * self.num_histogram_bins
            indexs[indexs > (self.num_histogram_bins - 1)] = self.num_histogram_bins - 1
            distribution = torch.bincount(indexs.view(-1).int(), minlength=self.num_histogram_bins - 1)
            distribution = distribution.float() / (torch.sum(distribution) + 1e-12)

            # 3 using kld to find the best threshold value
            min_kl_divergence = 66666
            target_bin = 2 ** (self.bit-1)
            target_threshold = distribution.shape[0] - 1
            threshold_sum = torch.sum(distribution[target_bin:])
            for threshold in range(target_bin, self.num_histogram_bins):
                clip_distribution = distribution[:threshold].clone().detach()
                clip_distribution[-1] += threshold_sum
                # The calculation will greatly reduce the computational load.
                threshold_sum = threshold_sum - distribution[threshold]

                num_per_bin = float(threshold) / target_bin

                quantize_distribution = [0. for _ in range(target_bin)]
                expand_distribution = [1e-9 for _ in range(threshold)]

                for i in range(target_bin):
                    start = i * num_per_bin
                    end = start + num_per_bin
                    left_upper = int(math.ceil(start))
                    if left_upper > start:
                        left_scale = left_upper - start
                        quantize_distribution[i] += left_scale * distribution[left_upper - 1]
                    right_lower = int(math.floor(end))
                    if right_lower < end:
                        right_scale = end - right_lower
                        quantize_distribution[i] += right_scale * distribution[right_lower]

                    quantize_distribution[i] += torch.sum(distribution[left_upper:right_lower])

                for i in range(0, target_bin):
                    start = i * num_per_bin
                    end = start + num_per_bin
                    count = 1e-12

                    left_upper = int(math.ceil(start))
                    left_scale = 0.0
                    if left_upper > start:
                        left_scale = left_upper - start
                        if distribution[left_upper - 1] != 0:
                            count += left_scale

                    right_lower = int(math.floor(end))
                    right_scale = 0.0
                    if right_lower < end:
                        right_scale = end - right_lower
                        if distribution[right_lower] != 0:
                            count += right_scale

                    for j in range(left_upper, right_lower):
                        if distribution[j] != 0:
                            count = count + 1
                    expand_value = quantize_distribution[i] / count

                    if left_upper > start:
                        if distribution[left_upper - 1] != 0:
                            expand_distribution[left_upper - 1] += expand_value * left_scale
                    if right_lower < end:
                        if distribution[right_lower] != 0:
                            expand_distribution[right_lower] += expand_value * right_scale
                    for j in range(left_upper, right_lower):
                        if distribution[j] != 0:
                            expand_distribution[j] += expand_value

                kl_divergence = self.compute_kl_divergence(clip_distribution, expand_distribution)
                if kl_divergence < min_kl_divergence:
                    min_kl_divergence = kl_divergence
                    target_threshold = threshold

            self.scale = ((target_threshold + 0.5) * self.max) / ((target_bin - 1) * self.num_histogram_bins)
            print(target_threshold, self.max, self.scale, self.max / self.scale, torch.round(self.max / self.scale))
        else:
            raise NotImplementedError(self.quant_mode + ' is not Implemented! ')

        self.init = True

    def compute_kl_divergence(self, dist_a, dist_b):
        dist_b = torch.tensor(dist_b).to(dist_a.device)
        nonzero_inds = dist_a != 0
        return torch.sum(dist_a[nonzero_inds] * torch.log(dist_a[nonzero_inds] / (dist_b[nonzero_inds] + 1e-12) + 1e-12))

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
                l_bound, u_bound = -2 ** (self.bit - 1), 2 ** (self.bit - 1) - 1
            else:
                l_bound, u_bound = 0, 2 ** (self.bit) - 1
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
                l_bound, u_bound = -2 ** (self.bit - 1), 2 ** (self.bit - 1) - 1
            else:
                l_bound, u_bound = 0, 2 ** (self.bit) - 1
            x_q = torch.clamp(x_int, min=l_bound, max=u_bound)

            '''
            De-quantizing
            '''
            if not self.is_symmetric:
                x_q -= offset
            x_float_q = x_q * scale
        return x_float_q

    def forward(self, x_f):
        if self.is_quantize and not self.init:
            self.quant_params(x_f)
        return self.quant_dequant(x_f, self.scale, self.offset) if self.is_quantize else x_f


class QConv2d(nn.Module):
    '''
    Fuses only the following sequence of modules:
    conv, bn
    conv, bn, relu
    conv, relu
    bn, relu
    '''
    def __init__(self, module, w_scheme='mse', w_bit = 8, b_bit=8, a_scheme='mse', a_bit=8):
        super(QConv2d, self).__init__()
        # weights
        self.conv = module
        self.weight_quantizer = Quantizer(w_scheme, w_bit, b_bit,
                                          is_symmetric = False, per_channel=True)
        self.kwarg = {'stride': self.conv.stride, 'padding': self.conv.padding,
                      'dilation': self.conv.dilation, 'groups': self.conv.groups}

        self.act = self.conv.act
        # activations
        self.act_quantizer = Quantizer(a_scheme, a_bit, None,
                                       is_symmetric=True, per_channel=False)
        self.pre_act = False

        # parameter
        self.weight_quantized = False
        self.quantized_w = None
        self.quantized_b = None
        self.act_quantized = False
        self.activations = []


    def fuse_bn(self):
        '''
                https://towardsdatascience.com/speed-up-inference-with-batch-normalization-folding-8a45a83a89d8

                W_fold = gamma * W / sqrt(var + eps)
                b_fold = (gamma * ( bias - mu ) / sqrt(var + eps)) + beta
                '''
        if hasattr(self.conv, 'gamma'):
            gamma = getattr(self.conv, 'gamma')
            beta = getattr(self.conv, 'beta')
            mu = getattr(self.conv, 'mu')
            var = getattr(self.conv, 'var')
            eps = getattr(self.conv, 'eps')

            denom = gamma.div(torch.sqrt(var + eps))

            if getattr(self.conv, 'bias') == None:
                self.conv.bias = torch.nn.Parameter(var.new_zeros(var.shape))
            b_fold = denom * (self.conv.bias.data - mu) + beta
            self.conv.bias.data.copy_(b_fold)

            denom = denom.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            self.conv.weight.data.mul_(denom)


    def quantize_weight(self):
        # quantize weight
        self.quantized_w = self.conv.weight.detach()
        if self.conv.bias != None:
            self.quantized_b = self.conv.bias.detach()
        else:
            self.quantized_b = None
        self.quantized_w = self.weight_quantizer(self.quantized_w)
        self.weight_quantized = True

    def quantize_act(self):
        # quantize activation
        # self.act_quantizer(out)
        self.act_quantized = True

    def quantize_act_prepare(self):
        # quantize activation
        # self.act_quantizer(out)
        self.act_quantized = True

    def quantize_act_run(self):
        activations = torch.cat(self.activations)
        self.activations = []
        self.act_quantizer(activations)

    def quantize(self):
        self.quantize_weight()

    def forward(self, x):
        if self.act_quantized:
            self.activations.append(x)
        if self.weight_quantized:
            out = F.conv2d(input=x, weight=self.quantized_w, bias=self.quantized_b, **self.kwarg)
            if self.act and not self.pre_act:
                out = self.act(out)
            return out

        out = self.conv(x)
        if self.act and not self.pre_act:
            out = self.act(out)
        return out



    def set_pre_activation(self):
        self.pre_act = True


    def forward2(self, x):
        w, b = self.get_params()
        out = F.conv2d(input=x, weight=w, bias=b, **self.kwarg)
        if self.act and not self.pre_act:
            out = self.act(out)
        out = self.act_quantizer(out)
        return out


class QLinear(nn.Module):
    '''
    Fuses only the following sequence of modules:
    linear, relu
    bn, relu
    '''
    def __init__(self, module, w_scheme='mse', w_bit = 8, b_bit=8, a_scheme='mse', a_bit=8):
        super(QLinear, self).__init__()
        self.fc = module
        # self.fc, self.norm, self.act = module
        self.weight_quantizer = Quantizer(w_scheme, w_bit, None,
                                          is_symmetric = False, per_channel=True)
        self.act = self.fc.act if hasattr(self.fc, 'act') else None
        # activations
        self.act_quantizer = Quantizer(a_scheme, a_bit, None,
                                       is_symmetric = True, per_channel=False)

        # parameter
        self.weight_quantized = False
        self.quantized_w = None
        self.quantized_b = None
        self.act_quantized = False
        self.activations = []


    def quantize_weight(self):
        # quantize weight
        self.quantized_w = self.fc.weight.detach()
        if self.fc.bias != None:
            self.quantized_b = self.fc.bias.detach()
        else:
            self.quantized_b = None
        self.quantized_w = self.weight_quantizer(self.quantized_w)
        self.weight_quantized = True

    def quantize_act(self):
        # quantize activation
        self.act_quantized = True

    def quantize_act_prepare(self):
        # quantize activation
        self.act_quantized = True

    def quantize_act_run(self):
        activations = torch.cat(self.activations)
        self.activations = []
        self.act_quantizer(activations)

    def quantize(self):
        self.quantize_weight()


    def forward(self, x):
        if self.act_quantized:
            self.activations.append(x)
        if self.weight_quantized:
            out = F.linear(x, self.quantized_w, self.quantized_b)
            if self.act and not self.pre_act:
                out = self.act(out)
            return out

        out = self.fc(x)
        if self.act and not self.pre_act:
            out = self.act(out)
        return out



class QIdentity(nn.Module):
    def __init__(self):
        super(QIdentity, self).__init__()

    def forward(self, x):
        return x
