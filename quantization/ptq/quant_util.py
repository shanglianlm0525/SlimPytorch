# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2023/3/9 9:30
# @Author : liumin
# @File : quant_util.py
import copy

import torch
import torch.nn as nn

from SlimPytorch.quantization.ptq.quant_module import QConv2d, QIdentity, QLinear, Quantizer
from SlimPytorch.quantization.ptq.utils import eval_model


class PTQ():
    def __init__(self, model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), calibration_data=None):
        self.device = device # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = copy.deepcopy(model)
        self.model.to(self.device)
        self.model.eval()

        # weight quantization parameters
        self.w_scheme = 'minmax' # org: 0.9216, 'mse': 0.9085 --> 0.9150 'minmax': 0.9020 --> 0.9085 'kl':
        self.w_bit = 8
        self.b_bit = 8
        self.weight_quantize = False
        self.weight_is_symmetric = False
        self.weight_per_channel = True

        # activation quantization parameters
        self.a_scheme = 'kl' # 'minmax'
        self.a_bit = 8
        self.act_quantize = False
        self.act_is_symmetric = True
        self.act_per_channel = False
        self.calibration_data = calibration_data


    def set_quantize(self, flag):
        self.weight_quantize = flag
        self.act_quantize = flag

    def fuse(self):
        layer_fuse_pairs = get_input_sequences(self.model, device=self.device)
        register_fuse_params_to_prev_layers(self.model, layer_fuse_pairs)
        # print(layer_fuse_pairs)
        replace_quant_ops(self.model, self.w_scheme, self.w_bit, self.b_bit, self.a_scheme, self.a_bit)

        self.model.apply(fuse_bn)
        return self.model

    def get_data_range(self, x_f, is_symmetric, per_channel, device=None):
        '''
         https://heartbeat.fritz.ai/quantization-arithmetic-421e66afd842

         There exist two modes
         1) Symmetric:
             Symmetric quantization uses absolute max value as its min/max meaning symmetric with respect to zero
         2) Asymmetric
             Asymmetric Quantization uses actual min/max, meaning it is asymmetric with respect to zero

         Scale factor uses full range [-2**n / 2, 2**n - 1]
         '''
        min_val = torch.Tensor([float('inf')])[0]
        max_val = torch.Tensor([float('-inf')])[0]

        if is_symmetric:
            if per_channel:
                x_f_tmp = x_f.view(x_f.shape[0], -1)
                x_min, x_max = -torch.max(torch.abs(x_f_tmp), 1)[0], torch.max(torch.abs(x_f_tmp), 1)[0]
            else:
                x_min, x_max = -torch.max(torch.abs(x_f)), torch.max(torch.abs(x_f))
        else:
            if per_channel:
                x_f_tmp = x_f.view(x_f.shape[0], -1)
                x_min, x_max = torch.min(x_f_tmp, 1)[0], torch.max(x_f_tmp, 1)[0]
            else:
                x_min, x_max = torch.min(x_f), torch.max(x_f)

        min_val = torch.min(x_min, min_val)
        max_val = torch.max(x_max, max_val)
        data_range = max_val - min_val
        return min_val, max_val, data_range

    def quantize_minmax(self, bit, data_range, min_val, is_symmetric):
        scale = data_range / float(2 ** bit - 1)

        if not is_symmetric:
            offset = torch.round(-min_val / scale)
            return scale, offset

        return scale, None

    def quantize_mse(self):
        pass


    def quantize_kl(self):
        pass


    def quantize_aciq(self):
        pass

    def quantize_eq(self):
        pass

    def weight_quantize(self, x_f):
        min_val, max_val, data_range = self.get_data_range(x_f, self.weight_is_symmetric, self.weight_per_channel)
        if self.w_scheme == 'minmax':
            scale, offset = self.quantize_minmax(self.w_bit, data_range, min_val, self.weight_is_symmetric)
        elif self.w_scheme == 'mse':
            pass
        elif self.w_scheme == 'aciq':
            pass
        elif self.w_scheme == 'eq':
            pass
        else:
            raise NotImplementedError(self.w_scheme + ' is not Implemented! ')

    def act_quantize(self, x_f):
        min_val, max_val, data_range = self.get_data_range(x_f, self.act_is_symmetric, self.act_per_channel)
        if self.a_scheme == 'minmax':
            scale, offset = self.quantize_minmax(self.a_bit, data_range, min_val, self.act_is_symmetric)
        elif self.a_scheme == 'mse':
            pass
        elif self.a_scheme == 'aciq':
            pass
        elif self.a_scheme == 'eq':
            pass
        elif self.a_scheme == 'kl':
            pass
        else:
            raise NotImplementedError(self.a_scheme + ' is not Implemented! ')


    def quantize(self):
        # quantize weight
        def quantize_weight(module):
            if isinstance(module, Quantizer):
                module.set_quantize(True)
            if isinstance(module, (QConv2d, QLinear)):
                module.quantize_weight()

        self.model.apply(quantize_weight)

        # quantize activation
        if self.a_scheme == 'kl':
            assert self.calibration_data is not None

            def quantize_act_prepare(module):
                if isinstance(module, (QConv2d, QLinear)):
                    module.quantize_act_prepare()
            def quantize_act_run(module):
                if isinstance(module, (QConv2d, QLinear)):
                    module.quantize_act_run()

            self.model.apply(quantize_act_prepare)

            eval_model(self.model, self.calibration_data, self.device)

            # self.model.apply(quantize_act_run)
            self.model = quantize_act_run1(self.model)
        else:
            raise NotImplementedError(self.a_scheme + ' is not Implemented! ')

        return self.model


def quantize_act_run1(model):
    for child_name, child in model.named_children():
        if isinstance(child, (QConv2d, QLinear)):
            child.quantize_act_run()
        else:
            model = quantize_act_run1(child)
    return model


def get_input_sequences(model, dummy_shape=[1, 3, 224, 224], device=None):
    layer_fuse_pairs = []

    def hook(name):
        def func(m, i, o):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if not layer_fuse_pairs:
                    layer_fuse_pairs.append((m, name))
                else:
                    if isinstance(layer_fuse_pairs[-1][0], (nn.Conv2d, nn.Linear)):
                        layer_fuse_pairs.pop()
                    else:
                        if isinstance(layer_fuse_pairs[-1][0], nn.BatchNorm2d):
                            layer_fuse_pairs.append((None, None))
                        layer_fuse_pairs.append((m, name))
            elif isinstance(m, (nn.BatchNorm2d)):
                if isinstance(layer_fuse_pairs[-1][0], (nn.Conv2d, nn.Linear)):
                    layer_fuse_pairs.append((m, name))
            elif isinstance(m, (nn.ReLU, nn.ReLU6)):
                if isinstance(layer_fuse_pairs[-1][0], nn.BatchNorm2d):
                    layer_fuse_pairs.append((nn.ReLU(inplace=True), name))
            else:
                raise ValueError("Unsupported optimizer type: {}".format(m))
        return func

    pre = None
    handlers = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') or isinstance(module, (nn.ReLU, nn.ReLU6)):
            handlers.append(module.register_forward_hook(hook(name)))
    dummy = torch.randn(dummy_shape).to(device)
    model(dummy)
    for handle in handlers:
        handle.remove()
    return layer_fuse_pairs


def register_fuse_params_to_prev_layers(model, layer_bn_pairs):
    idx = 0
    while idx + 2 < len(layer_bn_pairs):
        conv, bn, act = layer_bn_pairs[idx], layer_bn_pairs[idx + 1], layer_bn_pairs[idx + 2]
        conv, conv_name = conv
        # bn
        bn, bn_name = bn
        bn_state_dict = bn.state_dict()
        conv.register_buffer('eps', torch.tensor(bn.eps))
        conv.register_buffer('gamma', bn_state_dict['weight'].detach())
        conv.register_buffer('beta', bn_state_dict['bias'].detach())
        conv.register_buffer('mu', bn_state_dict['running_mean'].detach())
        conv.register_buffer('var', bn_state_dict['running_var'].detach())
        # act function
        act, act_name = act
        conv.act = act
        idx += 3


def replace_quant_ops(model, w_scheme, w_bit, b_bit, a_scheme, a_bit):
    prev_module = None
    for child_name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            new_op = QConv2d(child, w_scheme=w_scheme, w_bit = w_bit, b_bit=b_bit, a_scheme=a_scheme, a_bit=a_bit)
            setattr(model, child_name, new_op)
            prev_module = getattr(model, child_name)
        elif isinstance(child, nn.Linear):
            new_op = QLinear(child, w_scheme=w_scheme, w_bit = w_bit, b_bit=b_bit, a_scheme=a_scheme, a_bit=a_bit)
            setattr(model, child_name, new_op)
            prev_module = getattr(model, child_name)
        elif isinstance(child, (nn.ReLU, nn.ReLU6)):
            # prev_module.activation_function = child
            prev_module.act = nn.ReLU()
            setattr(model, child_name, QIdentity())
        elif isinstance(child, nn.BatchNorm2d):
            setattr(model, child_name, QIdentity())
        else:
            replace_quant_ops(child, w_scheme, w_bit, b_bit, a_scheme, a_bit)


def fuse_bn(module):
    if isinstance(module, (QConv2d)):
        module.fuse_bn()


def set_quant_mode(quantized):
    def set_precision_mode(module):
        if isinstance(module, Quantizer):
            module.set_quantize(quantized)
            module.estimate_range(True)
    return set_precision_mode
