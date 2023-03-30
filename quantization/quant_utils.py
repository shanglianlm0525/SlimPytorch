# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/5/26 18:38
# @Author : liumin
# @File : quant_utils.py

import torch
import torch.nn as nn
from SlimPytorch.quantization.quant_modules import QConv2d, QIdentity, QLinear, Quantizers


def get_input_sequences(model, dummy_shape=[1, 3, 224, 224]):
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
    dummy = torch.randn(dummy_shape).cuda()
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


def replace_quant_ops(model, w_bit, w_scheme, b_bit, a_bit, a_scheme):
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
            replace_quant_ops(child, w_bit, w_scheme, b_bit, a_bit, a_scheme)


def fuse_model(module):
    if isinstance(module, (QConv2d)):
        module.fuse_model()


def run_calibration(calibration):
    def estimate_range(module):
        if isinstance(module, Quantizers):
            module.estimate_range(flag = calibration)
    return estimate_range


def set_quant_mode(quantized):
    def set_precision_mode(module):
        if isinstance(module, Quantizers):
            module.set_quantize(quantized)
            module.estimate_range(False)
    return set_precision_mode

