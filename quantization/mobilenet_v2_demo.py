# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/5/28 9:56
# @Author : liumin
# @File : mobilenet_v2_demo.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from custom_model import MobileNetV2
from quant_utils import get_input_sequences, replace_quant_ops
from MCF.quantization.quant_modules import QConv2d, QLinear, QIdentity
# from MCF.quantization.quant_utils import register_bn_params_to_prev_layers, replace_quant_ops
from MCF.quantization.quant_utils import fuse_model, run_calibration, set_quant_mode


def get_input_sequences(model, dummy_shape=[1, 3, 224, 224]):
    layer_fuse_pairs = []

    def hook(name):
        def func(m, i, o):
            if m in (torch.nn.Conv2d, torch.nn.Linear):
                if not layer_fuse_pairs:
                    layer_fuse_pairs.append((m, name))
                else:
                    if layer_fuse_pairs[-1][0] in (torch.nn.Conv2d, torch.nn.Linear):
                        layer_fuse_pairs.pop()
            else:
                layer_fuse_pairs.append((m, name))

        return func

    handlers = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            handlers.append(module.register_forward_hook(hook(name)))
    dummy = torch.randn(dummy_shape).cuda()
    model(dummy)
    for handle in handlers:
        handle.remove()
    return layer_fuse_pairs


def register_bn_params_to_prev_layers(model, layer_bn_pairs):
    idx = 0
    while idx + 1 < len(layer_bn_pairs):
        conv, bn = layer_bn_pairs[idx], layer_bn_pairs[idx + 1]
        conv, conv_name = conv
        bn, bn_name = bn
        bn_state_dict = bn.state_dict()
        conv.register_buffer('eps', torch.tensor(bn.eps))
        conv.register_buffer('gamma', bn_state_dict['weight'].detach())
        conv.register_buffer('beta', bn_state_dict['bias'].detach())
        conv.register_buffer('mu', bn_state_dict['running_mean'].detach())
        conv.register_buffer('var', bn_state_dict['running_var'].detach())
        idx += 2


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


class MobilenetV2_PTQ():
    def __init__(self):
        self.data_dir = '/home/lmin/data/hymenoptera/val'
        self.model_dir = 'ckpt/mobilenet_v2_train.pt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # weight quantization parameters
        self.w_scheme = 'minmax' # 'mse': 0.9085, 'minmax': 0.9020
        self.w_bit = 8
        self.b_bit = 8

        # activation quantization parameters
        self.a_scheme = 'minmax'
        self.a_bit = 8

    def load_data(self):
        data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_dataset = datasets.ImageFolder(self.data_dir,
                                                  data_transform)
        dataload = torch.utils.data.DataLoader(image_dataset, batch_size=1,
                                                      shuffle=False, num_workers=4)
        return dataload


    def load_model(self):
        model = MobileNetV2('mobilenet_v2', classifier=True)
        num_ftrs = model.fc[1].in_features
        model.fc[1] = nn.Linear(num_ftrs, 2)
        model.load_state_dict(torch.load(self.model_dir, map_location='cpu'))
        return model


    def model_accuracy(self, model, dataload):
        print('-' * 10)
        # Each epoch has a training and validation phase
        model.eval()  # Set model to evaluate mode

        running_corrects = 0
        # Iterate over data.
        for inputs, labels in dataload:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

        acc = running_corrects.double() / len(dataload.dataset)
        return acc


    def fuse(self, model):
        print('step 1: model fuse and optimize.')
        layer_fuse_pairs = get_input_sequences(model)
        register_bn_params_to_prev_layers(model, layer_fuse_pairs)
        # print(layer_fuse_pairs)

        replace_quant_ops(model, self.w_bit, self.w_scheme, self.b_bit, self.a_bit, self.a_scheme)

        model.apply(fuse_model)
        return model


    def weight_quantize(self):
        pass


    def activation_quantize(self):
        pass


    def run(self):
        dataload = self.load_data()
        model = self.load_model()
        model.to(self.device)
        model.eval()

        acc = self.model_accuracy(model, dataload)
        print('float model Acc: {:.4f}'.format(acc))

        model = self.fuse(model)
        acc = self.model_accuracy(model, dataload)
        print('fuse model Acc: {:.4f}'.format(acc))

        model.apply(run_calibration(calibration=True))
        # 3 replace_quant_to_brecq_quant(model)
        model.apply(set_quant_mode(quantized=True))

        acc = self.model_accuracy(model, dataload)
        print('quant model Acc: {:.4f}'.format(acc))


if __name__ == '__main__':
    mobilenetV2_PTQ = MobilenetV2_PTQ()
    mobilenetV2_PTQ.run()
    print('done!')