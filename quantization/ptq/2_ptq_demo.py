# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2023/3/15 20:09
# @Author : liumin
# @File : 2_ptq_demo.py

import copy
import os

import time
import torch
import torch.nn as nn

from SlimPytorch.quantization.ptq.quant_util import get_input_sequences, register_fuse_params_to_prev_layers, \
    replace_quant_ops, fuse_bn, fuse_model, set_quant_mode
from SlimPytorch.quantization.ptq.utils import prepare_data, prepare_model, train_model, eval_model


def quant_proc(model, eval_loader, device, save_dir="model_quant.pth"):
    model.to(device)
    model.eval()
    model_to_quantize = copy.deepcopy(model)

    # w_scheme, w_bit, b_bit, a_scheme, a_bit = 'minmax', 8, 8, 'kl_divergence', 8
    w_scheme, w_bit, b_bit, a_scheme, a_bit = 'minmax', 8, 8, 'kl_divergence', 8
    fuse_model(model_to_quantize, w_scheme, w_bit, b_bit, a_scheme, a_bit)

    model_to_quantize.apply(set_quant_mode(quantized=True))

    '''
    quantized_model = convert_fx(prepared_model)
    # print("quantized model: ", quantized_model)
    if save_dir is not None:
        torch.save(quantized_model.state_dict(), save_dir)
    '''
    return model_to_quantize


if __name__ == "__main__":
    data_dir = '/home/lmin/data/hymenoptera'
    train_loader, eval_loader = prepare_data(data_dir=data_dir)
    model = prepare_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weight_path = "model_weight.pth"
    if os.path.exists(weight_path):
        # load
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    else:
        train_model(model, train_loader, eval_loader, device)
        # save
        torch.save(model.state_dict(), weight_path)

    device = torch.device("cpu")
    since = time.time()
    acc = eval_model(model, eval_loader, device)
    time_elapsed = time.time() - since
    print('float model Acc: {:.4f}, eval complete in  {:.0f}s'.format(acc, time_elapsed))

    quantized_model = quant_proc(model, eval_loader, device)

    since = time.time()
    acc = eval_model(quantized_model, eval_loader, device)
    time_elapsed = time.time() - since
    print('quant model Acc: {:.4f}, eval complete in {:.0f}s'.format(acc, time_elapsed))
