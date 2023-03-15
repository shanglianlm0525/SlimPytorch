# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2023/3/14 20:16
# @Author : liumin
# @File : 1_torch_ptq.py
import copy
import os
import torch
import torch.nn as nn
from torch.ao.quantization import QConfig
from torch.ao.quantization.fx.graph_module import ObservedGraphModule
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.observer import PerChannelMinMaxObserver, HistogramObserver

from SlimPytorch.quantization.ptq.utils import prepare_data, prepare_model, train_model, eval_model


def quant_fx(model, eval_loader, device, save_dir="model_quant.pth"):
    model.to(device)
    model.eval()
    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {
        "": qconfig,
    }
    model_to_quantize = copy.deepcopy(model)
    prepared_model = prepare_fx(model_to_quantize, qconfig_dict)
    # print("prepared model: ", prepared_model)

    prepared_model.eval()
    with torch.inference_mode():
        for inputs, labels in eval_loader:
            inputs = inputs.to(device)
            prepared_model(inputs)

    quantized_model = convert_fx(prepared_model)
    # print("quantized model: ", quantized_model)
    if save_dir is not None:
        torch.save(quantized_model.state_dict(), save_dir)

    return quantized_model


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

    acc = eval_model(model, eval_loader, device)
    print('float model Acc: {:.4f}'.format(acc))
    quantized_model = quant_fx(model, eval_loader, device)

    acc = eval_model(quantized_model, eval_loader, torch.device("cpu"))
    print('quant model Acc: {:.4f}'.format(acc))