# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2023/3/23 20:27
# @Author : liumin
# @File : 123456.py
import math
import torch

'''
target_bin = 128
num_distribution_bins = 2048
distribution = torch.load('histogram.pth')
'''

target_bin = 4
num_distribution_bins = 2048
distribution = torch.tensor([1, 3, 5, 7 ,9, 1])

min_kl_divergence = 66666
target_bin = 128
num_distribution_bins = 2048
distribution = torch.load('histogram.pth')

threshold_sum = torch.sum(distribution[target_bin:])

for threshold in range(target_bin, num_distribution_bins):
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

    print('distribution', distribution)
    print('clip_distribution', clip_distribution)
    print('quantize_distribution', quantize_distribution)
    quantize_distribution1 = [z / 1.25 for z in quantize_distribution]
    print('quantize_distribution1', quantize_distribution1)
    print('expand_distribution', expand_distribution)
    print('-'*30)

    assert threshold - target_bin < 2