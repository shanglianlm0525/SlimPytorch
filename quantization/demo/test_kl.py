# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2023/3/4 18:02
# @Author : liumin
# @File : test_kl.py

import numpy as np
import scipy.stats

# 随机生成两个离散型分布
x = [np.random.randint(1, 11) for i in range(10)]
print(x)
print(np.sum(x))
px = x / np.sum(x)
print(px)
y = [np.random.randint(1, 11) for i in range(10)]
print(y)
print(np.sum(y))
py = y / np.sum(y)
print(py)

# 利用scipy API进行计算
# scipy计算函数可以处理非归一化情况，因此这里使用
# scipy.stats.entropy(x, y)或scipy.stats.entropy(px, py)均可
KL = scipy.stats.entropy(x, y)
print(KL)

# 编程实现
KL = 0.0
for i in range(10):
    KL += px[i] * np.log(px[i] / py[i])
    # print(str(px[i]) + ' ' + str(py[i]) + ' ' + str(px[i] * np.log(px[i] / py[i])))

print(KL)
