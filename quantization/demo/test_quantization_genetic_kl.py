# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2023/3/4 19:06
# @Author : liumin
# @File : test_quantization_genetic_kl.py

import numpy as np
from numpy import random as rd
from copy import deepcopy
import matplotlib.pyplot as plt


def generate_R_data(data_count):
    """
    generate original float32 data R
    :param data_count: sample count
    :return: float data R
    """
    R = rd.randn(data_count)
    return R


def trans_R_2_Q(R, Z, S):
    """
    transfor R to Q
    :param R: original float32 R
    :param Z: zero point Z
    :param S: scale factor S
    :return: int data Q
    """
    Q = np.round(R / S + Z)
    return Q


def softmax(data):
    """
    get probability distribution of data
    :param data: a group of data
    :return: probability distribution
    """
    result = np.exp(data) / np.sum(np.exp(data))
    return result


def trans_Q_2_R(Q, S, Z):
    """
    transfor Q to R
    :param Q: int data Q which is generated from R
    :param S: scale factor
    :param Z: zero point
    :return: R recovered from Q
    """
    R = S * (Q - Z)
    return R


def get_Z_S(Rmax, Rmin, Qmax, Qmin):
    """
    get Z and S
    :param Rmax: max value of original R
    :param Rmin: min value of original R
    :param Qmax: max value of Q, specified according to the bit count, for example 8 bit is 2 ** 8 - 1
    :param Qmin: min value of Q, 0
    :return: scale factor S, zero point Z
    """
    S = (Rmax - Rmin) / (Qmax - Qmin)
    Z = Qmax - Rmax / S
    return S, Z


def get_trans_KL(R_softmax, R_recov_softmax):
    """
    get transoformation of KL divergence, convert Min task to Max task
    :param R_softmax: original R
    :param R_recov_softmax: R recovered from Q
    :return: 0.001  / (KL_value + 0.00000001)
    """
    KL_div = np.sum(R_softmax * np.log(R_softmax / R_recov_softmax))
    trans_KL_div = 0.001  / (KL_div + 0.00000001)
    return trans_KL_div


def get_S_Z_range(data_count, Qmax, Qmin):
    """
    get the genetic algorithm search range of S and Z
    :param data_count: sample count of R
    :param Qmax: max value of Q
    :param Qmin: min value of Q
    :return: Smax, Smin, Zmax, Zmin
    """
    Ss = []
    Zs = []
    for i in range(10000):
        R = generate_R_data(data_count)
        Rmax = np.max(R)
        Rmin = np.min(R)
        S, Z = get_Z_S(Rmax, Rmin, Qmax, Qmin)
        Ss.append(S)
        Zs.append(Z)
    Smax = np.max(Ss)
    Smin = np.min(Ss)
    Zmax = np.max(Zs)
    Zmin = np.min(Zs)
    return Smax, Smin, Zmax, Zmin


def code(pop_size, chrom_lenth):
    chroms = rd.randint(0, 2, (pop_size, chrom_lenth * 2))
    return chroms


def decode(chroms, Smin, Zmin, S_unit_bin, Z_unit_bin):
    decode_Z = []
    decode_S = []
    Z_chroms = chroms[:, :chroms.shape[1] // 2].astype(str).tolist()
    S_chroms = chroms[:, chroms.shape[1] // 2:].astype(str).tolist()
    for z_chrom, s_chrom in zip(Z_chroms, S_chroms):
        z_binary = "".join(z_chrom)
        s_binary = "".join(s_chrom)
        z = Zmin + int(z_binary, 2) * Z_unit_bin
        s = Smin + int(s_binary, 2) * S_unit_bin
        decode_Z.append(z)
        decode_S.append(s)
    return decode_Z, decode_S


def get_fitness(decode_Z, decode_S, R_original):
    fitnesses = []
    for Z, S in zip(decode_Z, decode_S):
        Q = trans_R_2_Q(R_original, Z, S)
        R_recov_from_Q = trans_Q_2_R(Q, S, Z)
        R_softmax = softmax(R_original)
        R_recov_softmax = softmax(R_recov_from_Q)
        trans_KL_div = get_trans_KL(R_softmax, R_recov_softmax)
        fitnesses.append(trans_KL_div)
    return fitnesses


def copy(chroms, fitnesses, decode_Z, decode_S):
    max_arg = np.argmax(fitnesses)
    best_fitness = fitnesses[max_arg]
    best_Z = decode_Z[max_arg]
    best_S = decode_S[max_arg]
    new_chroms = [deepcopy(chroms[max_arg].reshape((1, -1)))]
    probs = np.cumsum(fitnesses) / np.sum(fitnesses)
    for i in range(chroms.shape[0] - 1):
        rand_num = rd.random()
        mark = rand_num > probs
        select_index = np.sum(mark) - 1
        new_chroms.append(deepcopy(chroms[select_index].reshape((1, -1))))
    new_chroms = np.concatenate(new_chroms, axis=0)
    return new_chroms, best_fitness, best_Z, best_S


def cross(chroms, cp):
    cross_indexes = []
    for i in range(1, chroms.shape[0]):
        rand_num = rd.random()
        if rand_num < cp:
            cross_indexes.append(i)
    if len(cross_indexes) % 2 != 0:
        cross_indexes.pop()
    for i in range(0, len(cross_indexes), 2):
        index_1 = cross_indexes[i]
        index_2 = cross_indexes[i + 1]
        chrom1 = chroms[index_1]
        chrom2 = chroms[index_2]
        cross_point = rd.randint(1, len(chrom1))
        part_chrom1 = chrom1[cross_point:].copy()
        chrom1[cross_point:] = chrom2[cross_point:].copy()
        chrom2[cross_point:] = part_chrom1
    return chroms


def mutate(chroms, mp):
    rand_num = rd.random(chroms.shape[0] * chroms.shape[1] - chroms.shape[1])
    rand_num = np.concatenate([np.array([1.0] * chroms.shape[1]), rand_num], axis=0)
    mark = rand_num < mp
    chroms_flat = chroms.reshape((-1,)).copy()
    chroms_flat[mark] = 1 - chroms_flat[mark]
    chroms = chroms_flat.reshape(chroms.shape)
    return chroms


def ga(ga_iter_times, pop_size, chrom_lenth, Smin, Zmin, S_unit_bin, Z_unit_bin, R_original, cp, mp):
    chroms = code(pop_size, chrom_lenth)
    kl_divs_of_every_gen = []
    for g in range(ga_iter_times):
        decode_Z, decode_S = decode(chroms, Smin, Zmin, S_unit_bin, Z_unit_bin)
        fitnesses = get_fitness(decode_Z, decode_S, R_original)
        chroms, best_fitness, best_Z, best_S = copy(chroms, fitnesses, decode_Z, decode_S)
        kl_div_of_current_gen = 0.001 / best_fitness - 0.00000001
        kl_divs_of_every_gen.append(kl_div_of_current_gen)
        print("generation:%d, best KL:%.10f, best Z:%.5f, best S:%.5f" % (g + 1, kl_div_of_current_gen, best_Z, best_S))
        chroms = cross(chroms, cp)
        chroms = mutate(chroms, mp)
    return best_Z, best_S, kl_divs_of_every_gen


def plot_kl(kl_divs_of_every_gen):
    ax = plt.subplot(1, 1, 1)
    ax.plot(range(1, 1 + len(kl_divs_of_every_gen)), kl_divs_of_every_gen)
    ax.set_xlabel("generations")
    ax.set_ylabel("KL divergence")
    plt.show()


def compare(best_Z, best_S, R_original, Qmax, Qmin):
    """
    Compare the performance of obtaining Z and S by genetic algorithm and using common methods
    :param best_Z: Z obtained from genetic algorithm
    :param best_S: S obtained from genetic algorithm
    :param R_original: original R data
    :param Qmax: max value of Q
    :param Qmin: min value of Q
    :return:
    """
    Rmax = np.max(R_original)
    Rmin = np.min(R_original)
    Q_ga = trans_R_2_Q(R_original, best_Z, best_S)
    S_ordinary, Z_ordinary = get_Z_S(Rmax, Rmin, Qmax, Qmin)
    Q_ordinary = trans_R_2_Q(R_original, Z_ordinary, S_ordinary)
    R_ga_recov = trans_Q_2_R(Q_ga, best_S, best_Z)
    R_ordinary_recov = trans_Q_2_R(Q_ordinary, S_ordinary, Z_ordinary)
    ordinary_kl = 0.001 / get_trans_KL(softmax(R_original), softmax(R_ordinary_recov)) - 0.00000001
    ga_kl = 0.001 / get_trans_KL(softmax(R_original), softmax(R_ga_recov)) - 0.00000001
    mse_ordinary = float(np.mean(np.power(R_original - R_ordinary_recov, 2)))
    mse_ga = float(np.mean(np.power(R_original - R_ga_recov, 2)))
    print("##########compare KL divergence#############")
    print("KL divergence of ordinary method:%.10f\nKL divergence of genetic algorith method:%.10f" % (ordinary_kl, ga_kl))
    print("##########compare MSE#############")
    print("MSE of ordinary method:%.10f\nMSE of genetic algorith method:%.10f" % (mse_ordinary, mse_ga))


if __name__ == "__main__":
    # quantization param
    data_count = 1000  # sample count of R data
    Qmax = 255  # 2 ** 8 - 1
    Qmin = 0
    # genetic algorith param
    ga_iter_times = 1000 # genetic algorithm iteration times
    chrom_lenth = 10  # lenth of chromosome of genetic algorithm
    cp = 0.7  # probability of cross operation of genetic algorith
    mp = 0.2  # probability of mutation operation of genetic algorith
    pop_size = 1000 # size of population of genetic algorith
    Smax, Smin, Zmax, Zmin = get_S_Z_range(data_count, Qmax, Qmin)
    S_unit_bin = (Smax - Smin) / (2 ** chrom_lenth - 1)
    Z_unit_bin = (Zmax - Zmin) / (2 ** chrom_lenth - 1)
    R_original = generate_R_data(data_count)
    best_Z, best_S, kl_divs_of_every_gen = ga(ga_iter_times, pop_size, chrom_lenth, Smin, Zmin, S_unit_bin, Z_unit_bin, R_original, cp, mp)
    print("best Z:", best_Z)
    print("best S:", best_S)
    compare(best_Z, best_S, R_original, Qmax, Qmin)
    plot_kl(kl_divs_of_every_gen)