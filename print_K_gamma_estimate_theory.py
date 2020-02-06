import matplotlib.pyplot as plt
from math import log
import numpy as np
import sys
from time import time
import pickle

K_MAX = 71


def f(K, x):
    if x == 0:
        return 0
    elif x == 1:
        return 1
    return K * (1 / x - 1) / (1 / x + K - 1) / log(1 / x)


def fp(K, p_in, p_out):
    val = f(K, p_out / p_in)
    if val == 0:
        print(K, p_in, p_out)
    return val


xs = np.linspace(0, 1.0, 2 ** 12)
gamma_means = [0] * K_MAX
gamma_maxs = [0] * K_MAX

for k in range(2, K_MAX):
    ys = [fp(k, p_in, p_out) for p_in in xs for p_out in xs if 1 > p_in > p_out > 0]
    current_mean, current_max = np.mean(ys), np.max(ys)
    print("K={}: mean={:.3f}, max={:.3f}".format(k, current_mean, current_max))
    gamma_means[k] = current_mean
    gamma_maxs[k] = current_max

pickle.dump((gamma_means, gamma_maxs), open("maximum_gammas.p", "wb"))
