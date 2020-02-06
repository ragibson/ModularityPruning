from random import random, choice
import matplotlib.pyplot as plt
import numpy as np


def trial(m):
    m_in = m * random()

    kappa_1 = (2 * m) * random()
    # kappa_2 = 2 * m - kappa_1
    kappa_2 = (2 * m - kappa_1) * random()
    kappa_3 = 2 * m - kappa_1 - kappa_2

    sum_kappa_sqr = kappa_1 ** 2 + kappa_2 ** 2 + kappa_3 ** 2

    omega_in = (4 * m * m_in) / sum_kappa_sqr
    omega_out = (4 * m ** 2 - 4 * m * m_in) / (4 * m ** 2 - sum_kappa_sqr)

    if omega_in < omega_out:
        return trial(m)

    # assert 4 * m ** 2 / 3 <= sum_kappa_sqr <= 4 * m * m_in
    assert omega_in + (3 - 1) * omega_out <= 3 + 1e-5
    return omega_in, omega_out


xs, ys = list(zip(*[trial(100) for _ in range(10 ** 5)]))
plt.scatter(xs, ys, s=1)
plt.xlim([1, 3])
plt.ylim([0, 1])
plt.show()
