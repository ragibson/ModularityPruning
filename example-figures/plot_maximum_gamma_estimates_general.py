from math import exp, log
from random import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import lambertw


def parametric_omega(gamma, other_omega):
    if gamma == 0:
        return 0
    if other_omega < gamma:
        return -gamma * lambertw(-other_omega * exp(-other_omega / gamma) / gamma, k=-1).real
    return -gamma * lambertw(-other_omega * exp(-other_omega / gamma) / gamma).real


def trial(m):
    m_in = m * random()

    kappa_1 = (2 * m) * random()
    kappa_2 = 2 * m - kappa_1
    # kappa_2 = (2 * m - kappa_1) * random()
    # kappa_3 = 2 * m - kappa_1 - kappa_2

    sum_kappa_sqr = kappa_1 ** 2 + kappa_2 ** 2  # + kappa_3 ** 2

    omega_in = (4 * m * m_in) / sum_kappa_sqr
    omega_out = (4 * m ** 2 - 4 * m * m_in) / (4 * m ** 2 - sum_kappa_sqr)

    # if omega_in < omega_out:
    #     return trial(m)

    # assert 4 * m ** 2 / 3 <= sum_kappa_sqr <= 4 * m * m_in
    # assert omega_in + (3 - 1) * omega_out <= 3 + 1e-5
    return omega_in, omega_out


plt.close()
NUM_POINTS = 500
xs = np.linspace(0, 3, NUM_POINTS)
xs_early = np.linspace(0, 1, NUM_POINTS)
xs_late = np.linspace(1, 3, NUM_POINTS)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(xs, [(2 - x) / (2 - 1) for x in xs], color="C0")
plt.fill_between(xs_early, [(2 - x) / (2 - 1) for x in xs_early], [2] * len(xs_early), color="C0",
                 label=r"Possible $\Omega$ values", alpha=0.5)
plt.fill_between(xs_late, [(2 - x) / (2 - 1) for x in xs_late], [0] * len(xs_late), color="C0", alpha=0.5)
plt.plot(xs, [parametric_omega(1.0, x) for x in xs], label=r"$\gamma=1.0$", linestyle="dashed", color="C2")
plt.xlim([0.0, 3.0])
plt.ylim([0.0, 2.0])
plt.xlabel("$\omega_{in}$", fontsize=14)
plt.ylabel("$\omega_{out}$", fontsize=14)
plt.title(r"Maximum Expected $\gamma$ Estimates, $K=2$", fontsize=14)
plt.legend()
plt.savefig("general_max_gamma2.pdf")

plt.close()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(xs, [(3 - x) / (3 - 1) for x in xs], color="C1")
plt.fill_between(xs_early, [(3 - x) / (3 - 1) for x in xs_early], [2] * len(xs_early), color="C1",
                 label=r"Possible $\Omega$ values", alpha=0.5)
plt.fill_between(xs_late, [(3 - x) / (3 - 1) for x in xs_late], [0] * len(xs_late), color="C1", alpha=0.5)
plt.plot(xs, [parametric_omega(1.0926, x) for x in xs], label=r"$\gamma=1.0926$", linestyle="dashed", color="C3")
plt.xlim([0.0, 3.0])
plt.ylim([0.0, 2.0])
plt.xlabel("$\omega_{in}$", fontsize=14)
plt.ylabel("$\omega_{out}$", fontsize=14)
plt.title(r"Maximum Expected $\gamma$ Estimates, $K=3$", fontsize=14)
plt.legend()
plt.savefig("general_max_gamma3.pdf")
