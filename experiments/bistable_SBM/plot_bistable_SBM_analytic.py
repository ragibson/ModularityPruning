# Generates figure B.1

from math import log
import matplotlib.pyplot as plt
import numpy as np

PLOT_XLIM = [-0.003, 0.063]


def gamma_cross(b, d):
    return (7040 * b * d * (b * (7920 * d + 1099) - 1000)) / ((b * (5280 * d + 433) - 400) ** 2)


def gamma_cross_limit(d):
    return (7040 * d * (1099 + 7920 * d)) / (433 + 5280 * d) ** 2


def gamma_2(d):
    numerator = 2 * (80 * d + 11) ** 2 * (256 * d + 19)
    denominator_scalar = 9 * (160 * d + 13) * (2560 * d ** 2 + 416 * d + 25)
    denominator_scalar2 = (log((4 * (8 * d + 1) * (80 * d + 11)) / (2560 * d ** 2 + 416 * d + 25)) -
                           log((160 * d + 22) / (1440 * d + 117)))
    return numerator / (denominator_scalar * denominator_scalar2)


def gamma_3(d):
    numerator = 8 * (80 * d + 11) ** 2 * (320 * d * (80 * d - 7) - 549)
    denominator_scalar = (160 * d + 13) * (160 * d + 49) * (320 * d * (80 * d + 13) + 331)
    denominator_scalar2 = (log((8 * (80 * d + 1) * (80 * d + 11)) / ((160 * d + 13) * (160 * d + 49))) -
                           log((80 * (80 * d + 11)) / (320 * d * (80 * d + 13) + 331)))
    return numerator / (denominator_scalar * denominator_scalar2)


if __name__ == "__main__":
    deltas = np.linspace(0.0, 0.06, 100)
    g3s = [gamma_3(d) for d in deltas]
    g2s = [gamma_2(d) for d in deltas]
    cross1 = 0.02251
    cross2 = 0.04966

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.axvline(cross1, color='red', linestyle='dashed', label=rf"$\delta={cross1}$")
    plt.axvline(cross2, color='purple', linestyle='dashed', label=rf"$\delta={cross2}$")
    plt.plot(deltas, g2s, label=r"2-community ground truth $\gamma$ estimate")
    plt.plot(deltas, g3s, label=r"3-community ground truth $\gamma$ estimate")
    plt.plot(deltas, gamma_cross_limit(deltas), label="Optimality switch from $K=2$ to $K=3$")
    plt.xlabel(r"$\delta$", fontsize=14)
    plt.ylabel(r"$\gamma$", fontsize=14)
    plt.title("Analytic Solutions for Example Bistable SBM", fontsize=14)
    plt.legend()
    plt.xlim(PLOT_XLIM)
    plt.savefig("analytic_bistable_sbm.pdf")
