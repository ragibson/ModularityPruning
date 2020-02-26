# Generates figure 6.3

from math import exp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import lambertw

XMIN = 0
XMAX = 3
YMIN = 0
YMAX = 2


def parametric_omega(gamma, other_omega):
    if gamma == 0:
        return 0
    if other_omega < gamma:
        return -gamma * lambertw(-other_omega * exp(-other_omega / gamma) / gamma, k=-1).real
    return -gamma * lambertw(-other_omega * exp(-other_omega / gamma) / gamma).real


def plot_figure1():
    # plots the left panel of figure 6.3
    plt.close()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(xs, [(2 - x) / (2 - 1) for x in xs], color="C0")
    plt.fill_between(xs_fill_above, [(2 - x) / (2 - 1) for x in xs_fill_above], above_line, color="C0",
                     label=r"Possible $\Omega$ values", alpha=0.5)
    plt.fill_between(xs_fill_below, [(2 - x) / (2 - 1) for x in xs_fill_below], below_line, color="C0",
                     alpha=0.5)
    plt.plot(xs, [parametric_omega(1.0, x) for x in xs], label=r"$\gamma=1.0$", linestyle="dashed", color="C2")
    plt.xlim([XMIN, XMAX])
    plt.ylim([YMIN, YMAX])
    plt.xlabel(r"$\omega_{in}$", fontsize=14)
    plt.ylabel(r"$\omega_{out}$", fontsize=14)
    plt.title(r"Maximum Expected $\gamma$ Estimates, $K=2$", fontsize=14)
    plt.legend()
    plt.savefig("general_max_gamma2.pdf")


def plot_figure2():
    # plots the right panel of figure 6.3
    plt.close()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(xs, [(3 - x) / (3 - 1) for x in xs], color="C1")
    plt.fill_between(xs_fill_above, [(3 - x) / (3 - 1) for x in xs_fill_above], above_line, color="C1",
                     label=r"Possible $\Omega$ values", alpha=0.5)
    plt.fill_between(xs_fill_below, [(3 - x) / (3 - 1) for x in xs_fill_below], below_line, color="C1",
                     alpha=0.5)
    plt.plot(xs, [parametric_omega(1.0926, x) for x in xs], label=r"$\gamma=1.0926$", linestyle="dashed", color="C3")
    plt.xlim([XMIN, XMAX])
    plt.ylim([YMIN, YMAX])
    plt.xlabel(r"$\omega_{in}$", fontsize=14)
    plt.ylabel(r"$\omega_{out}$", fontsize=14)
    plt.title(r"Maximum Expected $\gamma$ Estimates, $K=3$", fontsize=14)
    plt.legend()
    plt.savefig("general_max_gamma3.pdf")


if __name__ == "__main__":
    xs = np.linspace(XMIN, XMAX, 500)
    xs_fill_above = np.linspace(XMIN, 1, len(xs))
    xs_fill_below = np.linspace(1, XMAX, len(xs))
    below_line = [YMIN] * len(xs)
    above_line = [YMAX] * len(xs)

    plot_figure1()
    plot_figure2()
