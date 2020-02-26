# Generates figures A.1 and A.2

from math import exp
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.special import lambertw

PLOT_START = 0
PLOT_END = 3
NUM_COLOR_BANDS = 11


def parametric_omega(gamma, other_omega):
    if gamma == 0:
        return 0
    if other_omega < gamma:
        return -gamma * lambertw(-other_omega * exp(-other_omega / gamma) / gamma, k=-1).real
    return -gamma * lambertw(-other_omega * exp(-other_omega / gamma) / gamma).real


def plot_figure1():
    # plots the left panel of figure A.1
    plt.close()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    gamma_contours = [[parametric_omega(gamma, x) for x in xs] for gamma in color_band_starts]

    for i, gamma_start in enumerate(color_band_starts):
        plt.plot(xs, gamma_contours[i], color=cmap(norm(gamma_start)))
        if i + 1 < len(gamma_contours):
            plt.fill_between(xs, gamma_contours[i], gamma_contours[i + 1], color=cmap(norm(gamma_start)))

    plt.xlabel(r"$\omega_{in}$", fontsize=14)
    plt.ylabel(r"$\omega_{out}$", fontsize=14)
    plt.xlim([PLOT_START, PLOT_END])
    plt.ylim([PLOT_START, PLOT_END])
    plt.gca().set_aspect('equal', adjustable='box')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ticks=color_band_starts, boundaries=color_band_starts)
    cbar.set_label(r"$\gamma$", fontsize=14)
    plt.title(r"$\gamma$ Estimates for Various $\omega_{in}$ and $\omega_{out}$", fontsize=14)
    plt.savefig("gamma_omega_map1.pdf")


def plot_figure2():
    # plots the right panel of figure A.1
    plt.close()
    plt.plot([PLOT_START, PLOT_END], [PLOT_START, PLOT_END], linestyle="dashed", color="black")
    for gamma_center in gamma_centers:
        plt.plot(xs, [parametric_omega(gamma_center, x) for x in xs], label=rf"$\gamma={gamma_center}$")
    plt.xlabel(r"$\omega_{in}$", fontsize=14)
    plt.ylabel(r"$\omega_{out}$", fontsize=14)
    plt.xlim([PLOT_START, PLOT_END])
    plt.ylim([PLOT_START, PLOT_END])
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(r"$\gamma$ Estimates for Various $\omega_{in}$ and $\omega_{out}$", fontsize=14)
    plt.savefig("gamma_omega_map2.pdf")


def plot_figure3():
    # plots figure A.2
    plt.close()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot([PLOT_START, PLOT_END], [PLOT_START, PLOT_END], linestyle="dashed", color="gray")
    for gamma_center in gamma_centers:
        plt.fill_between(xs,
                         [parametric_omega(gamma_center - 0.1, x) for x in xs],
                         [parametric_omega(gamma_center + 0.1, x) for x in xs],
                         label=rf"$\gamma={gamma_center} \pm 0.1$")
        plt.plot(xs, [parametric_omega(gamma_center, x) for x in xs], color="black")

    plt.xlabel(r"$\omega_{in}$", fontsize=14)
    plt.ylabel(r"$\omega_{out}$", fontsize=14)
    plt.xlim([PLOT_START, PLOT_END])
    plt.ylim([PLOT_START, PLOT_END])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(r"$\Omega$ Values for Ranges of $\gamma$ Estimates")
    plt.legend()
    plt.savefig("gamma_omega_map_ranges.pdf")


if __name__ == "__main__":
    xs = np.linspace(PLOT_START, PLOT_END, 1000)
    gamma_centers = [0.5, 1.0, 1.5, 2.0]
    color_band_starts = np.linspace(PLOT_START, PLOT_END, NUM_COLOR_BANDS)
    cmap = matplotlib.cm.get_cmap('viridis')
    norm = matplotlib.colors.Normalize(vmin=PLOT_START, vmax=PLOT_END)

    plot_figure1()
    plot_figure2()
    plot_figure3()
