# Generates figure 6.2

from math import exp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import lambertw


def parametric_omega(gamma, other_omega):
    if gamma == 0:
        return 0
    if other_omega < gamma:
        return -gamma * lambertw(-other_omega * exp(-other_omega / gamma) / gamma, k=-1).real
    return -gamma * lambertw(-other_omega * exp(-other_omega / gamma) / gamma).real


if __name__ == "__main__":
    xs = np.linspace(0, 3.0, 500)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(xs, [(2 - x) / (2 - 1) for x in xs], label=r"Possible $\Omega$ values for $K=2$")
    plt.plot(xs, [(3 - x) / (3 - 1) for x in xs], label=r"Possible $\Omega$ values for $K=3$")
    plt.plot(xs, [parametric_omega(1.0, x) for x in xs], label=r"$\gamma=1.0$", linestyle="dashed")
    plt.plot(xs, [parametric_omega(1.0926, x) for x in xs], label=r"$\gamma=1.0926$", linestyle="dashed")
    plt.xlim([0.0, 3.0])
    plt.ylim([0.0, 2.0])
    plt.xlabel(r"$\omega_{in}$", fontsize=14)
    plt.ylabel(r"$\omega_{out}$", fontsize=14)
    plt.title(r"Maximum Expected $\gamma$ Estimates with $K=2,3$ and $N \to \infty$", fontsize=14)
    plt.legend()
    plt.savefig("maximum_gamma_estimates_in_omega_space.pdf")
