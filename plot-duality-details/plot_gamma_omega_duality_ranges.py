from math import exp, log
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import lambertw


def gamma_estimate(omega_in, omega_out):
    if omega_in == 0 or omega_out == 0 or omega_in == omega_out:
        return np.nan
    return (omega_in - omega_out) / (log(omega_in) - log(omega_out))


def parametric_omega(gamma, other_omega):
    if gamma == 0:
        return 0
    if other_omega < gamma:
        return -gamma * lambertw(-other_omega * exp(-other_omega / gamma) / gamma, k=-1).real
    return -gamma * lambertw(-other_omega * exp(-other_omega / gamma) / gamma).real


NUM_POINTS = 1000
xs = np.linspace(0, 3.0, NUM_POINTS)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot([0, 3], [0, 3], linestyle="dashed", color="gray")
for gamma_center in [0.5, 1.0, 1.5, 2.0]:
    plt.fill_between(xs,
                     [parametric_omega(gamma_center - 0.1, x) for x in xs],
                     [parametric_omega(gamma_center + 0.1, x) for x in xs],
                     label=r"$\gamma={} \pm 0.1$".format(gamma_center))
    plt.plot(xs, [parametric_omega(gamma_center, x) for x in xs], color="black")

plt.xlabel(r"$\omega_{in}$", fontsize=14)
plt.ylabel(r"$\omega_{out}$", fontsize=14)
plt.xlim([-0.00, 3.00])
plt.ylim([-0.00, 3.00])
plt.gca().set_aspect('equal', adjustable='box')
plt.title(r"$\Omega$ Values for Ranges of $\gamma$ Estimates")
plt.legend()
plt.savefig("gamma_omega_map_ranges.pdf")
