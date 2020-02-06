from math import exp, log
import matplotlib.pyplot as plt
import matplotlib
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
xs = np.linspace(0, 3, NUM_POINTS)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

gamma_contours = [[parametric_omega(gamma, x) for x in xs] for gamma in np.linspace(0.0, 3.0, 11)]
cmap = matplotlib.cm.get_cmap('viridis')
norm = matplotlib.colors.Normalize(vmin=0.0, vmax=3.0)

for i, gamma_start in zip(range(len(gamma_contours) - 1), np.linspace(0.0, 3.0, 11)):
    plt.plot(xs, gamma_contours[i], color=cmap(norm(gamma_start)))
    plt.fill_between(xs, gamma_contours[i], gamma_contours[i + 1], color=cmap(norm(gamma_start)))

# X, Y = np.meshgrid(xs, xs, indexing='xy')
# Z = np.array([gamma_estimate(x, y) for x in xs for y in xs]).reshape(NUM_POINTS, NUM_POINTS)
# plt.contourf(X, Y, Z, 10)

plt.xlabel(r"$\omega_{in}$", fontsize=14)
plt.ylabel(r"$\omega_{out}$", fontsize=14)
plt.xlim([-0.00, 3.00])
plt.ylim([-0.00, 3.00])
plt.gca().set_aspect('equal', adjustable='box')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ticks=np.linspace(0, 3.0, 11), boundaries=np.linspace(0, 3.0, 11))
cbar.set_label(r"$\gamma$", fontsize=14)
plt.title(r"$\gamma$ Estimates for Various $\omega_{in}$ and $\omega_{out}$", fontsize=14)
plt.savefig("gamma_omega_map1.pdf")

plt.close()
plt.plot([0, 3], [0, 3], linestyle="dashed", color="black")
plt.plot(xs, [parametric_omega(0.5, x) for x in xs], label=r"$\gamma=0.5$")
plt.plot(xs, [parametric_omega(1.0, x) for x in xs], label=r"$\gamma=1.0$")
plt.plot(xs, [parametric_omega(1.5, x) for x in xs], label=r"$\gamma=1.5$")
plt.plot(xs, [parametric_omega(2.0, x) for x in xs], label=r"$\gamma=2.0$")
plt.xlabel(r"$\omega_{in}$", fontsize=14)
plt.ylabel(r"$\omega_{out}$", fontsize=14)
plt.xlim([-0.00, 3.00])
plt.ylim([-0.00, 3.00])
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.title(r"$\gamma$ Estimates for Various $\omega_{in}$ and $\omega_{out}$", fontsize=14)
plt.savefig("gamma_omega_map2.pdf")
