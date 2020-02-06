import matplotlib.pyplot as plt
from math import log
import numpy as np


def f(K, x): return K * (1 / x - 1) / (1 / x + K - 1) / log(1 / x)


# xs = np.linspace(1e-9, 1 - 1e-9, 1000)
#
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.xlabel(r"$p_{out}/p_{in}$", fontsize=14)
# plt.ylabel(r"$\gamma$", fontsize=14)
#
# plt.plot(xs, [f(2, x) for x in xs], label=r"$K=2$")
# plt.plot(xs, [f(3, x) for x in xs], label=r"$K=3$")
# plt.plot(xs, [f(4, x) for x in xs], label=r"$K=4$")
# plt.plot(xs, [f(5, x) for x in xs], label=r"$K=5$")
# plt.plot(xs, [f(6, x) for x in xs], label=r"$K=6$")
# plt.plot(xs, [f(7, x) for x in xs], label=r"$K=7$")
# plt.plot(xs, [f(8, x) for x in xs], label=r"$K=8$")
# plt.title(r"Expected $\gamma$ Estimates for Various $K$", fontsize=14)
# plt.legend()
# plt.savefig("maximum_gamma_estimates.pdf")

last_vals = [0] * 20
current_vals = [0] * 20

for num_points in [2 ** x for x in range(10, 30)]:
    xs = np.linspace(1e-9, 1 - 1e-9, num_points)
    print("===== {} points =====".format(num_points))
    for K in range(2, 20):
        maximum_gamma = max(f(K, x) for x in xs)
        print("K={}, max gamma is {:.4f}".format(K, maximum_gamma))
        current_vals[K] = maximum_gamma

    if all(abs(lv - cv) < 1e-6 for lv, cv in zip(last_vals, current_vals)):
        print("Done")
        break
    else:
        last_vals = [x for x in current_vals]
