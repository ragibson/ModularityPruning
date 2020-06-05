# Generates figure 6.1

import matplotlib.pyplot as plt
from math import log
import numpy as np


def f(K, x):
    return K * (1 / x - 1) / (1 / x + K - 1) / log(1 / x)


def compute_and_print_maximum_gamma_estimates():
    last_vals = [0] * 20
    current_vals = [0] * 20

    num_points = 2 ** 10
    while True:
        xs = np.linspace(1e-9, 1 - 1e-9, num_points)
        print(f"===== {num_points} sample points =====")
        for K in range(2, 20):
            maximum_gamma = max(f(K, x) for x in xs)
            print(f"K={K}, max gamma is {maximum_gamma:.4f}")
            current_vals[K] = maximum_gamma

        if all(abs(lv - cv) < 1e-6 for lv, cv in zip(last_vals, current_vals)):
            print("Done")
            break
        else:
            last_vals = [x for x in current_vals]
            num_points *= 2


if __name__ == "__main__":
    xs = np.linspace(1e-9, 1 - 1e-9, 1000)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel(r"$p_{out}/p_{in}$", fontsize=14)
    plt.ylabel(r"$\gamma$", fontsize=14)

    for K in range(2, 9):
        plt.plot(xs, [f(K, x) for x in xs], label=rf"$K={K}$")
    plt.title(r"Expected $\gamma$ Estimates for Various $K$", fontsize=14)
    plt.legend()
    plt.savefig("maximum_gamma_estimates.pdf")
