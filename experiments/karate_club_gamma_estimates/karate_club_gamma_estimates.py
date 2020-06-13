# Generates figure 2.1

import igraph as ig
import matplotlib.pyplot as plt
from modularitypruning.parameter_estimation import iterative_monolayer_resolution_parameter_estimation
from modularitypruning.progress import Progress
import numpy as np


def one_estimation_step(G, gamma, method):
    return iterative_monolayer_resolution_parameter_estimation(G, gamma=gamma, max_iter=1, method=method)[0]


def run_estimation_until_convergence(method):
    G = ig.Graph.Famous("Zachary")
    gamma_0s = []
    gamma_fs = []

    progress = Progress(300, length=50, name=f"{method} until convergence:")
    for gamma in np.linspace(0.3, 3.0, 300):
        estimation_trials = 10 if method == "louvain" else 1
        for _ in range(estimation_trials):
            try:
                final_gamma, _ = iterative_monolayer_resolution_parameter_estimation(G, gamma=gamma, method=method)
                gamma_0s.append(gamma)
                gamma_fs.append(final_gamma)
            except ValueError:
                pass
        progress.increment()
    progress.done()

    return gamma_0s, gamma_fs


def plot_one_step_estimation(method):
    G = ig.Graph.Famous("Zachary")

    progress = Progress(45, length=50, name=f"{method} one step:")
    for g0 in np.linspace(0.3, 3.0, 45):
        total = 0
        count = 0
        estimation_trials = 100 if method == "louvain" else 1
        for _ in range(estimation_trials):
            try:
                total += one_estimation_step(G, gamma=g0, method=method) - g0
                count += 1
            except ValueError:
                # print('value error on ', g0)
                pass
        if count > 0:
            average_gamma_movement = total / count
            plt.plot([g0], [g0], c='black', marker='o', markersize=2)
            plt.arrow(g0, g0, average_gamma_movement, 0, width=0.005, head_length=0.04, head_width=0.04,
                      length_includes_head=True, color="black")
        progress.increment()
    progress.done()


if __name__ == "__main__":
    # Louvain generates the left side of the figure
    # Spinglass generates the right side of the figure
    for method in ["louvain", "2-spinglass"]:
        gamma_0s, gamma_fs = run_estimation_until_convergence(method)

        plt.figure()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.xlim([0.2, 3.1])
        plt.ylim([0.2, 3.1])
        plt.ylabel(r"Initial $\gamma$", size=14)
        plt.xlabel(r"Final $\gamma$", size=14)

        plot_one_step_estimation(method)

        if method == "louvain":
            plt.scatter(gamma_fs, gamma_0s, alpha=0.1, s=15)
            plt.title(r"$\gamma$ Estimation on the Karate Club with Louvain", fontsize=14)
            plt.tight_layout()
            plt.savefig("karate_club_gamma_estimation_louvain.pdf")
        else:
            plt.scatter(gamma_fs, gamma_0s, alpha=1.0, s=15)
            plt.title(r"$\gamma$ Estimation on the Karate Club with Spinglass", fontsize=14)
            plt.tight_layout()
            plt.savefig("karate_club_gamma_estimation_spinglass.pdf")
