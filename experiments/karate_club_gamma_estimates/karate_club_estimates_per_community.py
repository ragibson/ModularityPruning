# Generates figure 2.2

import igraph as ig
import matplotlib.pyplot as plt
from modularitypruning.parameter_estimation import iterative_monolayer_resolution_parameter_estimation
from modularitypruning.partition_utilities import num_communities
from multiprocessing import Pool, cpu_count
import numpy as np

if __name__ == "__main__":
    G = ig.Graph.Famous("Zachary")

    def one_run(gamma):
        """Returns (gamma_estimate, num_communities) from a gamma estimation run starting at :gamma:"""
        try:
            final_gamma, part = iterative_monolayer_resolution_parameter_estimation(G, gamma=gamma, max_iter=1)
            return final_gamma, num_communities(part.membership)
        except ValueError:
            return None, None

    gammas = np.linspace(0.0, 2.0, 10 ** 6)
    pool = Pool(cpu_count())
    gammas, Ks = list(zip(*[x for x in pool.map(one_run, gammas) if x[0] is not None]))
    pool.close()

    assert min(gammas) > 0.6 and max(gammas) < 1.6
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    bins = np.linspace(0.6, 1.6, 100)

    for fixed_K in range(2, 10):
        K_community_gammas = [gamma for gamma, K in zip(gammas, Ks) if K == fixed_K]
        if len(K_community_gammas) > 0:
            weights = np.zeros_like(K_community_gammas) + 1. / len(K_community_gammas)
            plt.hist(K_community_gammas, label=f"{fixed_K} Communities", bins=bins, weights=weights, alpha=0.75)

    plt.title(r"Dependence of $\gamma$ Estimate on $K$", fontsize=14)
    plt.xlabel(r"$\gamma$ Estimate", fontsize=14)
    plt.ylabel(r"Relative Frequency", fontsize=14)
    plt.legend(loc='upper right')
    plt.savefig("karate_club_gamma_estimates_per_K.pdf")
