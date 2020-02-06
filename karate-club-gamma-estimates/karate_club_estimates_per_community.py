import igraph as ig
import matplotlib.pyplot as plt
from utilities import iterative_monolayer_resolution_parameter_estimation, Progress, num_communities
from multiprocessing import Pool, cpu_count
from utilities import CHAMP_2D
import louvain
import numpy as np
from time import time

G = ig.Graph.Famous("Zachary")


def one_run(gamma):
    try:
        final_gamma, part = iterative_monolayer_resolution_parameter_estimation(G, gamma=gamma, verbose=False,
                                                                                max_iter=1)
        return final_gamma, part.membership
    except ValueError:
        return None, None


def run_chunk(gammas):
    local_parts = []
    for g0 in gammas:
        gf, membership = one_run(g0)
        if gf is not None:
            local_parts.append((gf, membership))
    return local_parts


louvain_parts = []
chunk_size = 10000
gammas = np.linspace(0.0, 2.0, 1000000)
pool = Pool(cpu_count())
progress = Progress(len(gammas) // chunk_size)
for i in range(0, len(gammas), chunk_size):
    for gf, membership in pool.map(one_run, gammas[i:i + chunk_size]):
        if gf is not None:
            louvain_parts.append((gf, membership))
    progress.increment()
progress.done()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

min_gamma = min(x[0] for x in louvain_parts)
max_gamma = max(x[0] for x in louvain_parts)
assert min_gamma > 0.6 and max_gamma < 1.6
bins = np.linspace(0.6, 1.6, 100)
for K in range(2, 10):
    K_community_gammas = [x[0] for x in louvain_parts if num_communities(x[1]) == K]
    if len(K_community_gammas) > 0:
        weights = np.zeros_like(K_community_gammas) + 1. / len(K_community_gammas)
        plt.hist(K_community_gammas, label="{} Communities".format(K), bins=bins, weights=weights, alpha=0.75)

plt.title(r"Dependence of $\gamma$ Estimate on $K$", fontsize=14)
plt.xlabel(r"$\gamma$ Estimate", fontsize=14)
plt.ylabel(r"Relative Frequency", fontsize=14)
plt.legend(loc='upper right')
plt.savefig("karate_club_gamma_estimates_per_K.pdf")