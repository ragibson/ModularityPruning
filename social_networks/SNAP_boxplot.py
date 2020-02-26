# Generates figure 6.4

import os
import louvain
from multiprocessing import Pool, cpu_count
from utilities import num_communities, gamma_estimate, sorted_tuple
from utilities import gamma_estimate_from_parameters as gamma
from social_networks import read_graphs
import matplotlib.pyplot as plt
import numpy as np
from time import time
import pickle

K_MAX = 71


def run_louvain(graphnum):
    G = Gs[graphnum]
    parts = []
    start = time()

    for gamma in np.linspace(0, 10, 1000):
        part = louvain.find_partition(G, louvain.RBConfigurationVertexPartition,
                                      resolution_parameter=gamma).membership

        if num_communities(part) > 100:
            break
        else:
            parts.append(part)

    print(f"Running on Graph {graphnum}, n={G.vcount()}, m={G.ecount()}: "
          f"In {time() - start:.2f} s, found {len(parts)} partitions at {(time() - start) / len(parts):.2f} "
          "seconds per partition")
    return graphnum, {sorted_tuple(tuple(p)) for p in parts}


def run_louvain_if_necessary():
    graphnums = [i for i in range(len(Gs)) if not os.path.exists(f"parts{i}.p")]
    pool = Pool(cpu_count())
    for graphnum, parts in pool.map(run_louvain, graphnums):
        pickle.dump(parts, open(f"parts{graphnum}.p", "wb"))
    pool.close()


def find_maximum_gammas():
    def gamma_maxmean(K, points=100):
        # omega_in + (K-1)*omega_out = K
        total = 0
        count = 0
        max_gamma = 0
        for omega_in in np.linspace(1.0, K, points):
            for omega_out in np.linspace(0, K / (K - 1), points):
                if omega_in + (K - 1) * omega_out > K:
                    break

                current_gamma = gamma(omega_in, omega_out)
                if current_gamma is not None:
                    max_gamma = max(max_gamma, current_gamma)
                    total += gamma(omega_in, omega_out)
                    count += 1

        return total / count, max_gamma

    gamma_means = [0] * K_MAX
    gamma_maxs = [0] * K_MAX

    for k in range(2, K_MAX):
        current_mean, current_max = gamma_maxmean(k, points=1000)
        gamma_means[k] = current_mean
        gamma_maxs[k] = current_max

    pickle.dump((gamma_means, gamma_maxs), open("maximum_gammas.p", "wb"))


def generate_boxplot_results():
    results = []
    for i in range(len(Gs)):
        for p in pickle.load(open(f"parts{i}.p", "rb")):
            K = num_communities(p)
            g_est = gamma_estimate(Gs[i], p)

            if g_est is not None and 2 <= K < K_MAX:
                assert g_est < 15
                results.append((K, g_est))

    pickle.dump(results, open("boxplot_results.p", "wb"))


def generate_SNAP_boxplot():
    Ks = list(range(2, K_MAX))
    est_means, est_maxs = pickle.load(open("maximum_gammas.p", "rb"))
    est_means = est_means[2:]
    est_maxs = est_maxs[2:]
    print(est_maxs[0])
    assert abs(est_maxs[0] - 1.0) < 1e-4

    boxplot_results = pickle.load(open("boxplot_results.p", "rb"))
    boxplots = []
    for K in Ks:
        gamma_ests = [x[1] for x in boxplot_results if x[0] == K]
        boxplots.append(gamma_ests)

    plt.close()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(10, 5))
    plt.plot(Ks, est_means, '-', label=r"$\gamma_{mean}$")
    plt.plot(Ks, est_maxs, '-', label=r"$\gamma_{max}$")
    plt.boxplot(boxplots, sym='+', flierprops={"markersize": 5}, medianprops={"color": "black"}, positions=Ks)
    plt.title(r"Empirical $\gamma$ Estimates from SNAP Networks as $K$ Varies", fontsize=14)
    plt.ylabel(r"$\gamma$", fontsize=14)
    plt.xlabel(r"Number of Communities $K$", fontsize=14)
    plt.legend(fontsize=14)
    plt.ylim([0, 10])

    ax = plt.gca()
    xticks = ax.xaxis.get_major_ticks()
    for i in range(len(xticks)):
        xticks[i].label1.set_visible(False)

    for i in range(0, len(xticks), 4):
        xticks[i].label1.set_visible(True)

    plt.savefig("empirical_gamma_max.pdf")


if __name__ == "__main__":
    Gs = read_graphs()

    run_louvain_if_necessary()

    if True or not os.path.exists("maximum_gammas.p"):
        print("Finding maximum gammas...")
        find_maximum_gammas()

    if not os.path.exists("boxplot_results.p"):
        print("Generating boxplot results...")
        generate_boxplot_results()

    generate_SNAP_boxplot()
