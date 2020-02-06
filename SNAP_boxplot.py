# This generates figure 6.4

from utilities import *
import os
import louvain
from social_networks import read_graphs
from multiprocessing import Pool, cpu_count
from utilities import num_communities
import matplotlib.pyplot as plt
from math import log
import numpy as np
from time import time
import pickle

Gs = read_graphs()
print("Import complete.")


def run_louvain(graphnum):
    G = Gs[graphnum]
    parts = []
    start = time()

    for gamma in np.linspace(0, 10, 1000):
        part = louvain.find_partition(G, louvain.RBConfigurationVertexPartition, resolution_parameter=gamma).membership

        if num_communities(part) > 100:
            break
        else:
            parts.append(part)

    print("Running on Graph {}, n={}, m={}: "
          "In {:.2f} s, found {} partitions at {:.2f} seconds per partition"
          "".format(graphnum, G.vcount(), G.ecount(), time() - start, len(parts), (time() - start) / len(parts)))
    return graphnum, parts


graphnums = [i for i in list(range(len(Gs))) if not os.path.exists(f"parts{i}.p")]
pool = Pool(cpu_count())
for graphnum, parts in pool.map(run_louvain, graphnums):
    pickle.dump(parts, open("parts{}.p".format(graphnum), "wb"))
pool.close()

K_MAX = 71


def f(K, x):
    if x == 0:
        return 0
    elif x == 1:
        return 1
    return K * (1 / x - 1) / (1 / x + K - 1) / log(1 / x)


def fp(K, p_in, p_out):
    val = f(K, p_out / p_in)
    if val == 0:
        print(K, p_in, p_out)
    return val


if not os.path.exists("maximum_gammas.p"):
    xs = np.linspace(0, 1.0, 2 ** 12)
    gamma_means = [0] * K_MAX
    gamma_maxs = [0] * K_MAX

    for k in range(2, K_MAX):
        ys = [fp(k, p_in, p_out) for p_in in xs for p_out in xs if 1 > p_in > p_out > 0]
        current_mean, current_max = np.mean(ys), np.max(ys)
        print("K={}: mean={:.3f}, max={:.3f}".format(k, current_mean, current_max))
        gamma_means[k] = current_mean
        gamma_maxs[k] = current_max

    pickle.dump((gamma_means, gamma_maxs), open("maximum_gammas.p", "wb"))

Ks = list(range(2, K_MAX))
est_means, est_maxs = pickle.load(open("maximum_gammas.p", "rb"))
est_means = est_means[2:]
est_maxs = est_maxs[2:]
assert abs(est_maxs[0] - 1.0) < 1e-4

if not os.path.exists("boxplot_results.p"):
    Gs = read_graphs()
    print("Graphs imported.")

    results = []
    for i in range(16):
        print("Processing parts{}...".format(i))
        for p in pickle.load(open("parts{}.p".format(i), "rb")):
            K = num_communities(p)
            g_est = gamma_estimate(Gs[i], p)

            if g_est is not None and 2 <= K < K_MAX:
                assert g_est < 15
                results.append((K, g_est))

    pickle.dump(results, open("boxplot_results.p", "wb"))

results = pickle.load(open("boxplot_results.p", "rb"))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure(figsize=(10, 5))
plt.plot(Ks, est_means, '-', label=r"$\gamma_{mean}$")
plt.plot(Ks, est_maxs, '-', label=r"$\gamma_{max}$")

boxplots = []
for K in Ks:
    gamma_ests = [x[1] for x in results if x[0] == K]
    boxplots.append(gamma_ests)
    print("K={}, count={}".format(K, len(gamma_ests)))

plt.boxplot(boxplots, sym='+', flierprops={"markersize": 5}, medianprops={"color": "black"}, positions=Ks)
plt.title(r"Empirical $\gamma$ Estimates from SNAP Networks as $K$ Varies", fontsize=14)
plt.ylabel(r"$\gamma$", fontsize=14)
plt.xlabel(r"Number of Communities $K$", fontsize=14)
plt.legend(fontsize=14)
plt.ylim([0, 10])

ax = plt.axes()
xticks = ax.xaxis.get_major_ticks()
for i in range(len(xticks)):
    xticks[i].label1.set_visible(False)

for i in range(0, len(xticks), 4):
    xticks[i].label1.set_visible(True)

plt.savefig("empirical_gamma_max.pdf")
