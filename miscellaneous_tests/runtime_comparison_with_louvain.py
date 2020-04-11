import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from modularitypruning import prune_to_stable_partitions
from modularitypruning.louvain_utilities import repeated_parallel_louvain_from_gammas, repeated_louvain_from_gammas
import pickle
import os
from time import time

GAMMA_START = 0
GAMMA_END = 3


def generate_runtimes(G, num_gammas):
    louvain_durations = []
    pruning_durations = []
    num_unique_partitions = []

    for num_louvain_iterations in num_gammas:
        gammas = np.linspace(GAMMA_START, GAMMA_END, num_louvain_iterations)
        louvain_start = time()
        parts = repeated_parallel_louvain_from_gammas(G, gammas, show_progress=False, chunk_dispatch=False)
        louvain_duration = time() - louvain_start

        pruning_start = time()
        stable_parts = prune_to_stable_partitions(G, parts, GAMMA_START, GAMMA_END)
        pruning_duration = time() - pruning_start

        louvain_durations.append(louvain_duration)
        pruning_durations.append(pruning_duration)
        num_unique_partitions.append(len(parts))

    return louvain_durations, pruning_durations, num_unique_partitions


if __name__ == "__main__":
    num_gammas = range(0, 25001, 1000)
    G = ig.Graph.Erdos_Renyi(n=1000, m=5000, directed=False)
    while not G.is_connected():
        G = ig.Graph.Erdos_Renyi(n=1000, m=5000, directed=False)

    if not os.path.exists("runtime_comparison_results.p"):
        pickle.dump(generate_runtimes(G, num_gammas), open("runtime_comparison_results.p", "wb"))

    louvain_durations, pruning_durations, num_unique_partitions = pickle.load(open("runtime_comparison_results.p",
                                                                                   "rb"))

    plt.figure()
    plt.plot(num_gammas, louvain_durations, linestyle='--', marker='o', label="Louvain")
    plt.plot(num_gammas, pruning_durations, linestyle='--', marker='o', label="ModularityPruning")
    plt.title("Runtime of Louvain and ModularityPruning on G(n=1000, m=5000)")
    plt.xlabel("Number of Louvain iterations")
    plt.ylabel("Runtime (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ER_louvain_pruning_runtime.pdf")

    plt.figure()
    plt.plot(num_gammas, num_unique_partitions, linestyle='--', marker='o')
    plt.title("Number of Unique Partitions Returned by Louvain on G(n=1000, m=5000)")
    plt.xlabel("Number of Louvain iterations")
    plt.ylabel("Number of unique partitions")
    plt.tight_layout()
    plt.savefig("ER_unique_partition_count.pdf")
