# Runs our pruning method on a hierarchical SBM with 3 and 9 block ground truths

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import louvain
from modularitypruning.champ_utilities import CHAMP_2D
from modularitypruning.louvain_utilities import repeated_parallel_louvain_from_gammas
from modularitypruning.parameter_estimation_utilities import ranges_to_gamma_estimates, \
    gamma_estimates_to_stable_partitions
from modularitypruning.partition_utilities import num_communities
from modularitypruning.plotting import plot_estimates
from modularitypruning.progress import Progress
import pickle
import os

GAMMA_START = 0.0
GAMMA_END = 2.5


def generate_hierarchical_SBM(N=450):
    p1 = 0.5 / (2 * N / 3)  # expect 0.5 edges per node out of community
    p2 = 3 / (2 * N / 9)  # expect 3 edges per node inside larger communities
    p3 = 6 / (1 * N / 9)  # expect 6 edges per node inside smaller communities

    B = N // 9
    pref_matrix = [[p3 if i == j else p2 if i // 3 == j // 3 else p1 for i in range(9)] for j in range(9)]
    block_sizes = [B] * 9

    G = ig.Graph.SBM(N, pref_matrix, block_sizes)

    while not G.is_connected():
        G = ig.Graph.SBM(N, pref_matrix, block_sizes)

    assert G.is_connected()

    ground_truth9 = [i // B for i in range(N)]
    ground_truth3 = [i // (3 * B) for i in range(N)]

    return G, ground_truth3, ground_truth9


def run_louvain(G, gamma_start=GAMMA_START, gamma_end=GAMMA_END, gamma_iters=10000):
    gammas = np.linspace(gamma_start, gamma_end, gamma_iters)
    all_parts = repeated_parallel_louvain_from_gammas(G, gammas, show_progress=False, chunk_dispatch=False)
    return all_parts


def run_CHAMP(G, all_parts, gamma_start=GAMMA_START, gamma_end=GAMMA_END):
    ranges = CHAMP_2D(G, all_parts, gamma_start, gamma_end)
    gamma_estimates = ranges_to_gamma_estimates(G, ranges)
    return gamma_estimates


def plot_CHAMP_gamma_estimates(gamma_estimates):
    # Plot gamma estimates and domains of optimality when the number of communities is not restricted
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_estimates(gamma_estimates)
    plt.title(r"Domains of Optimality and $\gamma$ Estimates", fontsize=14)
    plt.xlabel(r"$\gamma$", fontsize=14)
    plt.ylabel("Number of communities", fontsize=14)


def plot_SBM_example():
    G, gt3, gt9 = generate_hierarchical_SBM()
    layout = G.layout_fruchterman_reingold(niter=10 ** 3)
    for p in [gt3, gt9]:
        ig.plot(louvain.RBConfigurationVertexPartition(G, p), bbox=(1000, 1000), layout=layout,
                target=f"hierarchical_sbm_example_{num_communities(p)}.png")


def find_example_with_4_stable_partitions():
    while True:
        G, gt3, gt9 = generate_hierarchical_SBM()

        all_parts = run_louvain(G)
        gamma_estimates = run_CHAMP(G, all_parts)
        stable_parts = gamma_estimates_to_stable_partitions(gamma_estimates)

        num_stable_partitions_below_nine = len([p for p in stable_parts if num_communities(p) <= 9])

        if num_stable_partitions_below_nine > 3:
            all_parts = run_louvain(G)
            gamma_estimates = run_CHAMP(G, all_parts)
            # stable_parts = gamma_estimates_to_stable_partitions(gamma_estimates)
            plot_CHAMP_gamma_estimates(gamma_estimates)
            plt.savefig("hierarchical_sbm_gamma_estimates.pdf")
            plt.close()

            layout = G.layout_fruchterman_reingold(niter=10 ** 3)
            for p in stable_parts:
                ig.plot(louvain.RBConfigurationVertexPartition(G, p), bbox=(1000, 1000), layout=layout,
                        target=f"hierarchical_sbm_{num_communities(p)}-community.png")
            return
        else:
            print(f"Trial completed with {num_stable_partitions_below_nine} partitions with K <= 9. Continuing...")


def find_stability_probabilities(num_trials=500):
    progress = Progress(num_trials)
    num_stable = [0 for _ in range(20)]
    for _ in range(num_trials):
        G, gt3, gt9 = generate_hierarchical_SBM()

        all_parts = run_louvain(G)
        gamma_estimates = run_CHAMP(G, all_parts)
        stable_parts = gamma_estimates_to_stable_partitions(gamma_estimates)

        for p in stable_parts:
            assert num_communities(p) < len(num_stable)
            num_stable[num_communities(p)] += 1
        progress.increment()
    progress.done()

    stability_probabilities = [x / num_trials for x in num_stable]
    pickle.dump(stability_probabilities, open("hierarchical_stability_probabilities.p", "wb"))


if __name__ == "__main__":
    plot_SBM_example()
    find_example_with_4_stable_partitions()

    if not os.path.exists("hierarchical_stability_probabilities.p"):
        find_stability_probabilities()

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    stability_probabilities = pickle.load(open("hierarchical_stability_probabilities.p", "rb"))
    plt.bar(list(range(3, 17)), stability_probabilities[3:17])
    plt.title("Frequency of finding stable partitions by number of communities", fontsize=14)
    plt.ylabel(r"Probability of finding a stable partition", fontsize=14)
    plt.xlabel(r"Number of communities $K$", fontsize=14)
    plt.xticks(list(range(3, 17)))
    plt.gca().tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig("hierarchical_sbm_stability_probs.pdf")
