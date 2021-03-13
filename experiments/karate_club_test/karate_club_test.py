# Generates figures 5.1 and 5.2

import igraph as ig
import louvain
import matplotlib.pyplot as plt
import numpy as np
from modularitypruning.champ_utilities import CHAMP_2D
from modularitypruning.louvain_utilities import repeated_parallel_louvain_from_gammas
from modularitypruning.parameter_estimation_utilities import gamma_estimates_to_stable_partitions, \
    ranges_to_gamma_estimates
from modularitypruning.partition_utilities import num_communities
from modularitypruning.plotting import plot_estimates

GAMMA_START = 0.0
GAMMA_END = 2.0
GAMMA_ITERS = 10000000


def run_louvain():
    G = ig.Graph.Famous("Zachary")
    gammas = np.linspace(GAMMA_START, GAMMA_END, GAMMA_ITERS)
    return repeated_parallel_louvain_from_gammas(G, gammas, show_progress=True)


def plot_gamma_estimates(all_parts):
    G = ig.Graph.Famous("Zachary")

    # Print details on the CHAMP sets when the number of communities is restricted
    # print(len(all_parts), "unique partitions in total")
    # for K in range(2, 9):
    #     restricted_parts = {p for p in all_parts if num_communities(p) == K}
    #     print(f"{len(restricted_parts)} unique partitions with {K} communities")
    #     ranges = CHAMP_2D(G, restricted_parts, GAMMA_START, GAMMA_END)
    #     print(f"{len(ranges)} unique partitions in {K}-community CHAMP set")
    #     print("=" * 50)

    ranges = CHAMP_2D(G, all_parts, GAMMA_START, GAMMA_END)
    gamma_estimates = ranges_to_gamma_estimates(G, ranges)

    # Print details on the CHAMP set when the number of communities is not restricted
    # community_counts = [0] * 9
    # for _, _, membership, _ in gamma_estimates:
    #     community_counts[num_communities(membership)] += 1
    # for k, count in enumerate(community_counts):
    #     print(f"{count} unique partitions with {k} communities in total CHAMP set")

    # Plot gamma estimates and domains of optimality when the number of communities is not restricted
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_estimates(gamma_estimates)
    plt.title(r"Karate Club CHAMP Domains of Optimality and $\gamma$ Estimates", fontsize=14)
    plt.xlabel(r"$\gamma$", fontsize=14)
    plt.ylabel("Number of communities", fontsize=14)
    plt.savefig("karate_club_CHAMP_gamma_estimates.pdf")


def plot_stable_partitions(all_parts):
    G = ig.Graph.Famous("Zachary")

    # Store shared force-directed layout to make later plotting layouts consistent
    layout = G.layout_fruchterman_reingold(niter=1000)

    # Plot stable partitions when the number of communities is restricted to 2-4
    for K in range(2, 5):
        restricted_parts = {p for p in all_parts if num_communities(p) == K}

        if len(restricted_parts) > 0:
            ranges = CHAMP_2D(G, restricted_parts, GAMMA_START, GAMMA_END)
            gamma_estimates = ranges_to_gamma_estimates(G, ranges)
            stable_parts = gamma_estimates_to_stable_partitions(gamma_estimates)

            for i, p in enumerate(stable_parts):
                ig.plot(louvain.RBConfigurationVertexPartition(G, initial_membership=p),
                        f"karate_club_{K}_stable{i}.png", bbox=(600, 600), layout=layout)


if __name__ == "__main__":
    all_parts = run_louvain()
    plot_gamma_estimates(all_parts)
    plot_stable_partitions(all_parts)
