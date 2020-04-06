import matplotlib.pyplot as plt
import numpy as np
from utilities import *
import pickle

GAMMA_START = 0.0
GAMMA_END = 2.5


def generate_hierarchical_SBM(N=450):
    print("Generating hierarchical SBM...")

    p1 = 0.5 / (2 * N / 3)  # expect 0.5 edges per node out of community
    p2 = 3 / (2 * N / 9)  # expect 3 edges per node inside larger communities
    p3 = 6 / (1 * N / 9)  # expect 6 edges per node inside smaller communities

    B = N // 9
    pref_matrix = [[p3 if i == j else p2 if i // 3 == j // 3 else p1 for i in range(9)] for j in range(9)]
    block_sizes = [B] * 9

    G = ig.Graph.SBM(N, pref_matrix, block_sizes)

    while not G.is_connected():
        G = ig.Graph.SBM(N, pref_matrix, block_sizes)

    # print(f"Mean degree is {np.mean(G.degree())}")
    assert G.is_connected()

    ground_truth9 = [i // B for i in range(N)]
    ground_truth3 = [i // (3 * B) for i in range(N)]

    return G, ground_truth3, ground_truth9


def run_louvain(G, gamma_start=GAMMA_START, gamma_end=GAMMA_END, gamma_iters=10000):
    print("Running Louvain...")

    gammas = np.linspace(gamma_start, gamma_end, gamma_iters)
    all_parts = repeated_parallel_louvain_from_gammas(G, gammas, show_progress=True)
    return all_parts


def run_CHAMP(G, all_parts, gamma_start=GAMMA_START, gamma_end=GAMMA_END):
    print("Running CHAMP...")

    ranges = CHAMP_2D(G, all_parts, gamma_start, gamma_end)
    gamma_estimates = ranges_to_gamma_estimates(G, ranges)
    return gamma_estimates


def plot_CHAMP_gamma_estimates(gamma_estimates):
    print("Plotting...")

    # Plot gamma estimates and domains of optimality when the number of communities is not restricted
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_estimates(gamma_estimates)
    plt.title("Domains of Optimality and Gamma Estimates", fontsize=14)
    plt.xlabel(r"$\gamma$", fontsize=14)
    plt.ylabel("Number of communities", fontsize=14)


if __name__ == "__main__":
    ...
    while True:
        G, gt3, gt9 = generate_hierarchical_SBM()

        all_parts = run_louvain(G)
        gamma_estimates = run_CHAMP(G, all_parts)
        stable_parts = gamma_estimates_to_stable_partitions(gamma_estimates)

        if len([p for p in stable_parts if num_communities(p) <= 9]) > 3:
            pickle.dump((G, gt3, gt9), open("hierarchical_SBM.p", "wb"))
            plot_CHAMP_gamma_estimates(gamma_estimates)
            plt.show()
            break

G, gt3, gt9 = pickle.load(open("hierarchical_SBM.p", "rb"))
all_parts = run_louvain(G)
gamma_estimates = run_CHAMP(G, all_parts)
stable_parts = gamma_estimates_to_stable_partitions(gamma_estimates)
plot_CHAMP_gamma_estimates(gamma_estimates)
plt.savefig("hierarchical_sbm_gamma_estimates.pdf")

layout = G.layout_fruchterman_reingold(niter=10 ** 3)
for p in stable_parts:
    out = ig.plot(louvain.RBConfigurationVertexPartition(G, p), bbox=(1000, 1000), layout=layout)
    out.save(f"hierarchical_sbm_{num_communities(p)}-community.png")
