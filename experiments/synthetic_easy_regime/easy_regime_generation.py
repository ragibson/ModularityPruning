# Generates figures 5.3, 5.4, 5.5, and 5.6

from random import random, randint
from collections import Counter
import igraph as ig
import os
from time import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from modularitypruning.champ_utilities import CHAMP_3D
from modularitypruning.louvain_utilities import repeated_parallel_louvain_from_gammas_omegas
from modularitypruning.parameter_estimation import iterative_multilayer_resolution_parameter_estimation
from modularitypruning.parameter_estimation_utilities import gamma_omega_estimate, domains_to_gamma_omega_estimates
from modularitypruning.partition_utilities import num_communities
from modularitypruning.plotting import plot_2d_domains_with_estimates, plot_2d_domains_with_ami, \
    plot_2d_domains_with_num_communities
from modularitypruning.progress import Progress

ITERATION_GAMMA_START = 0.6
ITERATION_GAMMA_END = 1.45
ITERATION_OMEGA_START = 0.4
ITERATION_OMEGA_END = 1.6

CHAMP_GAMMA_START = 0.0
CHAMP_GAMMA_END = 2.0
CHAMP_OMEGA_START = 0.0
CHAMP_OMEGA_END = 2.0


def generate_synthetic_network():
    eta = 0.7  # copying probability
    epsilon = 0.4  # p_in/p_out ratio
    p_in = 20 / 75
    p_out = epsilon * p_in

    n_per_layer = 150
    num_layers = 15
    layer_vec = [i // n_per_layer for i in range(n_per_layer * num_layers)]
    K = 2

    while True:
        comm_per_layer = [[0] * n_per_layer for _ in range(num_layers)]
        comm_per_layer[0] = [i // (n_per_layer // K) for i in range(n_per_layer)]

        comm_counts = Counter(comm_per_layer[0])
        assert all(v == comm_counts[0] for v in comm_counts.values())

        for layer in range(1, num_layers):
            for v in range(n_per_layer):
                p = random()
                if p < eta:  # copy community from last layer
                    comm_per_layer[layer][v] = comm_per_layer[layer - 1][v]
                else:  # assign random community
                    comm_per_layer[layer][v] = randint(0, K - 1)

        comm_vec = [item for sublist in comm_per_layer for item in sublist]

        intralayer_edges = []
        interlayer_edges = [(n_per_layer * layer + v, n_per_layer * layer + v + n_per_layer)
                            for layer in range(num_layers - 1)
                            for v in range(n_per_layer)]

        for v in range(len(comm_vec)):
            for u in range(v + 1, len(comm_vec)):
                if layer_vec[v] == layer_vec[u]:
                    p = random()
                    if comm_vec[v] == comm_vec[u]:
                        if p < p_in:
                            intralayer_edges.append((u, v))
                    else:
                        if p < p_out:
                            intralayer_edges.append((u, v))

        G_intralayer = ig.Graph(intralayer_edges)
        G_interlayer = ig.Graph(interlayer_edges, directed=True)

        ground_truth_gamma, ground_truth_omega = gamma_omega_estimate(G_intralayer, G_interlayer, layer_vec, comm_vec)

        if abs(ground_truth_gamma - 0.94) < 5e-3 and abs(ground_truth_omega - 0.98) < 5e-3:
            print(f"Accepted graph generation with ground truth (omega, gamma) = "
                  f"({ground_truth_omega:.3f}, {ground_truth_gamma:.3f})")
            break

    return G_intralayer, G_interlayer, layer_vec, comm_vec


def run_pamfil_iteration():
    G_intralayer, G_interlayer, layer_vec, _ = pickle.load(open("easy_regime_multilayer.p", "rb"))

    def one_step(gamma, omega):
        try:
            g_new, o_new, _ = iterative_multilayer_resolution_parameter_estimation(G_intralayer, G_interlayer,
                                                                                   layer_vec,
                                                                                   gamma=gamma, omega=omega, max_iter=1)
            return g_new, o_new
        except ValueError:
            return None, None

    values = []
    all_g0s = np.linspace(ITERATION_GAMMA_START, ITERATION_GAMMA_END, 15)
    all_o0s = np.linspace(ITERATION_OMEGA_START, ITERATION_OMEGA_END, 15)
    progress = Progress(len(all_g0s) * len(all_o0s))

    for g0 in all_g0s:
        for o0 in all_o0s:
            gdiffs = []
            odiffs = []
            for repeat in range(5):
                g1, o1 = one_step(g0, o0)
                if g1 is not None and o1 is not None:
                    gdiffs.append(g1 - g0)
                    odiffs.append(o1 - o0)
            else:
                values.append((g0, o0, gdiffs, odiffs))

            progress.increment()
    progress.done()

    return values


def run_easy_regime_louvain():
    G_intralayer, G_interlayer, layer_vec, ground_truth_comms = pickle.load(open("easy_regime_multilayer.p", "rb"))
    return repeated_parallel_louvain_from_gammas_omegas(G_intralayer, G_interlayer, layer_vec,
                                                        gammas=np.linspace(CHAMP_GAMMA_START, CHAMP_GAMMA_END, 225),
                                                        omegas=np.linspace(CHAMP_OMEGA_START, CHAMP_OMEGA_END, 225))


def plot_easy_regime_iteration():
    G_intralayer, G_interlayer, layer_vec, ground_truth_comms = pickle.load(open("easy_regime_multilayer.p", "rb"))

    values = pickle.load(open("easy_regime_test_results.p", "rb"))
    # values.sort(key=lambda x: 1000 * x[0] + x[1])

    plt.close()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for g0, o0, gdiffs, odiffs in values:
        gdiff = np.mean(gdiffs)
        odiff = np.mean(odiffs)
        arrow_scale = 0.1
        plt.arrow(o0, g0, arrow_scale * odiff, arrow_scale * gdiff, width=1e-3, head_length=10e-3, head_width=15e-3,
                  color="black", **{"overhang": 0.5}, alpha=0.75, length_includes_head=True)

    # Plot ground truth parameter estimates
    ground_truth_gamma, ground_truth_omega = gamma_omega_estimate(G_intralayer, G_interlayer,
                                                                  layer_vec, ground_truth_comms)
    plt.scatter([ground_truth_omega], [ground_truth_gamma], s=50, color='blue', edgecolor='black', linewidths=1,
                marker='o')

    plt.title(r"Synthetic Network ($\omega$, $\gamma$) Estimates from Louvain", fontsize=14)
    plt.xlabel(r"$\omega$", fontsize=14)
    plt.ylabel(r"$\gamma$", fontsize=14)
    plt.xlim([ITERATION_OMEGA_START, ITERATION_OMEGA_END])
    plt.ylim([ITERATION_GAMMA_START, ITERATION_GAMMA_END])
    plt.savefig("synthetic_network_pamfil_iteration.pdf")


def plot_easy_regime_domains():
    # Prune partitions with CHAMP and get parameter estimates
    domains_with_estimates = pickle.load(open("synthetic_champ_domains_with_estimates.p", "rb"))

    # Plot domains of optimality with parameter estimates
    plt.close()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_2d_domains_with_estimates(domains_with_estimates, [ITERATION_OMEGA_START, ITERATION_OMEGA_END],
                                   [ITERATION_GAMMA_START, ITERATION_GAMMA_END], flip_axes=True)
    plt.title(r"Synthetic Network Domains and ($\omega$, $\gamma$) Estimates", fontsize=14)
    plt.xlabel(r"$\omega$", fontsize=14)
    plt.ylabel(r"$\gamma$", fontsize=14)
    plt.savefig("synthetic_network_with_gamma_omega_estimates.pdf")


def plot_easy_regime_domains_with_ami_and_Ks():
    # Import graph and CHAMP's pruned partitions with estimates
    G_intralayer, G_interlayer, _, ground_truth = pickle.load(open("easy_regime_multilayer.p", "rb"))
    domains_with_estimates = pickle.load(open("synthetic_champ_domains_with_estimates.p", "rb"))

    plt.close()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_2d_domains_with_ami(domains_with_estimates, ground_truth, [ITERATION_OMEGA_START, ITERATION_OMEGA_END],
                             [ITERATION_GAMMA_START, ITERATION_GAMMA_END], flip_axes=True)
    plt.title("AMI of Domains with Ground Truth", fontsize=14)
    plt.xlabel(r"$\omega$", fontsize=14)
    plt.ylabel(r"$\gamma$", fontsize=14)
    plt.savefig("synthetic_network_domains_with_ground_truth_ami.pdf")

    plt.close()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_2d_domains_with_num_communities(domains_with_estimates, [ITERATION_OMEGA_START, ITERATION_OMEGA_END],
                                         [ITERATION_GAMMA_START, ITERATION_GAMMA_END], flip_axes=True)
    plt.title("Domains with Number of Communities", fontsize=14)
    plt.xlabel(r"$\omega$", fontsize=14)
    plt.ylabel(r"$\gamma$", fontsize=14)
    plt.savefig("synthetic_network_domains_with_num_communities.pdf")


def generate_domains_with_estimates(restrict_communities=None):
    # Import graph and partitions
    G_intralayer, G_interlayer, layer_vec, ground_truth = pickle.load(open("easy_regime_multilayer.p", "rb"))
    all_parts = pickle.load(open("easy_regime_50K_louvain.p", "rb"))

    if restrict_communities:
        all_parts = {p for p in all_parts if num_communities(p) == restrict_communities}
    else:
        all_parts = {p for p in all_parts}

    # Prune partitions with CHAMP
    print("Starting CHAMP...")
    start = time()
    domains = CHAMP_3D(G_intralayer, G_interlayer, layer_vec, all_parts, CHAMP_GAMMA_START, CHAMP_GAMMA_END,
                       CHAMP_OMEGA_START, CHAMP_OMEGA_END)
    print(f"Took {time() - start:.2f} s")

    # Get parameter estimates
    print("Starting parameter estimation...")
    start = time()
    domains_with_estimates = domains_to_gamma_omega_estimates(G_intralayer, G_interlayer, layer_vec, domains)
    print(f"Took {time() - start:.2f} s")

    return domains_with_estimates


def plot_easy_regime_domains_restricted_communities():
    domains_with_estimates = pickle.load(open("synthetic_champ_2-community_domains_with_estimates.p", "rb"))

    # Plot domains of optimality with parameter estimates
    plt.close()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_2d_domains_with_estimates(domains_with_estimates, [ITERATION_OMEGA_START, ITERATION_OMEGA_END],
                                   [ITERATION_GAMMA_START, ITERATION_GAMMA_END], flip_axes=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title(r"Synthetic Network Domains and ($\omega$, $\gamma$) Estimates, $K=2$", fontsize=14)
    plt.xlabel(r"$\omega$", fontsize=14)
    plt.ylabel(r"$\gamma$", fontsize=14)
    plt.savefig("synthetic_network_with_2-community_gamma_omega_estimates.pdf")


# def print_num_stable_stables_restricted_communities():
#     # Import graph and CHAMP's pruned partitions with estimates
#     domains_with_estimates = pickle.load(open("synthetic_champ_2-community_domains_with_estimates.p", "rb"))
#
#     print(len(domains_with_estimates))
#     print(len(gamma_omega_estimates_to_stable_partitions(domains_with_estimates)))


if __name__ == "__main__":
    if not os.path.exists("easy_regime_multilayer.p"):
        print("Generating synthetic network...")
        G_intralayer, G_interlayer, layer_vec, comm_vec = generate_synthetic_network()
        pickle.dump((G_intralayer, G_interlayer, layer_vec, comm_vec), open("easy_regime_multilayer.p", "wb"))

    if not os.path.exists("easy_regime_test_results.p"):
        print("Running pamfil iteration...")
        values = run_pamfil_iteration()
        pickle.dump(values, open("easy_regime_test_results.p", "wb"))

    if not os.path.exists("easy_regime_50K_louvain.p"):
        print("Running easy regime Louvain...")
        all_parts = run_easy_regime_louvain()
        pickle.dump(all_parts, open("easy_regime_50K_louvain.p", "wb"))

    if not os.path.exists("synthetic_champ_domains_with_estimates.p"):
        print("Generating CHAMP domains with estimates...")
        domains_with_estimates = generate_domains_with_estimates()
        pickle.dump(domains_with_estimates, open("synthetic_champ_domains_with_estimates.p", "wb"))

    if not os.path.exists("synthetic_champ_2-community_domains_with_estimates.p"):
        print("Generating CHAMP domains with estimates when K=2...")
        domains_with_estimates = generate_domains_with_estimates(restrict_communities=2)
        pickle.dump(domains_with_estimates, open("synthetic_champ_2-community_domains_with_estimates.p", "wb"))

    plot_easy_regime_iteration()
    plot_easy_regime_domains()
    plot_easy_regime_domains_with_ami_and_Ks()
    plot_easy_regime_domains_restricted_communities()
