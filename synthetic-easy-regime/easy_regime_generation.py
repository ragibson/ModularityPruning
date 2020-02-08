# Generates figures 5.3, 5.4, 5.5, and 5.6

from random import random, randint
from collections import Counter
import igraph as ig
from champ.parameter_estimation import iterative_multilayer_resolution_parameter_estimation
from utilities import Progress, repeated_parallel_louvain_from_gammas_omegas
import os
from utilities import plot_2d_domains_with_num_communities
from time import time
import pickle
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
from utilities import CHAMP_3D, domains_to_gamma_omega_estimates, plot_2d_domains_with_estimates
from utilities import ami, num_communities, gamma_omega_estimates_to_stable_partitions


def generate_synthetic_network():
    # TODO: this needs some checks to ensure later steps don't fail -- need to investigate more
    eta = 0.7  # copying probability
    epsilon = 0.4  # p_in/p_out ratio
    p_in = 20 / 75
    p_out = epsilon * p_in

    n_per_layer = 150
    num_layers = 15
    K = 2

    comm_per_layer = [[0] * n_per_layer for _ in range(num_layers)]
    comm_per_layer[0] = [i // (n_per_layer // K) for i in range(n_per_layer)]

    layer_vec = [i // n_per_layer for i in range(n_per_layer * num_layers)]

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
    interlayer_edges = [(n_per_layer * l + v, n_per_layer * l + v + n_per_layer)
                        for l in range(num_layers - 1)
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
    pickle.dump((G_intralayer, G_interlayer, comm_vec), open("easy_regime_multilayer.p", "wb"))


def run_pamfil_iteration():
    G_intralayer, G_interlayer, _ = pickle.load(open("easy_regime_multilayer.p", "rb"))

    n_per_layer = 150
    num_layers = 15
    layer_vec = [i // n_per_layer for i in range(n_per_layer * num_layers)]

    def one_step(gamma, omega):
        try:
            g_new, o_new, _ = iterative_multilayer_resolution_parameter_estimation(G_intralayer, G_interlayer,
                                                                                   layer_vec,
                                                                                   gamma=gamma, omega=omega, max_iter=1)
            return g_new, o_new
        except ValueError:
            return None, None

    values = []
    all_g0s = np.linspace(0.6, 1.45, 15)
    all_o0s = np.linspace(0.4, 1.6, 15)
    progress = Progress(len(all_g0s) * len(all_o0s))
    for g0 in all_g0s:
        for o0 in all_o0s:
            gdiffs = []
            odiffs = []
            REPEAT = 5
            for repeat in range(REPEAT):
                # print("\rgamma={:.3f}, omega={:.3f} Progress: [".format(g0, o0) +
                #       "#" * repeat + "-" * (REPEAT - repeat - 1) + "] ", end='', flush=True)
                g1, o1 = one_step(g0, o0)
                if g1 is not None and o1 is not None:
                    gdiffs.append(g1 - g0)
                    odiffs.append(o1 - o0)
            else:
                # if len(odiffs) > 0:
                #     print("has movement ({:.3f},{:.3f}) with count {}"
                #           "".format(np.mean(gdiffs), np.mean(odiffs), len(gdiffs)))
                # else:
                #     print()
                values.append((g0, o0, gdiffs, odiffs))

            progress.increment()

    progress.done()
    pickle.dump(values, open("easy_regime_test_results.p", "wb"))
    print(values)


def run_easy_regime_louvain():
    G_intralayer, G_interlayer, ground_truth_comms = pickle.load(open("easy_regime_multilayer.p", "rb"))
    n_per_layer = 150
    num_layers = 15
    layer_vec = [i // n_per_layer for i in range(n_per_layer * num_layers)]

    all_g0s = np.linspace(0.0, 2.0, 225)
    all_o0s = np.linspace(0.0, 2.0, 225)
    all_parts = repeated_parallel_louvain_from_gammas_omegas(G_intralayer, G_interlayer, layer_vec, all_g0s, all_o0s)
    pickle.dump(all_parts, open("easy_regime_50K_louvain.p", "wb"))


def plot_easy_regime_iteration():
    values = pickle.load(open("easy_regime_test_results.p", "rb"))
    values.sort(key=lambda x: 1000 * x[0] + x[1])

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots()

    iteration = 0
    for g0, o0, gdiffs, odiffs in values:
        gdiff = np.mean(gdiffs)
        odiff = np.mean(odiffs)
        SCALE = 0.1
        plt.arrow(o0, g0, SCALE * odiff, SCALE * gdiff, width=1e-3, head_length=10e-3, head_width=15e-3,
                  color="black", **{"overhang": 0.5}, alpha=0.75, length_includes_head=True)

    # TODO: this shouldn't be hardcoded here (this must assume the specific graph we were using before?)
    ground_truth_gamma = 0.9357510425040243
    ground_truth_omega = 0.984333998485813
    plt.scatter([ground_truth_omega], [ground_truth_gamma], s=50, color='blue', edgecolor='black', linewidths=1,
                marker='o')

    plt.title("Synthetic Network (Gamma, Omega) Estimates from Louvain", fontsize=14)
    plt.xlabel(r"$\omega$", fontsize=14)
    plt.ylabel(r"$\gamma$", fontsize=14)
    plt.xlim([0.4, 1.6])
    plt.ylim([0.6, 1.4])
    plt.savefig("synthetic_network_pamfil_iteration.pdf")


def plot_easy_regime_domains():
    from utilities import CHAMP_3D, domains_to_gamma_omega_estimates, plot_2d_domains_with_estimates
    import pickle
    import matplotlib.pyplot as plt
    from time import time

    # Import graph and partitions
    G_intralayer, G_interlayer, ground_truth = pickle.load(open("easy_regime_multilayer.p", "rb"))

    n_per_layer = 150
    num_layers = 15
    layer_vec = [i // n_per_layer for i in range(n_per_layer * num_layers)]
    gamma_start, gamma_end = 0.0, 2.0
    omega_start, omega_end = 0.0, 2.0
    all_parts = pickle.load(open("easy_regime_50K_louvain.p", "rb"))

    # Prune partitions with CHAMP
    print("Starting CHAMP...")
    start = time()
    domains = CHAMP_3D(G_intralayer, G_interlayer, layer_vec, all_parts, gamma_start, gamma_end, omega_start, omega_end)
    print("Took {:.2f} s".format(time() - start))

    # Get parameter estimates
    print("Starting parameter estimation...")
    start = time()
    domains_with_estimates = domains_to_gamma_omega_estimates(G_intralayer, G_interlayer, layer_vec, domains)
    print("Took {:.2f} s".format(time() - start))

    pickle.dump(domains_with_estimates, open("synthetic_champ_domains_with_estimates.p", "wb"))
    # domains_with_estimates = pickle.load(open("synthetic_champ_domains_with_estimates.p", "rb"))

    # Plot domains of optimality with parameter estimates
    for repeat in range(5):
        plt.close()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plot_2d_domains_with_estimates(domains_with_estimates, [0.4, 1.6], [0.6, 1.45], flip_axes=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.title("Synthetic Network Domains and (Gamma, Omega) Estimates", fontsize=14)
        plt.xlabel(r"$\omega$", fontsize=14)
        plt.ylabel(r"$\gamma$", fontsize=14)
        plt.savefig("synthetic_network_with_gamma_omega_estimates{}.pdf".format(repeat))


def plot_easy_regime_domains_with_ami_and_Ks():
    # Import graph and CHAMP's pruned partitions with estimates
    G_intralayer, G_interlayer, ground_truth = pickle.load(open("easy_regime_multilayer.p", "rb"))
    n_per_layer = 150
    num_layers = 15
    layer_vec = [i // n_per_layer for i in range(n_per_layer * num_layers)]
    domains_with_estimates = pickle.load(open("synthetic_champ_domains_with_estimates.p", "rb"))

    def plot_2d_domains_with_ami(domains_with_estimates, xlim, ylim, flip_axes=False):
        fig, ax = plt.subplots()
        patches = []

        for polyverts, membership, gamma_est, omega_est in domains_with_estimates:
            if flip_axes:
                polyverts = [(x[1], x[0]) for x in polyverts]

            polygon = Polygon(polyverts, True)
            patches.append(polygon)

        cm = matplotlib.cm.copper
        amis = np.array([ami(membership, ground_truth) for _, membership, _, _ in domains_with_estimates] + [1.0])

        p = PatchCollection(patches, cmap=cm, alpha=1.0, edgecolors='black', linewidths=2)
        p.set_array(amis)
        ax.add_collection(p)

        cbar = plt.colorbar(p)
        cbar.set_label('AMI', fontsize=14, labelpad=15)

        plt.xlim(xlim)
        plt.ylim(ylim)

    plt.close()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_2d_domains_with_ami(domains_with_estimates, [0.4, 1.6], [0.6, 1.45], flip_axes=True)
    plt.title("AMI of Domains with Ground Truth", fontsize=14)
    plt.xlabel(r"$\omega$", fontsize=14)
    plt.ylabel(r"$\gamma$", fontsize=14)
    plt.savefig("synthetic_network_domains_with_ground_truth_ami.pdf")

    plt.close()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_2d_domains_with_num_communities(domains_with_estimates, [0.4, 1.6], [0.6, 1.45], flip_axes=True)
    plt.title("Domains with Number of Communities", fontsize=14)
    plt.xlabel(r"$\omega$", fontsize=14)
    plt.ylabel(r"$\gamma$", fontsize=14)
    plt.savefig("synthetic_network_domains_with_num_communities.pdf")


def generate_domains_with_estimates():
    # Import graph and partitions
    G_intralayer, G_interlayer, ground_truth = pickle.load(open("easy_regime_multilayer.p", "rb"))
    n_per_layer = 150
    num_layers = 15
    layer_vec = [i // n_per_layer for i in range(n_per_layer * num_layers)]
    gamma_start, gamma_end = 0.0, 2.0
    omega_start, omega_end = 0.0, 2.0
    all_parts = pickle.load(open("easy_regime_50K_louvain.p", "rb"))
    all_parts = {p for p in all_parts if num_communities(p) == 2}
    print(len(all_parts))

    # Prune partitions with CHAMP
    print("Starting CHAMP...")
    start = time()
    domains = CHAMP_3D(G_intralayer, G_interlayer, layer_vec, all_parts, gamma_start, gamma_end, omega_start, omega_end)
    print("Took {:.2f} s".format(time() - start))

    # Get parameter estimates
    print("Starting parameter estimation...")
    start = time()
    domains_with_estimates = domains_to_gamma_omega_estimates(G_intralayer, G_interlayer, layer_vec, domains)
    print("Took {:.2f} s".format(time() - start))
    pickle.dump(domains_with_estimates, open("synthetic_champ_2-community_domains_with_estimates.p", "wb"))


def plot_easy_regime_domains_restricted_communities():
    domains_with_estimates = pickle.load(open("synthetic_champ_2-community_domains_with_estimates.p", "rb"))

    # Plot domains of optimality with parameter estimates
    for repeat in range(5):
        plt.close()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plot_2d_domains_with_estimates(domains_with_estimates, [0.4, 1.6], [0.6, 1.45], flip_axes=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.title("Synthetic Network Domains and (Gamma, Omega) Estimates, $K=2$", fontsize=14)
        plt.xlabel(r"$\omega$", fontsize=14)
        plt.ylabel(r"$\gamma$", fontsize=14)
        plt.savefig("synthetic_network_with_2-community_gamma_omega_estimates{}.pdf".format(repeat))


def num_stable_stables_restricted_communities():
    # Import graph and CHAMP's pruned partitions with estimates
    G_intralayer, G_interlayer, ground_truth = pickle.load(open("easy_regime_multilayer.p", "rb"))
    n_per_layer = 150
    num_layers = 15
    layer_vec = [i // n_per_layer for i in range(n_per_layer * num_layers)]
    domains_with_estimates = pickle.load(open("synthetic_champ_2-community_domains_with_estimates.p", "rb"))

    print(len(domains_with_estimates))
    print(len(gamma_omega_estimates_to_stable_partitions(domains_with_estimates)))


if not os.path.exists("easy_regime_multilayer.p"):
    print("Generating synthetic network...")
    generate_synthetic_network()

if not os.path.exists("easy_regime_test_results.p"):
    print("Running pamfil iteration...")
    run_pamfil_iteration()

if not os.path.exists("easy_regime_50K_louvain.p"):
    print("Running easy regime Louvain...")
    run_easy_regime_louvain()

if not os.path.exists("synthetic_champ_2-community_domains_with_estimates.p"):
    print("Generating CHAMP domains with estimates...")
    generate_domains_with_estimates()

plot_easy_regime_iteration()
plot_easy_regime_domains()
plot_easy_regime_domains_with_ami_and_Ks()
plot_easy_regime_domains_restricted_communities()