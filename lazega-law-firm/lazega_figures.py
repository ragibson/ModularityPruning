# Generates figures 5.7 through 5.10

import os
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from utilities import iterative_multilayer_resolution_parameter_estimation, \
    repeated_parallel_louvain_from_gammas_omegas, Progress, CHAMP_3D
from time import time
import igraph as ig
from utilities import domains_to_gamma_omega_estimates
from utilities import plot_2d_domains_with_estimates, plot_2d_domains_with_num_communities
from utilities import gamma_omega_estimates_to_stable_partitions, num_communities
import numpy as np
import pickle
import matplotlib.pyplot as plt
from utilities import plot_multiplex_community

CHAMP_GAMMA_END = 2.0
CHAMP_OMEGA_END = 3.0
xlims = [0.4, 3.0]
ylims = [0.8, 1.275]


def adjacency_from_dat(lines):
    adjacency = []
    for line in lines:
        line = line.strip("\r\n")
        if len(line) > 0:
            adjacency.append([int(x) for x in line.split()])
    return adjacency


def generate_lazega_igraph():
    advice_adj = adjacency_from_dat(open("ELadv.dat", "r").readlines())
    friend_adj = adjacency_from_dat(open("ELfriend.dat", "r").readlines())
    work_adj = adjacency_from_dat(open("ELwork.dat", "r").readlines())

    N, T = 71, 3
    assert N == len(advice_adj)
    assert N == len(friend_adj)
    assert N == len(work_adj)

    layer_vec = [i // N for i in range(N * T)]
    interlayer_edges = [(i, t * N + i) for i in range(N) for t in range(T)] + \
                       [(t * N + i, i) for i in range(N) for t in range(T)]
    intralayer_edges = []
    for t, adj in enumerate([advice_adj, friend_adj, work_adj]):
        for i in range(len(adj)):
            for j in range(len(adj[i])):
                supranode_i = N * t + i
                supranode_j = N * t + j
                if adj[i][j] == 1:
                    intralayer_edges.append((supranode_i, supranode_j))

    assert all(0 <= e[0] < N * T and 0 <= e[1] < N * T for e in interlayer_edges)
    assert all(0 <= e[0] < N * T and 0 <= e[1] < N * T for e in intralayer_edges)

    G_intralayer = ig.Graph(edges=intralayer_edges, directed=True)
    G_interlayer = ig.Graph(edges=interlayer_edges, directed=True)
    return G_intralayer, G_interlayer, layer_vec


def plot_parameter_estimation_iteration(G_intralayer, G_interlayer, layer_vec):
    # Note: this is not multithreaded and performance analyzes need to take this into account
    gfs, ofs = [], []
    progress = Progress(10 * 10)
    for g0 in np.linspace(0.5, 1.25, 10):
        for o0 in np.linspace(0.5, 1.25, 10):
            try:
                gamma, omega, part = iterative_multilayer_resolution_parameter_estimation(
                    G_intralayer, G_interlayer, layer_vec, gamma=g0, omega=o0, model='multiplex',
                    gamma_tol=1e-2, omega_tol=1e-2, max_iter=100, omega_max=1.25)

                gfs.append(gamma)
                ofs.append(omega)
                assert gamma is not None and omega is not None
            except ValueError:
                pass
            progress.increment()
    progress.done()

    plt.scatter(ofs, gfs)
    plt.xlabel("omega")
    plt.ylabel("gamma")
    plt.xlim([0.5, 1.4])
    plt.ylim([0.5, 1.25])
    plt.show()


def run_lazega_louvain():
    print("Running lazega louvain...")

    G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()
    gammas = np.linspace(0.5, 1.5, 2000)
    omegas = np.linspace(0.5, 3.0, 500)

    assert len(gammas) * len(omegas) == 10 ** 6

    parts = set()
    for i in range(0, len(gammas), len(gammas) // 10):
        current_parts = repeated_parallel_louvain_from_gammas_omegas(G_intralayer, G_interlayer, layer_vec,
                                                                     gammas[i:i + (len(gammas) // 10)], omegas)
        parts.update(current_parts)

    pickle.dump(parts, open("lazega_1M_louvain.p", "wb"))

    # for K in range(2, 5):
    #     layer_vec = np.array(layer_vec)
    #     all_parts = {p for p in parts if num_communities(p) == K}
    #     domains = CHAMP_3D(G_intralayer, G_interlayer, layer_vec, all_parts,
    #                        0.0, CHAMP_GAMMA_END, 0.0, CHAMP_OMEGA_END)
    #     domains_with_estimates = domains_to_gamma_omega_estimates(G_intralayer, G_interlayer, layer_vec, domains,
    #                                                               model='multiplex')
    #
    #     # Truncate infinite omega solutions to our maximum omega
    #     domains_with_estimates = [(polyverts, membership, g_est, min(o_est, CHAMP_OMEGA_END - 1e-3))
    #                               for polyverts, membership, g_est, o_est in domains_with_estimates
    #                               if g_est is not None]
    #     stable_parts = gamma_omega_estimates_to_stable_partitions(domains_with_estimates)
    #     print("K={}, {} stable".format(K, len(stable_parts)))
    #     for _, _, gamma_est, omega_est in stable_parts:
    #         print("{:.1f} {:.1f}".format(omega_est, gamma_est))


def run_champ_on_lazega_partitions():
    print("Running lazega CHAMP...")

    G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()
    layer_vec = np.array(layer_vec)

    all_parts = pickle.load(open("lazega_1M_louvain.p", "rb"))
    # print("Starting CHAMP...")
    start = time()
    domains = CHAMP_3D(G_intralayer, G_interlayer, layer_vec, all_parts, 0.0, CHAMP_GAMMA_END, 0.0, CHAMP_OMEGA_END)
    print("CHAMP took {:.2f} s".format(time() - start))

    pickle.dump(domains, open("lazega_CHAMP.p", "wb"))


def run_champ_on_lazega_partitions_restricted_K(K):
    print(f"Running lazega CHAMP restricted to {K} communities...")

    G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()
    layer_vec = np.array(layer_vec)

    all_parts = pickle.load(open("lazega_1M_louvain.p", "rb"))
    all_parts = {p for p in all_parts if num_communities(p) == K}

    # print("Starting CHAMP...")
    start = time()
    domains = CHAMP_3D(G_intralayer, G_interlayer, layer_vec, all_parts, 0.0, CHAMP_GAMMA_END, 0.0, CHAMP_OMEGA_END)
    print("CHAMP took {:.2f} s".format(time() - start))

    pickle.dump(domains, open("lazega_CHAMP{}.p".format(K), "wb"))


def plot_figure1():
    G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()
    domains = pickle.load(open("lazega_CHAMP.p", "rb"))
    domains_with_estimates = domains_to_gamma_omega_estimates(G_intralayer, G_interlayer, layer_vec, domains,
                                                              model='multiplex')

    # Truncate infinite omega solutions to our maximum omega
    domains_with_estimates = [(polyverts, membership, g_est, min(o_est, CHAMP_OMEGA_END - 1e-3))
                              for polyverts, membership, g_est, o_est in domains_with_estimates
                              if g_est is not None]

    stable_parts = gamma_omega_estimates_to_stable_partitions(domains_with_estimates)

    for repeat in range(1):
        plt.close()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plot_2d_domains_with_estimates(domains_with_estimates, xlims, ylims, flip_axes=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.title("Lazega Law Firm Domains and (Gamma, Omega) Estimates", fontsize=14)
        plt.xlabel(r"$\omega$", fontsize=14)
        plt.ylabel(r"$\gamma$", fontsize=14)
        plt.savefig(f"lazega_domains_and_estimates{repeat}.pdf")

    for repeat in range(1):
        plt.close()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plot_2d_domains_with_estimates(stable_parts, xlims, ylims, flip_axes=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.title("Lazega Law Firm Stable Partitions", fontsize=14)
        plt.xlabel(r"$\omega$", fontsize=14)
        plt.ylabel(r"$\gamma$", fontsize=14)
        plt.savefig(f"lazega_stable_partitions{repeat}.pdf")


def plot_figure2():
    G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()
    domains = pickle.load(open("lazega_CHAMP.p", "rb"))
    domains_with_estimates = domains_to_gamma_omega_estimates(G_intralayer, G_interlayer, layer_vec, domains,
                                                              model='multiplex')

    # Truncate infinite omega solutions to our maximum omega
    domains_with_estimates = [(polyverts, membership, g_est, min(o_est, 3.0 - 1e-3))
                              for polyverts, membership, g_est, o_est in domains_with_estimates
                              if g_est is not None]

    plt.close()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_2d_domains_with_num_communities(domains_with_estimates, xlims, ylims, flip_axes=True, tick_step=1)
    plt.title("Lazega Law Firm Domains with Number of Communities", fontsize=14)
    plt.xlabel(r"$\omega$", fontsize=14)
    plt.ylabel(r"$\gamma$", fontsize=14)
    plt.savefig("lazega_domains_with_num_communities.pdf")


# def print_partition_summary():
#     G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()
#
#     all_parts = pickle.load(open("lazega_1M_louvain.p", "rb"))
#     print("{} unique partitions".format(len(all_parts)))
#
#     for K in range(2, 8):
#         print("With K={}".format(K), len({p for p in all_parts if num_communities(p) == K}))
#
#     print("With K>=8", len({p for p in all_parts if num_communities(p) >= 8}))
#
#     domains = pickle.load(open("lazega_CHAMP.p", "rb"))
#     print("CHAMP's pruned subset has {} partitions:".format(len(domains)))
#
#     for K in range(2, 8):
#         print("With K={}".format(K), len({p for _, p in domains if num_communities(p) == K}))
#
#     print("With K>=8", len({p for _, p in domains if num_communities(p) >= 8}))
#
#     # Truncate infinite omega solutions to our maximum omega
#     domains_with_estimates = domains_to_gamma_omega_estimates(G_intralayer, G_interlayer, layer_vec, domains,
#                                                               model='multiplex')
#     domains_with_estimates = [(polyverts, membership, g_est, min(o_est, CHAMP_OMEGA_END - 1e-3))
#                               for polyverts, membership, g_est, o_est in domains_with_estimates
#                               if g_est is not None]
#     stable_parts = gamma_omega_estimates_to_stable_partitions(domains_with_estimates)
#
#     for K in range(2, 8):
#         print("With K={}".format(K), len({p for _, p, _, _ in stable_parts if num_communities(p) == K}))
#
#     print("With K>=8", len({p for _, p, _, _ in stable_parts if num_communities(p) >= 8}))


# def print_partition_summaries_with_fixed_K():
#     G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()
#
#     for K in range(2, 8):
#         domains = pickle.load(open("lazega_CHAMP{}.p".format(K), "rb"))
#         print("With K={}".format(K), len(domains))
#
#     print("Stable partitions:")
#     for K in range(2, 8):
#         domains = pickle.load(open("lazega_CHAMP{}.p".format(K), "rb"))
#
#         # Truncate infinite omega solutions to our maximum omega
#         domains_with_estimates = domains_to_gamma_omega_estimates(G_intralayer, G_interlayer, layer_vec, domains,
#                                                                   model='multiplex')
#         domains_with_estimates = [(polyverts, membership, g_est, min(o_est, 3.0 - 1e-3))
#                                   for polyverts, membership, g_est, o_est in domains_with_estimates
#                                   if g_est is not None]
#
#         stable_parts = gamma_omega_estimates_to_stable_partitions(domains_with_estimates)
#
#         assert all(num_communities(tuple(part)) == K for _, part, _, _ in stable_parts)
#
#         print("With K={}".format(K), len(stable_parts))
#
#         plot_2d_domains_with_estimates(stable_parts, [0, 3], [0, 2], flip_axes=True)
#         plt.show()


def plot_figure3():
    fig, axes = plt.subplots(1, 5, sharey=True)

    G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()
    N = 71
    T = 3

    all_stable_parts = []
    for K in range(2, 5):
        domains = pickle.load(open("lazega_CHAMP{}.p".format(K), "rb"))

        # Truncate infinite omega solutions to our maximum omega
        domains_with_estimates = domains_to_gamma_omega_estimates(G_intralayer, G_interlayer, layer_vec, domains,
                                                                  model='multiplex')
        domains_with_estimates = [(polyverts, membership, g_est, min(o_est, 3.0 - 1e-3))
                                  for polyverts, membership, g_est, o_est in domains_with_estimates
                                  if g_est is not None]

        stable_parts = gamma_omega_estimates_to_stable_partitions(domains_with_estimates)

        # for _, membership, g_est, o_est in stable_parts:
        #     print("{:.1f} {:.1f}".format(o_est, g_est))

        all_stable_parts.extend([membership for _, membership, _, _ in stable_parts])

    sort = np.array([46, 21, 64, 55, 48, 54, 68, 40, 70, 56, 65, 66, 53, 51, 67, 38, 42,
                     39, 37, 26, 20, 10, 12, 22, 25, 0, 23, 19, 7, 35, 61, 69, 63, 41,
                     28, 16, 15, 8, 11, 9, 14, 1, 3, 36, 18, 52, 43, 47, 33, 44, 60,
                     59, 45, 31, 34, 27, 62, 5, 49, 30, 58, 50, 17, 57, 4, 32, 6, 2,
                     24, 13, 29])

    for i, membership in enumerate(all_stable_parts):
        plt.close()
        K = num_communities(membership)
        # print("Stable partition {} has K={}".format(i, K))

        membership = np.array(membership)
        m1, m2, m3 = (membership[i * N:(i + 1) * N] for i in range(T))
        if K == 2:
            m1, m2, m3 = m1[sort], m2[sort], m3[sort]
        elif K == 3:
            m1, m2, m3 = m1[sort], m2[sort], m3[sort]
        elif K == 4:
            m1, m2, m3 = m1[sort], m2[sort], m3[sort]
        membership = np.concatenate((m1, m3, m2))  # Concatenate in order advice, work, friend

        if i == 6:
            # Recolor to make plot nicer
            for j in range(len(membership)):
                if membership[j] == 0:
                    membership[j] = 3
                elif membership[j] == 1:
                    membership[j] = 2
                elif membership[j] == 3:
                    membership[j] = 0
                elif membership[j] == 2:
                    membership[j] = 1

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ax = plot_multiplex_community(np.array(membership), np.array(layer_vec))
        ax.set_xticks(np.linspace(0, T, 2 * T + 1))
        ax.set_xticklabels(["", "Advice", "", "Coworker", "", "Friend"], fontsize=14)
        plt.title("Multiplex Communities in Stable Partition {}".format(i + 1), fontsize=14)
        plt.ylabel("Node ID", fontsize=14)
        plt.savefig("lazega_stable_community{}.pdf".format(i))


def plot_figure1_restricted_K():
    G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()

    xlims = [0.4, 3.0]
    ylims = [0.4, 1.4]

    all_stable = []
    for K in range(2, 5):
        domains = pickle.load(open("lazega_CHAMP{}.p".format(K), "rb"))
        domains_with_estimates = domains_to_gamma_omega_estimates(G_intralayer, G_interlayer, layer_vec, domains,
                                                                  model='multiplex')

        # Truncate infinite omega solutions to our maximum omega
        domains_with_estimates = [(polyverts, membership, g_est, min(o_est, CHAMP_OMEGA_END - 1e-3))
                                  for polyverts, membership, g_est, o_est in domains_with_estimates
                                  if g_est is not None]

        stable_parts = gamma_omega_estimates_to_stable_partitions(domains_with_estimates)
        all_stable.extend([part for _, part, _, _ in stable_parts])

        # for repeat in range(5):
        # plt.close()
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        # plot_2d_domains_with_estimates(domains_with_estimates, xlims, ylims, flip_axes=True)
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        # plt.title("Lazega Law Firm Domains and (Gamma, Omega) Estimates, $K={}$".format(K), fontsize=14)
        # plt.xlabel(r"$\omega$", fontsize=14)
        # plt.ylabel(r"$\gamma$", fontsize=14)
        # plt.show()

        plt.close()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plot_2d_domains_with_estimates(stable_parts, xlims, ylims, flip_axes=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.title("Lazega Law Firm Stable Partitions, $K={}$".format(K), fontsize=14)
        plt.xlabel(r"$\omega$", fontsize=14)
        plt.ylabel(r"$\gamma$", fontsize=14)
        plt.savefig("lazega_stable_partitions_K={}.pdf".format(K))

    pickle.dump(all_stable, open("lazega_stable_K_2-4.p", "wb"))


if not os.path.exists("lazega_1M_louvain.p"):
    run_lazega_louvain()

if not os.path.exists("lazega_CHAMP.p"):
    run_champ_on_lazega_partitions()

for k in range(2, 8):
    if not os.path.exists(f"lazega_CHAMP{k}.p"):
        run_champ_on_lazega_partitions_restricted_K(k)

plot_figure1()
plot_figure2()
plot_figure3()
plot_figure1_restricted_K()
