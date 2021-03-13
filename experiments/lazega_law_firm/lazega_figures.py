# Generates figures 5.7 through 5.10

import os
from modularitypruning.champ_utilities import CHAMP_3D
from modularitypruning.louvain_utilities import repeated_parallel_louvain_from_gammas_omegas
from modularitypruning.parameter_estimation_utilities import domains_to_gamma_omega_estimates, \
    gamma_omega_estimates_to_stable_partitions
from modularitypruning.partition_utilities import nmi, num_communities
from modularitypruning.plotting import plot_2d_domains_with_estimates, plot_2d_domains_with_num_communities, \
    plot_multiplex_community
import igraph as ig
import numpy as np
import pickle
import matplotlib.pyplot as plt

CHAMP_GAMMA_END = 2.0
CHAMP_OMEGA_END = 3.0


def adjacency_from_dat(lines):
    adjacency = []
    for line in lines:
        line = line.strip("\r\n")
        if len(line) > 0:
            adjacency.append([int(x) for x in line.split()])
    return adjacency


def generate_lazega_igraph():
    missing_files = []
    for expected_file in ["ELadv.dat", "ELfriend.dat", "ELwork.dat", "ELattr.dat"]:
        if not os.path.exists(expected_file):
            missing_files.append(expected_file)

    if missing_files:
        raise FileNotFoundError(f"Missing Lazega Law Firm data files. Expected to find {missing_files}, "
                                "but these file(s) do not exist. Download these from "
                                "https://www.stats.ox.ac.uk/~snijders/siena/Lazega_lawyers_data.htm")

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


def run_lazega_louvain():
    G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()
    gammas = np.linspace(0.0, CHAMP_GAMMA_END, 2000)
    omegas = np.linspace(0.0, CHAMP_OMEGA_END, 500)
    assert len(gammas) * len(omegas) == 10 ** 6

    parts = repeated_parallel_louvain_from_gammas_omegas(G_intralayer, G_interlayer, layer_vec, gammas, omegas)

    pickle.dump(parts, open("lazega_1M_louvain.p", "wb"))


def run_champ_on_lazega_partitions():
    G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()
    layer_vec = np.array(layer_vec)

    all_parts = pickle.load(open("lazega_1M_louvain.p", "rb"))
    domains = CHAMP_3D(G_intralayer, G_interlayer, layer_vec, all_parts, 0.0, CHAMP_GAMMA_END, 0.0, CHAMP_OMEGA_END)

    pickle.dump(domains, open("lazega_CHAMP.p", "wb"))


def run_champ_on_lazega_partitions_restricted_K(K):
    G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()
    layer_vec = np.array(layer_vec)

    all_parts = pickle.load(open("lazega_1M_louvain.p", "rb"))
    all_parts = {p for p in all_parts if num_communities(p) == K}
    domains = CHAMP_3D(G_intralayer, G_interlayer, layer_vec, all_parts, 0.0, CHAMP_GAMMA_END, 0.0, CHAMP_OMEGA_END)

    pickle.dump(domains, open(f"lazega_CHAMP{K}.p", "wb"))


def plot_figure1():
    """Generates figure 5.8"""
    G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()
    domains = pickle.load(open("lazega_CHAMP.p", "rb"))
    domains_with_estimates = domains_to_gamma_omega_estimates(G_intralayer, G_interlayer, layer_vec, domains,
                                                              model='multiplex')

    # Truncate infinite omega solutions to our maximum omega
    domains_with_estimates = [(polyverts, membership, g_est, min(o_est, CHAMP_OMEGA_END - 1e-3))
                              for polyverts, membership, g_est, o_est in domains_with_estimates
                              if g_est is not None]

    stable_parts = gamma_omega_estimates_to_stable_partitions(domains_with_estimates)

    xlims = [0.4, 3.0]
    ylims = [0.8, 1.275]

    # Formatting:
    #   title fontsize=16
    #   axis label fontsize=20
    #   tick labelsize=12
    #   legend fontsize=14
    #   tight layout
    plt.close()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_2d_domains_with_estimates(domains_with_estimates, xlims, ylims, flip_axes=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title(r"Lazega Law Firm Domains and ($\omega$, $\gamma$) Estimates", fontsize=16)
    plt.xlabel(r"$\omega$", fontsize=20)
    plt.ylabel(r"$\gamma$", fontsize=20)
    plt.gca().tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig("lazega_domains_and_estimates.pdf")

    # Formatting:
    #   title fontsize=16
    #   axis label fontsize=20
    #   tick labelsize=12
    #   legend fontsize=14
    #   tight layout
    plt.close()
    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_2d_domains_with_estimates(stable_parts, xlims, ylims, plot_estimate_points=False, flip_axes=True)
    plt.title("Lazega Law Firm Stable Partitions", fontsize=16)
    plt.xlabel(r"$\omega$", fontsize=20)
    plt.ylabel(r"$\gamma$", fontsize=20)
    plt.gca().tick_params(axis='both', labelsize=12)

    for K, marker, color in reversed([(2, "o", "C0"), (3, "^", "C1"), (4, "s", "C2")]):
        domains = pickle.load(open(f"lazega_CHAMP{K}.p", "rb"))
        domains_with_estimates = domains_to_gamma_omega_estimates(G_intralayer, G_interlayer, layer_vec, domains,
                                                                  model='multiplex')

        # Truncate infinite omega solutions to our maximum omega
        domains_with_estimates = [(polyverts, membership, g_est, min(o_est, CHAMP_OMEGA_END - 1e-2))
                                  for polyverts, membership, g_est, o_est in domains_with_estimates
                                  if g_est is not None]
        stable_parts = gamma_omega_estimates_to_stable_partitions(domains_with_estimates)

        gamma_estimates = [g_est for polyverts, membership, g_est, o_est in stable_parts]
        omega_estimates = [o_est for polyverts, membership, g_est, o_est in stable_parts]
        plt.scatter(omega_estimates, gamma_estimates, marker=marker, color=color, s=60,
                    label=f"stable, $K={K}$", linewidths=1, edgecolors="black")

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.gca().legend(handles, labels, fontsize=14)
    plt.tight_layout()
    plt.savefig("lazega_stable_partitions.pdf")


def plot_figure1_restricted_K():
    """Generates figure 5.9"""
    G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()

    all_stable = []
    for K in range(2, 5):
        domains = pickle.load(open(f"lazega_CHAMP{K}.p", "rb"))
        domains_with_estimates = domains_to_gamma_omega_estimates(G_intralayer, G_interlayer, layer_vec, domains,
                                                                  model='multiplex')

        # Truncate infinite omega solutions to our maximum omega
        domains_with_estimates = [(polyverts, membership, g_est, min(o_est, CHAMP_OMEGA_END - 1e-3))
                                  for polyverts, membership, g_est, o_est in domains_with_estimates
                                  if g_est is not None]

        stable_parts = gamma_omega_estimates_to_stable_partitions(domains_with_estimates)
        all_stable.extend([part for _, part, _, _ in stable_parts])

        xlims = [0.4, 3.0]
        ylims = [0.4, 1.4]

        plt.close()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plot_2d_domains_with_estimates(stable_parts, xlims, ylims, flip_axes=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.title(rf"Lazega Law Firm Stable Partitions, $K={K}$", fontsize=14)
        plt.xlabel(r"$\omega$", fontsize=14)
        plt.ylabel(r"$\gamma$", fontsize=14)
        plt.savefig(f"lazega_stable_partitions_K={K}.pdf")


def plot_figure2():
    """Generates figure 5.7"""
    G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()
    domains = pickle.load(open("lazega_CHAMP.p", "rb"))
    domains_with_estimates = domains_to_gamma_omega_estimates(G_intralayer, G_interlayer, layer_vec, domains,
                                                              model='multiplex')

    # Truncate infinite omega solutions to our maximum omega
    domains_with_estimates = [(polyverts, membership, g_est, min(o_est, CHAMP_OMEGA_END - 1e-3))
                              for polyverts, membership, g_est, o_est in domains_with_estimates
                              if g_est is not None]

    xlims = [0.4, 3.0]
    ylims = [0.8, 1.275]

    # Formatting:
    #   title fontsize=16
    #   axis label fontsize=20
    #   tick labelsize=12
    #   legend fontsize=14
    #   tight layout
    plt.close()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_2d_domains_with_num_communities(domains_with_estimates, xlims, ylims, flip_axes=True,
                                         K_max=5,  # we have a tiny sliver of K=6 which messes up the colorbar
                                         tick_step=1)
    plt.title("Lazega Law Firm Domains with Number of Communities", fontsize=16)
    plt.xlabel(r"$\omega$", fontsize=20)
    plt.ylabel(r"$\gamma$", fontsize=20)
    plt.gca().tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.gcf().set_size_inches(8.05, 4.8)  # longer width for cbar
    plt.savefig("lazega_domains_with_num_communities.pdf")


def plot_figure3():
    """Generates figure 5.10"""
    G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()
    N = 71
    T = 3

    all_stable_parts = []
    for K in range(2, 5):
        domains = pickle.load(open(f"lazega_CHAMP{K}.p", "rb"))

        # Truncate infinite omega solutions to our maximum omega
        domains_with_estimates = domains_to_gamma_omega_estimates(G_intralayer, G_interlayer, layer_vec, domains,
                                                                  model='multiplex')
        domains_with_estimates = [(polyverts, membership, g_est, min(o_est, CHAMP_OMEGA_END - 1e-3))
                                  for polyverts, membership, g_est, o_est in domains_with_estimates
                                  if g_est is not None]

        stable_parts = gamma_omega_estimates_to_stable_partitions(domains_with_estimates)

        all_stable_parts.extend([membership for _, membership, _, _ in stable_parts])

    # this sorting seems to keep the number of plot breaks low between all the common stable partitions
    sort = np.array([46, 21, 64, 55, 48, 54, 68, 40, 70, 56, 65, 66, 53, 51, 67, 38, 42,
                     39, 37, 26, 20, 10, 12, 22, 25, 0, 23, 19, 7, 35, 61, 69, 63, 41,
                     28, 16, 15, 8, 11, 9, 14, 1, 3, 36, 18, 52, 43, 47, 33, 44, 60,
                     59, 45, 31, 34, 27, 62, 5, 49, 30, 58, 50, 17, 57, 4, 32, 6, 2,
                     24, 13, 29])

    for i, membership in enumerate(all_stable_parts):
        plt.close()
        K = num_communities(membership)

        membership = np.array(membership)
        m1, m2, m3 = (membership[i * N:(i + 1) * N] for i in range(T))
        if K == 2:
            m1, m2, m3 = m1[sort], m2[sort], m3[sort]
        elif K == 3:
            m1, m2, m3 = m1[sort], m2[sort], m3[sort]
        elif K == 4:
            m1, m2, m3 = m1[sort], m2[sort], m3[sort]
        membership = np.concatenate((m1, m3, m2))  # Concatenate in order advice, work, friend

        plt.close()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ax = plot_multiplex_community(np.array(membership), np.array(layer_vec))
        ax.set_xticks(np.linspace(0, T, 2 * T + 1))
        ax.set_xticklabels(["", "Advice", "", "Coworker", "", "Friend", ""], fontsize=14)
        plt.title(f"Multiplex Communities in Stable Partition {i + 1}", fontsize=14)
        plt.ylabel("Node ID", fontsize=14)
        plt.savefig(f"lazega_stable_community{i}.pdf")


def print_metadata_nmis(memberships):
    """Prints the nmis between stable partitions and the various metadata labels"""

    lines = open("ELattr.dat", "r").readlines()
    seniority = []
    status = []
    gender = []
    office = []
    age = []
    practice = []
    law_school = []
    for line in lines:
        # Group seniority and age metadata into 5-year bins to match Pamfil et al.
        x = tuple(int(v) for v in line.strip().split())
        status.append(x[1])
        gender.append(x[2])
        office.append(x[3])
        seniority.append(x[4] // 5)
        age.append(x[5] // 5)
        practice.append(x[6])
        law_school.append(x[7])

    for i, memberships in enumerate(memberships):
        part = memberships
        print(f"=====Stable partition {i + 1} with K={num_communities(part)}=====")
        print(f"&{nmi(part, office * 3):.3f}")
        print(f"&{nmi(part, practice * 3):.3f}")
        print(f"&{nmi(part, age * 3):.3f}")
        print(f"&{nmi(part, seniority * 3):.3f}")
        print(f"&{nmi(part, status * 3):.3f}")
        print(f"&{nmi(part, gender * 3):.3f}")
        print(f"&{nmi(part, law_school * 3):.3f}")


if __name__ == "__main__":
    if not os.path.exists("lazega_1M_louvain.p"):
        print("Running Louvain...")
        run_lazega_louvain()

    if not os.path.exists("lazega_CHAMP.p"):
        print("Running CHAMP...")
        run_champ_on_lazega_partitions()

    for k in range(2, 8):
        if not os.path.exists(f"lazega_CHAMP{k}.p"):
            print(f"Running CHAMP when K={k}...")
            run_champ_on_lazega_partitions_restricted_K(k)

    plot_figure1()
    plot_figure1_restricted_K()
    plot_figure2()
    plot_figure3()
