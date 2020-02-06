# TODO: combine this with other lazega-law-firm scripts for figures 5.7 through 5.10

import igraph as ig
import pickle
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
from utilities import CHAMP_3D
from utilities import num_communities
from time import time

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


def plot_2d_domains_with_num_communities(domains_with_estimates, xlim, ylim, flip_axes=False):
    fig, ax = plt.subplots()
    patches = []
    cm = matplotlib.cm.viridis
    Ks = []

    for polyverts, membership, gamma_est, omega_est in domains_with_estimates:
        if flip_axes:
            polyverts = [(x[1], x[0]) for x in polyverts]

        if any(ylim[0] <= x[1] <= ylim[1] for x in polyverts):
            polygon = Polygon(polyverts, True)
            patches.append(polygon)
            Ks.append(num_communities(membership))

    p = PatchCollection(patches, cmap=cm, alpha=1.0, edgecolors='black', linewidths=2)
    p.set_array(np.array(Ks))
    ax.add_collection(p)

    cbar = plt.colorbar(p, ticks=range(2, max(Ks) + 1, 2))
    cbar.set_label("Number of Communities", fontsize=14, labelpad=15)

    plt.xlim(xlim)
    plt.ylim(ylim)


def run_champ_on_lazega_partitions():
    G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()
    layer_vec = np.array(layer_vec)

    all_parts = pickle.load(open("lazega_1M_louvain.p", "rb"))
    print("Starting CHAMP...")
    start = time()
    domains = CHAMP_3D(G_intralayer, G_interlayer, layer_vec, all_parts, 0.0, CHAMP_GAMMA_END, 0.0, CHAMP_OMEGA_END)
    print("CHAMP took {:.2f} s".format(time() - start))

    pickle.dump(domains, open("lazega_CHAMP.p", "wb"))


def run_champ_on_lazega_partitions_restricted_K(K):
    G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()
    layer_vec = np.array(layer_vec)

    all_parts = pickle.load(open("lazega_1M_louvain.p", "rb"))
    all_parts = {p for p in all_parts if num_communities(p) == K}

    print("Starting CHAMP...")
    start = time()
    domains = CHAMP_3D(G_intralayer, G_interlayer, layer_vec, all_parts, 0.0, CHAMP_GAMMA_END, 0.0, CHAMP_OMEGA_END)
    print("CHAMP took {:.2f} s".format(time() - start))

    pickle.dump(domains, open("lazega_CHAMP{}.p".format(K), "wb"))


run_champ_on_lazega_partitions_restricted_K(5)
run_champ_on_lazega_partitions_restricted_K(6)
run_champ_on_lazega_partitions_restricted_K(7)
