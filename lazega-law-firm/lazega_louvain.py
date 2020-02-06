import igraph as ig
import numpy as np
from utilities import iterative_multilayer_resolution_parameter_estimation, repeated_parallel_louvain_from_gammas_omegas
from utilities import Progress
import matplotlib.pyplot as plt
import pickle

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


G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()
# gammas = np.linspace(0.5, 1.5, 2000)
# omegas = np.linspace(0.5, 3.0, 500)
#
# assert len(gammas) * len(omegas) == 10 ** 6
#
# parts = set()
# for i in range(0, len(gammas), len(gammas) // 10):
#     current_parts = repeated_parallel_louvain_from_gammas_omegas(G_intralayer, G_interlayer, layer_vec,
#                                                                  gammas[i:i + (len(gammas) // 10)], omegas)
#     parts.update(current_parts)

from utilities import *

from time import time

start = time()
for g0 in np.linspace(0.75, 1.25, 10):
    for o0 in np.linspace(0.75, 1.25, 10):
        _ = iterative_multilayer_resolution_parameter_estimation(G_intralayer, G_interlayer, layer_vec, gamma=g0,
                                                                 omega=o0, gamma_tol=1e-3, omega_tol=1e-3,
                                                                 max_iter=25,
                                                                 model='multiplex')
print("{:.2f} s".format(time() - start))

gammas = np.linspace(0.5, 1.5, 45)
omegas = np.linspace(0.5, 3.0, 45)
start = time()
parts = [multilayer_louvain(G_intralayer, G_interlayer, layer_vec, gamma, omega) for gamma in gammas for omega in
         omegas]
print("{:.2f} s".format(time() - start))

# for K in range(2, 5):
#     layer_vec = np.array(layer_vec)
#     all_parts = {p for p in parts if num_communities(p) == K}
#     domains = CHAMP_3D(G_intralayer, G_interlayer, layer_vec, all_parts, 0.0, CHAMP_GAMMA_END, 0.0, CHAMP_OMEGA_END)
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

# pickle.dump(parts, open("lazega_1M_louvain.p", "wb"))
