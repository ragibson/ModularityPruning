import champ
import igraph as ig
import louvain
from math import ceil, inf
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import numpy as np
import pickle
from time import time
import igraph as ig
from utilities import domains_to_gamma_omega_estimates
from utilities import plot_2d_domains_with_estimates
import pickle
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
from champ import create_coefarray_from_partitions
from champ import get_intersection
from utilities import CHAMP_3D, partition_coefficients_3D
from utilities import num_communities, plot_2d_domains
from time import time


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


def sorted_tuple(t):
    sort_map = {x[0]: i for i, x in enumerate(sorted(zip(*np.unique(t, return_index=True)), key=lambda x: x[1]))}
    return tuple(sort_map[x] for x in t)


def plot_alternate(G_intralayer, G_interlayer, layer_vec, partitions, gamma_end, omega_end):
    NGAMMA = 10
    NOMEGA = 10
    PROGRESS_LENGTH = 50

    if 'weight' not in G_intralayer.es:
        G_intralayer.es['weight'] = [1.0] * G_intralayer.ecount()
    if 'weight' not in G_interlayer.es:
        G_interlayer.es['weight'] = [1.0] * G_interlayer.ecount()

    def part_color(membership):
        membership_val = hash(sorted_tuple(membership))
        return tuple((membership_val / x) % 1.0 for x in [157244317, 183849443, 137530733])

    denser_gammas = np.linspace(0, gamma_end, 200)
    denser_omegas = np.linspace(0, omega_end, 200)

    intralayer_part = louvain.RBConfigurationVertexPartitionWeightedLayers(G_intralayer, layer_vec=layer_vec,
                                                                           weights='weight')
    G_interlayer.es['weight'] = [1.0] * G_interlayer.ecount()
    interlayer_part = louvain.CPMVertexPartition(G_interlayer, resolution_parameter=0.0, weights='weight')

    total = len(partitions)
    iter_per_char = ceil(total / PROGRESS_LENGTH)
    current = 0
    start = time()

    # best_partitions = List(quality, partition, gamma, omega)
    best_partitions = [[(-inf,) * 4] * len(denser_omegas) for _ in range(len(denser_gammas))]
    for p in partitions:
        intralayer_part.set_membership(p)
        interlayer_part.set_membership(p)
        interlayer_base_quality = interlayer_part.quality()  # interlayer quality at omega=1.0
        for g_index, gamma in enumerate(denser_gammas):
            intralayer_quality = intralayer_part.quality(resolution_parameter=gamma)
            for o_index, omega in enumerate(denser_omegas):
                # omega * interlayer.quality() matches CPMVertexPartition.quality()
                # with omega as interlayer edge weights (as of 7/2)
                Q = intralayer_quality + omega * interlayer_base_quality
                if Q > best_partitions[g_index][o_index][0]:
                    best_partitions[g_index][o_index] = (Q, p, gamma, omega)

        current += 1
        print("\rSweep Progress: [{}{}], Time: {:.1f} s / {:.1f} s"
              "".format("#" * (current // iter_per_char),
                        "-" * (total // iter_per_char - current // iter_per_char),
                        time() - start,
                        (time() - start) * (total / current)), end="", flush=True)

    print("\nPlotting...", flush=True)

    gammas, omegas, colors = zip(*[(x[2], x[3], part_color(x[1])) for row in best_partitions for x in row])
    plt.scatter(gammas, omegas, color=colors, s=1, marker='s')
    plt.xlim([0, gamma_end])
    plt.ylim([0, omega_end])
    plt.xlabel("gamma")
    plt.ylabel("omega")
    plt.show()


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


G_intralayer, G_interlayer, layer_vec = generate_lazega_igraph()

domains = pickle.load(open("lazega_CHAMP.p", "rb"))

champ_parts = [membership for polyverts, membership in domains]
layer_vec = np.array(layer_vec)
nlayers = max(layer_vec) + 1
n = sum(layer_vec == 0)
champ_parts = np.array(champ_parts)

A = np.array(G_intralayer.get_adjacency().data)
C = np.array(G_interlayer.get_adjacency().data)
P = np.zeros((nlayers * n, nlayers * n))
for i in range(nlayers):
    c_degrees = np.array(G_intralayer.degree(list(range(n * i, n * i + n))))
    c_inds = np.where(layer_vec == i)[0]
    P[np.ix_(c_inds, c_inds)] = np.outer(c_degrees, c_degrees.T) / (1.0 * np.sum(c_degrees))

start = time()
coefarray = create_coefarray_from_partitions(champ_parts, A, P, C)
A_hats, P_hats, C_hats = partition_coefficients_3D(G_intralayer, G_interlayer, layer_vec, champ_parts)
our_coefarray = np.vstack((A_hats, P_hats, C_hats)).T
print(coefarray)

# domains = get_intersection(coefarray, max_pt=(1.0, 1.0))
print("CHAMP's implementation took {:.2f} s".format(time() - start))

print(our_coefarray)

domains = CHAMP_3D(G_intralayer, G_interlayer, layer_vec, champ_parts, 0.0, 2.0, 0.0, 2.0)
plot_2d_domains(domains, [0, 2.0], [0, 2.0])
plt.show()
plt.xlabel("gamma")
plt.ylabel("omega")
plot_alternate(G_intralayer, G_interlayer, layer_vec, champ_parts.tolist(), 2.0, 2.0)
plt.show()

# start = time()
# domains = CHAMP_3D(G_intralayer, G_interlayer, layer_vec, champ_parts.tolist(), 0.0, 1.0, 0.0, 1.0)
# print("Our implementation took {:.2f} s".format(time() - start))
