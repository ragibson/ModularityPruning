# Runs a performance comparison and consistency test between our CHAMP usage and the champ Python package
# Notably, our calculation of the partition coefficient array is significantly faster than CHAMP's
# Furthermore, CHAMP's calculation appears incorrect when the intralayer graph is directed

import louvain
from math import inf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import lzma
from time import time
from champ import create_coefarray_from_partitions
from champ import get_intersection as champ_get_intersection
from champ import plot_2d_domains as champ_plot_2d_domains
from modularitypruning.champ_utilities import CHAMP_3D, partition_coefficients_3D
from modularitypruning.louvain_utilities import sorted_tuple, repeated_parallel_louvain_from_gammas_omegas
from modularitypruning.plotting import plot_2d_domains

TEST_DIRECTED_INTRALAYER = False
GAMMA_END = 2.0
OMEGA_END = 2.0
NGAMMA = 25
NOMEGA = 25


def plot_manual_CHAMP(G_intralayer, G_interlayer, layer_vec, partitions):
    """Run an inefficient method to plot optimal modularity partititions across the (gamma, omega) plane to check
    consistency of the CHAMP implementation"""

    if 'weight' not in G_intralayer.es:
        G_intralayer.es['weight'] = [1.0] * G_intralayer.ecount()
    if 'weight' not in G_interlayer.es:
        G_interlayer.es['weight'] = [1.0] * G_interlayer.ecount()

    def part_color(membership):
        membership_val = hash(sorted_tuple(membership))
        return tuple((membership_val / x) % 1.0 for x in [157244317, 183849443, 137530733])

    denser_gammas = np.linspace(0, GAMMA_END, 250)
    denser_omegas = np.linspace(0, OMEGA_END, 250)

    intralayer_part = louvain.RBConfigurationVertexPartitionWeightedLayers(G_intralayer, layer_vec=layer_vec,
                                                                           weights='weight')
    G_interlayer.es['weight'] = [1.0] * G_interlayer.ecount()
    interlayer_part = louvain.CPMVertexPartition(G_interlayer, resolution_parameter=0.0, weights='weight')

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

    gammas, omegas, colors = zip(*[(x[2], x[3], part_color(x[1])) for row in best_partitions for x in row])
    plt.scatter(gammas, omegas, color=colors, s=1, marker='s')
    plt.xlabel("gamma")
    plt.ylabel("omega")


def run_CHAMP_with_their_coefarray_computation(G_intralayer, G_interlayer, layer_vec, partitions):
    layer_vec = np.array(layer_vec)
    nlayers = max(layer_vec) + 1

    A = np.array(G_intralayer.get_adjacency().data)
    C = np.array(G_interlayer.get_adjacency().data)
    P = np.zeros((G_intralayer.vcount(), G_intralayer.vcount()))
    for i in range(nlayers):
        c_inds = np.where(layer_vec == i)[0]
        c_degrees = np.array(G_intralayer.degree(c_inds))
        P[np.ix_(c_inds, c_inds)] = np.outer(c_degrees, c_degrees.T) / (1.0 * np.sum(c_degrees))

    start = time()
    coefarray = create_coefarray_from_partitions(np.array(list(partitions)), A, P, C)
    domains = champ_get_intersection(coefarray, max_pt=(GAMMA_END, OMEGA_END))

    print(f"CHAMP with their coefarray computation took {time() - start:.2f} s")
    champ_plot_2d_domains(domains)


def run_CHAMP_with_our_coefarray_computation(G_intralayer, G_interlayer, layer_vec, partitions):
    start = time()
    A_hats, P_hats, C_hats = partition_coefficients_3D(G_intralayer, G_interlayer, layer_vec, partitions)
    coefarray = np.vstack((A_hats, P_hats, C_hats)).T
    domains = champ_get_intersection(coefarray, max_pt=(GAMMA_END, OMEGA_END))

    print(f"CHAMP with our coefarray computation took {time() - start:.2f} s")
    champ_plot_2d_domains(domains)


def run_our_CHAMP_implementation(G_intralayer, G_interlayer, layer_vec, partitions):
    start = time()
    domains = CHAMP_3D(G_intralayer, G_interlayer, layer_vec, partitions, 0.0, GAMMA_END, 0.0, OMEGA_END)
    print(f"Our implementation took {time() - start:.2f} s")
    plot_2d_domains(domains, [0, GAMMA_END], [0, OMEGA_END])


if __name__ == "__main__":
    G_intralayer, G_interlayer = pickle.loads(lzma.decompress(pickle.load(open("compressed_network.p", "rb"))))

    if TEST_DIRECTED_INTRALAYER:
        G_intralayer.to_directed()

    n_per_layer = 150
    num_layers = 15
    layer_vec = [i // n_per_layer for i in range(n_per_layer * num_layers)]

    partitions = repeated_parallel_louvain_from_gammas_omegas(G_intralayer, G_interlayer, layer_vec,
                                                              gammas=np.linspace(0, GAMMA_END, NGAMMA),
                                                              omegas=np.linspace(0, OMEGA_END, NOMEGA))

    run_CHAMP_with_their_coefarray_computation(G_intralayer, G_interlayer, layer_vec, partitions)
    plt.savefig("CHAMP_test_their_coefarray.png")

    run_CHAMP_with_our_coefarray_computation(G_intralayer, G_interlayer, layer_vec, partitions)
    plt.savefig("CHAMP_test_our_coefarray.png")

    run_our_CHAMP_implementation(G_intralayer, G_interlayer, layer_vec, partitions)
    plt.savefig("CHAMP_test_our_implementation.png")

    plot_manual_CHAMP(G_intralayer, G_interlayer, layer_vec, partitions)
    plt.savefig("CHAMP_test_manual_plot.png")
