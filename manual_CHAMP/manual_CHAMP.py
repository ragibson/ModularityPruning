from utilities import CHAMP_3D, plot_2d_domains
import louvain
from math import ceil, inf
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import numpy as np
import pickle
from time import time

GAMMA_END = 2.0
OMEGA_END = 2.0
NGAMMA = 10
NOMEGA = 10
PROGRESS_LENGTH = 50
optimiser = louvain.Optimiser()


def sorted_tuple(t):
    sort_map = {x[0]: i for i, x in enumerate(sorted(zip(*np.unique(t, return_index=True)), key=lambda x: x[1]))}
    return tuple(sort_map[x] for x in t)


def one_run(G_intralayer, G_interlayer, layer_vec, gamma, omega):
    G_interlayer.es['weight'] = [omega] * G_interlayer.ecount()
    wl_part = louvain.RBConfigurationVertexPartitionWeightedLayers(G_intralayer, resolution_parameter=gamma,
                                                                   layer_vec=layer_vec, weights='weight')
    wli_part = louvain.CPMVertexPartition(G_interlayer, resolution_parameter=0.0, weights='weight')
    optimiser.optimise_partition_multiplex([wl_part, wli_part])
    return wl_part.membership


def repeated_multilayer_louvain(G_intralayer, G_interlayer, layer_vec, gammas, omegas, threads=cpu_count()):
    pool = Pool(processes=threads)
    start = time()
    partitions = [pool.apply_async(one_run, (G_intralayer, G_interlayer, layer_vec, g, o))
                  for g in gammas for o in omegas]

    total = len(partitions)
    iter_per_char = ceil(total / PROGRESS_LENGTH)
    start = time()

    def get_and_progress(i, p):
        res = p.get(timeout=60)
        i += 1
        print("\rLouvain Progress: [{}{}], Time: {:.1f} s / {:.1f} s"
              "".format("#" * (i // iter_per_char),
                        "-" * (total // iter_per_char - i // iter_per_char),
                        time() - start,
                        (time() - start) * (total / i)), end="", flush=True)
        return res

    partitions = [get_and_progress(i, p) for i, p in enumerate(partitions)]
    print()
    return partitions


def run_alternate(G_intralayer, G_interlayer, layer_vec):
    if 'weight' not in G_intralayer.es:
        G_intralayer.es['weight'] = [1.0] * G_intralayer.ecount()
    if 'weight' not in G_interlayer.es:
        G_interlayer.es['weight'] = [1.0] * G_interlayer.ecount()
    return repeated_multilayer_louvain(G_intralayer, G_interlayer, layer_vec,
                                       np.linspace(0, GAMMA_END, NGAMMA),
                                       np.linspace(0, OMEGA_END, NOMEGA))


def plot_alternate(G_intralayer, G_interlayer, layer_vec, partitions):
    if 'weight' not in G_intralayer.es:
        G_intralayer.es['weight'] = [1.0] * G_intralayer.ecount()
    if 'weight' not in G_interlayer.es:
        G_interlayer.es['weight'] = [1.0] * G_interlayer.ecount()

    def part_color(membership):
        membership_val = hash(sorted_tuple(membership))
        return tuple((membership_val / x) % 1.0 for x in [157244317, 183849443, 137530733])

    denser_gammas = np.linspace(0, GAMMA_END, 200)
    denser_omegas = np.linspace(0, OMEGA_END, 200)

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
    plt.xlabel("gamma")
    plt.ylabel("omega")
    plt.show()


G_intralayer, G_interlayer = pickle.load(open("iter1.p", "rb"))

n_per_layer = 150
num_layers = 15
layer_vec = [i // n_per_layer for i in range(n_per_layer * num_layers)]

print("\n'manual' louvain:")
partitions = run_alternate(G_intralayer, G_interlayer, layer_vec)

domains = CHAMP_3D(G_intralayer, G_interlayer, layer_vec, partitions, 0.0, GAMMA_END, 0.0, OMEGA_END)
plot_2d_domains(domains, [0, GAMMA_END], [0, OMEGA_END])
plt.show()

print("\n'manual' CHAMP (with manual louvain):")
plot_alternate(G_intralayer, G_interlayer, layer_vec, partitions)
