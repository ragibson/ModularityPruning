from .progress import Progress
import functools
# import leidenalg
import louvain
from multiprocessing import Pool, cpu_count
import numpy as np


@functools.lru_cache(maxsize=1000)
def sorted_tuple(t):
    """Converts a tuple :t: to a canonical form (labels' first occurrences are sorted)."""

    sort_map = {x[0]: i for i, x in enumerate(sorted(zip(*np.unique(t, return_index=True)), key=lambda x: x[1]))}
    return tuple(sort_map[x] for x in t)


def singlelayer_louvain(G, gamma):
    return tuple(louvain.find_partition(G, louvain.RBConfigurationVertexPartition,
                                        resolution_parameter=gamma).membership)


def multilayer_louvain(G_intralayer, G_interlayer, layer_vec, gamma, omega):
    # RBConfigurationVertexPartitionWeightedLayers implements a multilayer version of "standard" modularity (i.e.
    # the Reichardt and Bornholdt's Potts model with configuration null model).

    # TODO: does G_interlayer need to be copied?
    if 'weight' not in G_intralayer.es:
        G_intralayer.es['weight'] = [1.0] * G_intralayer.ecount()

    optimiser = louvain.Optimiser()
    G_interlayer.es['weight'] = omega
    intralayer_part = \
        louvain.RBConfigurationVertexPartitionWeightedLayers(G_intralayer, layer_vec=layer_vec, weights='weight',
                                                             resolution_parameter=gamma)
    interlayer_part = louvain.CPMVertexPartition(G_interlayer, resolution_parameter=0.0, weights='weight')
    optimiser.optimise_partition_multiplex([intralayer_part, interlayer_part])
    return tuple(intralayer_part.membership)


def louvain_part(G):
    return louvain.RBConfigurationVertexPartition(G)


def louvain_part_with_membership(G, membership):
    if isinstance(membership, np.ndarray):
        membership = membership.tolist()
    part = louvain_part(G)
    part.set_membership(membership)
    return part


def part_modularity(G, membership, gamma):
    """Returns the modularity of a partition with :membership: at :gamma:."""
    part = louvain_part_with_membership(G, membership)
    return part.quality(gamma)


def num_communities(membership):
    n = len(set(membership))
    assert n == max(membership) + 1
    return n


def repeated_parallel_louvain_from_gammas(G, gammas, show_progress=True):
    """
    Runs louvain at each gamma in :gammas:, using all CPU cores available.

    Returns a set of all unique partitions encountered.
    """

    if show_progress:
        progress = Progress(100)

    pool = Pool(processes=cpu_count())
    total = set()

    chunk_size = len(gammas) // 99
    if chunk_size > 0:
        chunk_params = ([(G, g) for g in gammas[i:i + chunk_size]] for i in range(0, len(gammas), chunk_size))
    else:
        chunk_params = [[(G, g) for g in gammas]]

    for chunk in chunk_params:
        for partition in pool.starmap(singlelayer_louvain, chunk):
            total.add(sorted_tuple(partition))

        if show_progress:
            progress.increment()

    if show_progress:
        progress.done()

    pool.close()
    return total


def repeated_parallel_louvain_from_gammas_omegas(G_intralayer, G_interlayer, layer_vec, gammas, omegas,
                                                 show_progress=True):
    resolution_parameter_points = [(gamma, omega) for gamma in gammas for omega in omegas]

    if show_progress:
        progress = Progress(100)

    pool = Pool(processes=cpu_count())
    total = set()

    chunk_size = len(resolution_parameter_points) // 99
    if chunk_size > 0:
        chunk_params = ([(G_intralayer, G_interlayer, layer_vec, gamma, omega)
                         for gamma, omega in resolution_parameter_points[i:i + chunk_size]]
                        for i in range(0, len(resolution_parameter_points), chunk_size))
    else:
        chunk_params = [[(G_intralayer, G_interlayer, layer_vec, gamma, omega)
                         for gamma, omega in resolution_parameter_points]]

    # chunk_idx = 0

    for chunk in chunk_params:
        for partition in pool.starmap(multilayer_louvain, chunk):
            total.add(sorted_tuple(partition))

        # import pickle
        # chunk_idx += 1
        # pickle.dump(total, open("senate_10K_chunk{}.p".format(chunk_idx), "wb"))

        if show_progress:
            progress.increment()

    if show_progress:
        progress.done()

    return total

# def repeated_parallel_louvain(G, gamma_0, gamma_f, gamma_iters=10000, repeat=10, show_progress=True):
#     """Runs repeated_louvain, but with mutiple processes."""
#     pool = Pool(processes=cpu_count())
#     args = [(G, gamma_0, gamma_f, gamma_iters, 1, show_progress) for _ in range(repeat)]
#
#     total = set()
#     for partitions in pool.starmap(repeated_louvain, args):
#         total.update(partitions)
#     pool.close()
#     return total


# def repeated_louvain(G, gamma_0, gamma_f, gamma_iters=10000, repeat=10, show_progress=True):
#     """
#     Runs louvain at :gamma_iters: gamma points in [:gamma_0:, :gamma_f:], repeating :repeat: times at each point.
#
#     Returns a set of all unique partitions encountered.
#     """
#
#     all_partitions = set()
#     if show_progress:
#         progress = Progress(gamma_iters)
#
#     for i, g in enumerate(np.linspace(gamma_0, gamma_f, gamma_iters)):
#         for _ in range(repeat):
#             part = louvain.find_partition(G, louvain.RBConfigurationVertexPartition, resolution_parameter=g)
#             all_partitions.add(sorted_tuple(tuple(part.membership)))
#
#         if show_progress:
#             progress.update(i)
#
#     if show_progress:
#         progress.done()
#
#     return all_partitions
