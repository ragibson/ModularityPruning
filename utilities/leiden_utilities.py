from .progress import Progress
import functools
import leidenalg
import louvain  # TODO: continue removing louvain usages
from math import ceil
from multiprocessing import Pool, cpu_count
import numpy as np
import psutil

LOW_MEMORY_THRESHOLD = 1e9  # 1 GB


@functools.lru_cache(maxsize=1000)
def sorted_tuple(t):
    """Converts a tuple to a canonical form in which the labels' first occurrences are sorted (e.g. label 0 will always
    occur before label 1 in the tuple).

    :param t: community membership of a partition
    :type t: tuple[int]
    :return: a canonical representation of the membership tuple
    :rtype: tuple[int]
    """

    sort_map = {x[0]: i for i, x in enumerate(sorted(zip(*np.unique(t, return_index=True)), key=lambda x: x[1]))}
    return tuple(sort_map[x] for x in t)


def singlelayer_leiden(G, gamma, return_partition=False):
    r"""Run the Leiden modularity maximization algorithm at a single :math:`\gamma` value.

    :param G: graph of interest
    :type G: igraph.Graph
    :param gamma: gamma (resolution parameter) to run Leiden at
    :type gamma: float
    :param return_partition: if True, return a leidenalg partition. Otherwise, return a community membership tuple
    :type return_partition: bool
    :return: partition from leidenalg
    :rtype: tuple[int] or leidenalg.RBConfigurationVertexPartition
    """
    if 'weight' not in G.es:
        G.es['weight'] = [1.0] * G.ecount()

    partition = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition, weights='weight',
                                         resolution_parameter=gamma)

    if return_partition:
        return partition
    else:
        return tuple(partition.membership)


def check_multilayer_louvain_capabilities(fatal=True):
    """Check if we are using the version of louvain with fast multilayer optimization.

    :param fatal: if True, raise an error when we are not using the desired louvain version
    """
    # TODO: this method is truly still using louvain
    try:
        louvain.RBConfigurationVertexPartitionWeightedLayers
    except AttributeError as e:
        if not fatal:
            return False

        raise AttributeError("Your installation of louvain does not support fast multilayer optimization. See "
                             "https://github.com/wweir827/louvain-igraph and "
                             "https://github.com/vtraag/louvain-igraph/pull/34") from e

    return True


def multilayer_louvain(G_intralayer, G_interlayer, layer_vec, gamma, omega, optimiser=None, return_partition=False):
    r"""Run the Louvain modularity maximization algorithm at a single (:math:`\gamma, \omega`) value.

    :param G_intralayer: intralayer graph of interest
    :type G_intralayer: igraph.Graph
    :param G_interlayer: interlayer graph of interest
    :type G_interlayer: igraph.Graph
    :param layer_vec: list of each vertex's layer membership
    :type layer_vec: list[int]
    :param gamma: gamma (intralayer resolution parameter) to run Louvain at
    :type gamma: float
    :param omega: omega (interlayer resolution parameter) to run Louvain at
    :type omega: float
    :param optimiser: if not None, use passed-in (potentially custom) louvain optimiser
    :type optimiser: louvain.Optimiser
    :param return_partition: if True, return a louvain partition. Otherwise, return a community membership tuple
    :type return_partition: bool
    :return: partition from louvain
    :rtype: tuple[int] or louvain.RBConfigurationVertexPartitionWeightedLayers
    """
    # TODO: this method is truly still using louvain
    # RBConfigurationVertexPartitionWeightedLayers implements a multilayer version of "standard" modularity (i.e.
    # the Reichardt and Bornholdt's Potts model with configuration null model).
    check_multilayer_louvain_capabilities()

    if 'weight' not in G_intralayer.es:
        G_intralayer.es['weight'] = [1.0] * G_intralayer.ecount()

    if 'weight' not in G_interlayer.es:
        G_interlayer.es['weight'] = [1.0] * G_interlayer.ecount()

    if optimiser is None:
        optimiser = louvain.Optimiser()

    intralayer_part = louvain.RBConfigurationVertexPartitionWeightedLayers(G_intralayer, layer_vec=layer_vec,
                                                                           weights='weight', resolution_parameter=gamma)
    interlayer_part = louvain.CPMVertexPartition(G_interlayer, resolution_parameter=0.0, weights='weight')
    optimiser.optimise_partition_multiplex([intralayer_part, interlayer_part], layer_weights=[1, omega])

    if return_partition:
        return intralayer_part
    else:
        return tuple(intralayer_part.membership)


def leiden_part(G):
    return leidenalg.RBConfigurationVertexPartition(G)


def leiden_part_with_membership(G, membership):
    if isinstance(membership, np.ndarray):
        membership = membership.tolist()
    part = leiden_part(G)
    part.set_membership(membership)
    return part


def multilayer_louvain_part(G_intralayer, G_interlayer, layer_membership):
    # TODO: this method is truly still using louvain
    if 'weight' not in G_intralayer.es:
        G_intralayer.es['weight'] = [1.0] * G_intralayer.ecount()

    if 'weight' not in G_interlayer.es:
        G_interlayer.es['weight'] = [1.0] * G_interlayer.ecount()

    intralayer_part = louvain.RBConfigurationVertexPartitionWeightedLayers(G_intralayer, layer_vec=layer_membership,
                                                                           weights='weight')
    interlayer_part = louvain.CPMVertexPartition(G_interlayer, resolution_parameter=0.0, weights='weight')
    return intralayer_part, interlayer_part


def multilayer_louvain_part_with_membership(G_intralayer, G_interlayer, layer_membership, community_membership):
    # TODO: this method is truly still using louvain
    if isinstance(community_membership, np.ndarray):
        community_membership = community_membership.tolist()
    intralayer_part, interlayer_part = multilayer_louvain_part(G_intralayer, G_interlayer, layer_membership)
    intralayer_part.set_membership(community_membership)
    interlayer_part.set_membership(community_membership)
    return intralayer_part, interlayer_part


def repeated_leiden_from_gammas(G, gammas):
    return {sorted_tuple(singlelayer_leiden(G, gamma)) for gamma in gammas}


def repeated_parallel_leiden_from_gammas(G, gammas, show_progress=True, chunk_dispatch=True):
    r"""Runs the Leiden modularity maximization algorithm at each provided :math:`\gamma` value, using all CPU cores.

    :param G: graph of interest
    :type G: igraph.Graph
    :param gammas: list of gammas (resolution parameters) to run Leiden at
    :type gammas: list[float]
    :param show_progress: if True, render a progress bar. This will only work if ``chunk_dispatch`` is also True
    :type show_progress: bool
    :param chunk_dispatch: if True, dispatch parallel work in chunks. Setting this to False may increase performance,
                           but can lead to out-of-memory issues
    :type chunk_dispatch: bool
    :return: a set of all unique partitions returned by the Leiden algorithm
    :rtype: set of tuple[int]
    """

    pool = Pool(processes=cpu_count())
    total = set()

    chunk_size = len(gammas) // 99
    if chunk_size > 0 and chunk_dispatch:
        chunk_params = ([(G, g) for g in gammas[i:i + chunk_size]] for i in range(0, len(gammas), chunk_size))
    else:
        chunk_params = [[(G, g) for g in gammas]]
        chunk_size = len(gammas)

    if show_progress:
        progress = Progress(ceil(len(gammas) / chunk_size))

    for chunk in chunk_params:
        for partition in pool.starmap(singlelayer_leiden, chunk):
            total.add(sorted_tuple(partition))

        if show_progress:
            progress.increment()

        if psutil.virtual_memory().available < LOW_MEMORY_THRESHOLD:
            # Reinitialize pool to get around an apparent memory leak in multiprocessing
            pool.close()
            pool = Pool(processes=cpu_count())

    if show_progress:
        progress.done()

    pool.close()
    return total


def repeated_louvain_from_gammas_omegas(G_intralayer, G_interlayer, layer_vec, gammas, omegas):
    # TODO: this method is truly still using louvain
    return {sorted_tuple(multilayer_louvain(G_intralayer, G_interlayer, layer_vec, gamma, omega))
            for gamma in gammas for omega in omegas}


def repeated_parallel_louvain_from_gammas_omegas(G_intralayer, G_interlayer, layer_vec, gammas, omegas,
                                                 show_progress=True, chunk_dispatch=True):
    """
    Runs louvain at each gamma and omega in ``gammas`` and ``omegas``, using all CPU cores available.

    :param G_intralayer: intralayer graph of interest
    :type G_intralayer: igraph.Graph
    :param G_interlayer: interlayer graph of interest
    :type G_interlayer: igraph.Graph
    :param layer_vec: list of each vertex's layer membership
    :type layer_vec: list[int]
    :param gammas: list of gammas to run louvain at
    :type gammas: list[float]
    :param omegas: list of omegas to run louvain at
    :type omegas: list[float]
    :param show_progress: if True, render a progress bar
    :type show_progress: bool
    :param chunk_dispatch: if True, dispatch parallel work in chunks. Setting this to False may increase performance,
                           but can lead to out-of-memory issues
    :type chunk_dispatch: bool
    :return: a set of all unique partitions encountered
    :rtype: set of tuple[int]
    """
    # TODO: this method is truly still using louvain
    resolution_parameter_points = [(gamma, omega) for gamma in gammas for omega in omegas]

    pool = Pool(processes=cpu_count())
    total = set()

    chunk_size = len(resolution_parameter_points) // 99
    if chunk_size > 0 and chunk_dispatch:
        chunk_params = ([(G_intralayer, G_interlayer, layer_vec, gamma, omega)
                         for gamma, omega in resolution_parameter_points[i:i + chunk_size]]
                        for i in range(0, len(resolution_parameter_points), chunk_size))
    else:
        chunk_params = [[(G_intralayer, G_interlayer, layer_vec, gamma, omega)
                         for gamma, omega in resolution_parameter_points]]
        chunk_size = len(gammas)

    if show_progress:
        progress = Progress(ceil(len(resolution_parameter_points) / chunk_size))

    for chunk in chunk_params:
        for partition in pool.starmap(multilayer_louvain, chunk):
            total.add(sorted_tuple(partition))

        if show_progress:
            progress.increment()

        if psutil.virtual_memory().available < LOW_MEMORY_THRESHOLD:
            # Reinitialize pool to get around an apparent memory leak in multiprocessing
            pool.close()
            pool = Pool(processes=cpu_count())

    if show_progress:
        progress.done()

    pool.close()
    return total
