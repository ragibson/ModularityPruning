from typing import Any
import functools
from multiprocessing import Pool, cpu_count

import igraph as ig
import leidenalg
import numpy as np
from tqdm import tqdm

LOW_MEMORY_THRESHOLD: float = 1e9  # 1 GB


@functools.lru_cache(maxsize=1000)
def sorted_tuple(t: tuple[int, ...]) -> tuple[int, ...]:
    """Converts a tuple to a canonical form in which the labels' first occurrences are sorted (e.g. label 0 will always
    occur before label 1 in the tuple).

    :param t: community membership of a partition
    :type t: tuple[int]
    :return: a canonical representation of the membership tuple
    :rtype: tuple[int]
    """

    sort_map: dict[np.int64, int] = {x[0]: i for i, x in enumerate(sorted(zip(*np.unique(t, return_index=True)), key=lambda x: x[1]))}
    return tuple(sort_map[x] for x in t)


def singlelayer_leiden(G: ig.Graph, gamma: float, return_partition: bool=False) -> leidenalg.RBConfigurationVertexPartition|tuple[int, ...]:
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

    partition: leidenalg.RBConfigurationVertexPartition = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition, weights='weight',
                                         resolution_parameter=gamma)

    if return_partition:
        return partition
    else:
        return tuple(partition.membership)


def _wrapped_singlelayer_leiden(args):
    """Wrapped singlelayer_leiden() for use in multiprocessing.Pool.imap_unordered."""
    return singlelayer_leiden(*args)


def leiden_part(G: ig.Graph) -> leidenalg.RBConfigurationVertexPartition:
    return leidenalg.RBConfigurationVertexPartition(G)


def leiden_part_with_membership(G: ig.Graph, membership: list[int]|tuple[int, ...]) -> leidenalg.RBConfigurationVertexPartition:
    if isinstance(membership, np.ndarray):
        membership = membership.tolist()
    part: leidenalg.RBConfigurationVertexPartition = leiden_part(G)
    part.set_membership(membership)
    return part


def split_intralayer_leiden_graph(G_intralayer: ig.Graph, layer_membership: list[int]) -> list[ig.Graph]:
    """
    Split intralayer network into a separate network for each layer.

    This is needed since leidenalg lacks support for faster multilayer optimization.

    :param G_intralayer: intralayer graph of interest
    :type G_intralayer: igraph.Graph
    :param layer_vec: list of each vertex's layer membership
    :type layer_vec: list[int]
    :return: list of intralayer networks
    :rtype: list[igraph.Graph]
    """
    # internally use hashable objects for memoization
    return _split_leiden_graph_layers_cached(n=G_intralayer.vcount(), G_es=tuple(G_intralayer.es),
                                             is_directed=G_intralayer.is_directed(),
                                             layer_membership=tuple(layer_membership))


@functools.lru_cache(maxsize=1)
def _split_leiden_graph_layers_cached(n: int, G_es: tuple[ig.Edge, ...], is_directed: bool, layer_membership: tuple[int, ...]) -> list[ig.Graph]:
    T: int = max(layer_membership) + 1

    edges_by_layer: list[list[tuple[int, int]]] = [[] for _ in range(T)]
    weights_by_layer: list[list[float]] = [[] for _ in range(T)]
    for e in G_es:
        e_layer: int = layer_membership[e.source]
        edges_by_layer[e_layer].append((e.source, e.target))
        weights_by_layer[e_layer].append(e['weight'])

    G_split_layers_list: list[ig.Graph] = []
    for layer_idx in range(T):
        G_layer: ig.Graph = ig.Graph(n=n, edges=edges_by_layer[layer_idx], directed=is_directed)
        G_layer.es['weight'] = weights_by_layer[layer_idx]
        G_split_layers_list.append(G_layer)
    return G_split_layers_list


def multilayer_leiden(G_intralayer: ig.Graph, G_interlayer: ig.Graph, layer_vec: list[int], gamma: float, omega: float, optimiser: leidenalg.Optimiser|None=None, return_partition: bool=False) -> list[leidenalg.RBConfigurationVertexPartition]|tuple[int, ...]:
    r"""Run the Leiden modularity maximization algorithm at a single (:math:`\gamma, \omega`) value.

    WARNING: Optimization can be EXTREMELY slow for large numbers of layers! Leidenalg does not properly implement
    multilayer optimization.

    :param G_intralayer: intralayer graph of interest
    :type G_intralayer: igraph.Graph
    :param G_interlayer: interlayer graph of interest
    :type G_interlayer: igraph.Graph
    :param layer_vec: list of each vertex's layer membership
    :type layer_vec: list[int]
    :param gamma: gamma (intralayer resolution parameter) to run Leiden at
    :type gamma: float
    :param omega: omega (interlayer resolution parameter) to run Leiden at
    :type omega: float
    :param optimiser: if not None, use passed-in (potentially custom) leidenalg optimiser
    :type optimiser: leidenalg.Optimiser
    :param return_partition: if True, return a leidenalg partition. Otherwise, return a community membership tuple
    :type return_partition: bool
    :return: partition from leidenalg
    :rtype: tuple[int] or leidenalg.RBConfigurationVertexPartitionWeightedLayers
    """
    if 'weight' not in G_intralayer.es:
        G_intralayer.es['weight'] = [1.0] * G_intralayer.ecount()

    if 'weight' not in G_interlayer.es:
        G_interlayer.es['weight'] = [1.0] * G_interlayer.ecount()

    G_split_layers: list[ig.Graph] = split_intralayer_leiden_graph(G_intralayer, layer_vec)

    if optimiser is None:
        optimiser = leidenalg.Optimiser()

    intralayer_parts: list[leidenalg.RBConfigurationVertexPartition] = [leidenalg.RBConfigurationVertexPartition(G_layer, weights='weight', resolution_parameter=gamma)
                        for G_layer in G_split_layers]
    interlayer_part: leidenalg.CPMVertexPartition = leidenalg.CPMVertexPartition(G_interlayer, resolution_parameter=0.0, weights='weight')
    optimiser.optimise_partition_multiplex(intralayer_parts + [interlayer_part],
                                           layer_weights=[1] * len(intralayer_parts) + [omega])

    if return_partition:
        return intralayer_parts
    else:
        return tuple(intralayer_parts[0].membership)


def _wrapped_multilayer_leiden(args):
    """Wrapped multilayer_leiden() for use in multiprocessing.Pool.imap_unordered."""
    return multilayer_leiden(*args)


def multilayer_leiden_part(G_intralayer: ig.Graph, G_interlayer: ig.Graph, layer_membership: list[int]) -> tuple[list[leidenalg.RBConfigurationVertexPartition], leidenalg.CPMVertexPartition]:
    if 'weight' not in G_intralayer.es:
        G_intralayer.es['weight'] = [1.0] * G_intralayer.ecount()

    if 'weight' not in G_interlayer.es:
        G_interlayer.es['weight'] = [1.0] * G_interlayer.ecount()

    G_split_layers: list[ig.Graph] = split_intralayer_leiden_graph(G_intralayer, layer_membership)
    intralayer_parts: list[leidenalg.RBConfigurationVertexPartition] = [leidenalg.RBConfigurationVertexPartition(G_layer, weights='weight')
                        for G_layer in G_split_layers]
    interlayer_part: leidenalg.CPMVertexPartition = leidenalg.CPMVertexPartition(G_interlayer, resolution_parameter=0.0, weights='weight')
    return intralayer_parts, interlayer_part


def multilayer_leiden_part_with_membership(G_intralayer: ig.Graph, G_interlayer: ig.Graph, layer_membership: list[int], community_membership: tuple[int, ...]) -> tuple[list[leidenalg.RBConfigurationVertexPartition], leidenalg.CPMVertexPartition]:
    if isinstance(community_membership, np.ndarray):
        community_membership = community_membership.tolist()
    intralayer_parts, interlayer_part = multilayer_leiden_part(G_intralayer, G_interlayer, layer_membership)
    for intralayer_part in intralayer_parts:
        intralayer_part.set_membership(community_membership)
    interlayer_part.set_membership(community_membership)
    return intralayer_parts, interlayer_part


def repeated_leiden_from_gammas(G: ig.Graph, gammas: list[float]) -> set[tuple[int, ...]]:
    return {sorted_tuple(singlelayer_leiden(G, gamma)) for gamma in gammas}


def repeated_parallel_leiden_from_gammas(G: ig.Graph, gammas: np.ndarray[Any, np.dtype[np.float64]], show_progress: bool=True) -> set[tuple[int, ...]]:
    r"""Runs the Leiden modularity maximization algorithm at each provided :math:`\gamma` value, using all CPU cores.

    :param G: graph of interest
    :type G: igraph.Graph
    :param gammas: list of gammas (resolution parameters) to run Leiden at
    :type gammas: list[float]
    :param show_progress: if True, render a progress bar
    :type show_progress: bool
    :return: a set of all unique partitions returned by the Leiden algorithm
    :rtype: set of tuple[int]
    """
    total: set[tuple[int, ...]] = set()
    pool_chunk_size: int = max(1, len(gammas) // (cpu_count() * 100))
    with Pool(processes=cpu_count()) as pool:
        pool_iterator: tqdm = pool.imap_unordered(_wrapped_singlelayer_leiden, [(G, g) for g in gammas],
                                            chunksize=pool_chunk_size)
        if show_progress:
            pool_iterator = tqdm(pool_iterator, total=len(gammas))

        for partition in pool_iterator:
            total.add(sorted_tuple(partition))

    return total


def repeated_leiden_from_gammas_omegas(G_intralayer: ig.Graph, G_interlayer: ig.Graph, layer_vec: list[int], gammas: list[float], omegas: list[float]) -> set[tuple[int, ...]]:
    return {sorted_tuple(multilayer_leiden(G_intralayer, G_interlayer, layer_vec, gamma, omega))
            for gamma in gammas for omega in omegas}


def repeated_parallel_leiden_from_gammas_omegas(G_intralayer: ig.Graph, G_interlayer: ig.Graph, layer_vec: list[int], gammas: np.ndarray[Any, np.dtype[np.float64]], omegas: np.ndarray[Any, np.dtype[np.float64]],
                                                show_progress: bool=True) -> set[tuple[int, ...]]:
    """
    Runs leidenalg at each gamma and omega in ``gammas`` and ``omegas``, using all CPU cores available.

    WARNING: Optimization can be EXTREMELY slow for large numbers of layers! Leidenalg does not properly implement
    multilayer optimization.

    :param G_intralayer: intralayer graph of interest
    :type G_intralayer: igraph.Graph
    :param G_interlayer: interlayer graph of interest
    :type G_interlayer: igraph.Graph
    :param layer_vec: list of each vertex's layer membership
    :type layer_vec: list[int]
    :param gammas: list of gammas to run leidenalg at
    :type gammas: list[float]
    :param omegas: list of omegas to run leidenalg at
    :type omegas: list[float]
    :param show_progress: if True, render a progress bar
    :type show_progress: bool
    :return: a set of all unique partitions encountered
    :rtype: set of tuple[int]
    """
    resolution_parameter_points: list[tuple[np.float64, np.float64]] = [(gamma, omega) for gamma in gammas for omega in omegas]

    total: set[tuple[int, ...]] = set()
    pool_chunk_size: int = max(1, len(resolution_parameter_points) // (cpu_count() * 100))
    with Pool(processes=cpu_count()) as pool:
        pool_iterator: tqdm = pool.imap_unordered(
            _wrapped_multilayer_leiden,
            [(G_intralayer, G_interlayer, layer_vec, gamma, omega) for gamma, omega in resolution_parameter_points],
            chunksize=pool_chunk_size
        )
        if show_progress:
            pool_iterator = tqdm(pool_iterator, total=len(resolution_parameter_points))

        for partition in pool_iterator:
            total.add(sorted_tuple(partition))

    return total
