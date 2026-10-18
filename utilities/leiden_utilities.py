from collections.abc import Iterable
from typing import overload, Literal
import functools
from multiprocessing import Pool, cpu_count

import igraph as ig
import leidenalg
import numpy as np
from tqdm import tqdm

from .types import (FloatSequence, IntSequence, LayerTuple, LayerVector, LeidenPartition,
                    Membership, MultilayerLeidenParts, MultilayerLeidenResult, SingleLayerLeidenResult)

LOW_MEMORY_THRESHOLD: float = 1e9  # 1 GB


@functools.lru_cache(maxsize=1000)
def sorted_tuple(t: Membership) -> Membership:
    """Converts a tuple to a canonical form in which the labels' first occurrences are sorted (e.g. label 0 will always
    occur before label 1 in the tuple).

    :param t: community membership of a partition
    :type t: tuple[int, ...]
    :return: a canonical representation of the membership tuple
    :rtype: tuple[int, ...]
    """

    sort_map: dict[int, int] = (
        {x[0]: i for i, x in enumerate(sorted(zip(*np.unique(t, return_index=True)), key=lambda x: x[1]))}
    )
    return tuple(sort_map[x] for x in t)


# explain to mypy that the return type depends on the return_partition argument
@overload
def singlelayer_leiden(
    G: ig.Graph, gamma: float, *, return_partition: Literal[True],
) -> LeidenPartition:
    ...


@overload
def singlelayer_leiden(
    G: ig.Graph, gamma: float, *, return_partition: Literal[False] = False,
) -> Membership:
    ...


def singlelayer_leiden(
    G: ig.Graph, gamma: float, return_partition: bool = False,
) -> SingleLayerLeidenResult:
    r"""Run the Leiden modularity maximization algorithm at a single :math:`\gamma` value.

    Directed, undirected, weighted, and unweighted graphs are supported.

    :param G: graph of interest
    :type G: igraph.Graph
    :param gamma: gamma (resolution parameter) to run Leiden at
    :type gamma: float
    :param return_partition: if True, return a leidenalg partition. Otherwise, return a community membership tuple
    :type return_partition: bool
    :return: partition from leidenalg
    :rtype: tuple[int, ...] or leidenalg.RBConfigurationVertexPartition
    """
    if not G.is_weighted():
        G.es['weight'] = [1.0] * G.ecount()

    partition: LeidenPartition = leidenalg.find_partition(
        G, leidenalg.RBConfigurationVertexPartition, weights='weight', resolution_parameter=gamma
    )

    if return_partition:
        return partition
    else:
        return tuple(partition.membership)


def _wrapped_singlelayer_leiden(args: tuple[ig.Graph, float]) -> Membership:
    """Wrapped singlelayer_leiden() for use in multiprocessing.Pool.imap_unordered."""
    return singlelayer_leiden(*args)


def leiden_part(G: ig.Graph) -> LeidenPartition:
    if G.is_weighted():
        return leidenalg.RBConfigurationVertexPartition(G, weights='weight')
    return leidenalg.RBConfigurationVertexPartition(G)


def leiden_part_with_membership(
    G: ig.Graph, membership: IntSequence
) -> LeidenPartition:
    if isinstance(membership, np.ndarray):
        membership = membership.tolist()
    part: LeidenPartition = leiden_part(G)
    part.set_membership(membership)
    return part


def split_intralayer_leiden_graph(G_intralayer: ig.Graph, layer_membership: LayerVector) -> list[ig.Graph]:
    """
    Split intralayer network into a separate network for each layer.

    This is needed since leidenalg lacks support for faster multilayer optimization.

    :param G_intralayer: intralayer graph of interest
    :type G_intralayer: igraph.Graph
    :param layer_membership: list of each vertex's layer membership
    :type layer_membership: list[int]
    :return: list of intralayer networks
    :rtype: list[igraph.Graph]
    """
    # internally use hashable objects for memoization
    return _split_leiden_graph_layers_cached(n=G_intralayer.vcount(), G_es=tuple(G_intralayer.es),
                                             is_directed=G_intralayer.is_directed(),
                                             layer_membership=tuple(layer_membership))


@functools.lru_cache(maxsize=1)
def _split_leiden_graph_layers_cached(
    n: int, G_es: tuple[ig.Edge, ...], is_directed: bool, layer_membership: LayerTuple
) -> list[ig.Graph]:
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


# explain to mypy that the return type depends on the return_partition argument
@overload
def multilayer_leiden(
    G_intralayer: ig.Graph, G_interlayer: ig.Graph, layer_vec: LayerVector, gamma: float,
    omega: float, optimiser: leidenalg.Optimiser | None = None, *, return_partition: Literal[True],
) -> list[LeidenPartition]:
    ...


@overload
def multilayer_leiden(
    G_intralayer: ig.Graph, G_interlayer: ig.Graph, layer_vec: LayerVector, gamma: float,
    omega: float, optimiser: leidenalg.Optimiser | None = None, *, return_partition: Literal[False] = False,
) -> Membership:
    ...


def multilayer_leiden(
    G_intralayer: ig.Graph, G_interlayer: ig.Graph, layer_vec: LayerVector, gamma: float,
    omega: float, optimiser: leidenalg.Optimiser | None = None, return_partition: bool = False,
) -> MultilayerLeidenResult:
    r"""Run the Leiden modularity maximization algorithm at a single (:math:`\gamma, \omega`) value.

    Both the intralayer and interlayer graphs may be directed or undirected, weighted or unweighted.

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
    :type optimiser: leidenalg.Optimiser | None
    :param return_partition: if True, return a leidenalg partition. Otherwise, return a community membership tuple
    :type return_partition: bool
    :return: partition from leidenalg
    :rtype: tuple[int, ...] or list[leidenalg.RBConfigurationVertexPartition]
    """
    if not G_intralayer.is_weighted():
        G_intralayer.es['weight'] = [1.0] * G_intralayer.ecount()

    if not G_interlayer.is_weighted():
        G_interlayer.es['weight'] = [1.0] * G_interlayer.ecount()

    G_split_layers: list[ig.Graph] = split_intralayer_leiden_graph(G_intralayer, layer_vec)

    if optimiser is None:
        optimiser = leidenalg.Optimiser()

    intralayer_parts: list[LeidenPartition] = (
        [leidenalg.RBConfigurationVertexPartition(G_layer, weights='weight', resolution_parameter=gamma)
         for G_layer in G_split_layers]
    )
    interlayer_part: leidenalg.CPMVertexPartition = leidenalg.CPMVertexPartition(
        G_interlayer, resolution_parameter=0.0, weights='weight'
    )
    optimiser.optimise_partition_multiplex(intralayer_parts + [interlayer_part],
                                           layer_weights=[1] * len(intralayer_parts) + [omega])

    if return_partition:
        return intralayer_parts
    else:
        return tuple(intralayer_parts[0].membership)


def _wrapped_multilayer_leiden(args: tuple[ig.Graph, ig.Graph, LayerVector, float, float]
                               ) -> Membership:
    """Wrapped multilayer_leiden() for use in multiprocessing.Pool.imap_unordered."""
    return multilayer_leiden(*args)


def multilayer_leiden_part(
    G_intralayer: ig.Graph, G_interlayer: ig.Graph, layer_membership: LayerVector
) -> MultilayerLeidenParts:
    if not G_intralayer.is_weighted():
        G_intralayer.es['weight'] = [1.0] * G_intralayer.ecount()

    if not G_interlayer.is_weighted():
        G_interlayer.es['weight'] = [1.0] * G_interlayer.ecount()

    G_split_layers: list[ig.Graph] = split_intralayer_leiden_graph(G_intralayer, layer_membership)
    intralayer_parts: list[LeidenPartition] = (
        [leidenalg.RBConfigurationVertexPartition(G_layer, weights='weight') for G_layer in G_split_layers]
    )
    interlayer_part: leidenalg.CPMVertexPartition = leidenalg.CPMVertexPartition(
        G_interlayer, resolution_parameter=0.0, weights='weight'
    )
    return intralayer_parts, interlayer_part


def multilayer_leiden_part_with_membership(
    G_intralayer: ig.Graph, G_interlayer: ig.Graph, layer_membership: LayerVector,
    community_membership: IntSequence
) -> MultilayerLeidenParts:
    if isinstance(community_membership, np.ndarray):
        community_membership = community_membership.tolist()
    intralayer_parts, interlayer_part = multilayer_leiden_part(G_intralayer, G_interlayer, layer_membership)
    for intralayer_part in intralayer_parts:
        intralayer_part.set_membership(community_membership)
    interlayer_part.set_membership(community_membership)
    return intralayer_parts, interlayer_part


def repeated_leiden_from_gammas(G: ig.Graph, gammas: FloatSequence) -> set[Membership]:
    return {sorted_tuple(singlelayer_leiden(G, gamma)) for gamma in gammas}


def repeated_parallel_leiden_from_gammas(
    G: ig.Graph, gammas: FloatSequence, show_progress: bool = True
) -> set[Membership]:
    r"""Runs the Leiden modularity maximization algorithm at each provided :math:`\gamma` value, using all CPU cores.

    :param G: graph of interest
    :type G: igraph.Graph
    :param gammas: list of gammas (resolution parameters) to run Leiden at
    :type gammas: Sequence[float] | numpy.ndarray
    :param show_progress: if True, render a progress bar
    :type show_progress: bool
    :return: a set of all unique partitions returned by the Leiden algorithm
    :rtype: set of tuple[int, ...]
    """
    total: set[Membership] = set()
    pool_chunk_size: int = max(1, len(gammas) // (cpu_count() * 100))
    with Pool(processes=cpu_count()) as pool:
        pool_iterator: Iterable[Membership] = (
            pool.imap_unordered(_wrapped_singlelayer_leiden, [(G, g) for g in gammas],
                                chunksize=pool_chunk_size)
        )
        if show_progress:
            pool_iterator = tqdm(pool_iterator, total=len(gammas))

        for partition in pool_iterator:
            total.add(sorted_tuple(partition))

    return total


def repeated_leiden_from_gammas_omegas(
    G_intralayer: ig.Graph, G_interlayer: ig.Graph, layer_vec: LayerVector,
    gammas: FloatSequence, omegas: FloatSequence
) -> set[Membership]:
    return {sorted_tuple(tuple(multilayer_leiden(G_intralayer, G_interlayer, layer_vec, gamma, omega)))
            for gamma in gammas for omega in omegas}


def repeated_parallel_leiden_from_gammas_omegas(
    G_intralayer: ig.Graph, G_interlayer: ig.Graph, layer_vec: LayerVector,
    gammas: FloatSequence, omegas: FloatSequence,
    show_progress: bool = True
) -> set[Membership]:
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
    :type gammas: Sequence[float] | numpy.ndarray
    :param omegas: list of omegas to run leidenalg at
    :type omegas: Sequence[float] | numpy.ndarray
    :param show_progress: if True, render a progress bar
    :type show_progress: bool
    :return: a set of all unique partitions encountered
    :rtype: set of tuple[int, ...]
    """
    resolution_parameter_points: list[tuple[float, float]] = (
        [(gamma, omega) for gamma in gammas for omega in omegas]
    )

    total: set[Membership] = set()
    pool_chunk_size: int = max(1, len(resolution_parameter_points) // (cpu_count() * 100))
    with Pool(processes=cpu_count()) as pool:
        pool_iterator: Iterable[Membership] = (
            pool.imap_unordered(
                _wrapped_multilayer_leiden,
                [(G_intralayer, G_interlayer, layer_vec, gamma, omega)
                 for gamma, omega in resolution_parameter_points],
                chunksize=pool_chunk_size
            )
        )
        if show_progress:
            pool_iterator = tqdm(pool_iterator, total=len(resolution_parameter_points))

        for partition in pool_iterator:
            total.add(sorted_tuple(partition))

    return total
