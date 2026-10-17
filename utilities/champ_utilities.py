from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import igraph
    import multiprocessing.pool
import warnings
from collections import defaultdict
from math import floor
from multiprocessing import Pool, cpu_count

import numpy as np
from numpy.random import choice
from scipy.linalg import LinAlgWarning
from scipy.optimize import linprog, OptimizeWarning
from scipy.spatial import HalfspaceIntersection

from .partition_utilities import (all_strengths, in_strengths, out_strengths,
                                  membership_to_communities, membership_to_layered_communities)
from .types import (Domain, FloatArray, GammaRange, LayerVector, Membership,
                    Partitions, PartitionCoefficients2D, PartitionCoefficients3D)


def get_interior_point(
    halfspaces: FloatArray, initial_num_sampled: int = 50, full_retry_limit: int = 10
) -> FloatArray:
    """
    Find interior point of halfspaces (needed to perform halfspace intersection)

    :param halfspaces: list of halfspaces
    :param initial_num_sampled: initial number of halfspaces sampled for the linear program. If the resulting point is
                                not interior to all halfspaces, this value is doubled and the procedure is retried.
    :param full_retry_limit: number of times to retry upon encountering numerical issues with all halfspaces.
    :return: an approximation to the point most interior to the halfspace intersection polyhedron (Chebyshev center)
    """

    # We suppress these two warnings to avoid cluttering output, some of these warnings are expected as the result is
    # converged to and we've checked the consistency of results in our own tests. Moreover, we explicitly check the
    # interior point's validity prior to returning.
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    warnings.filterwarnings("ignore", category=OptimizeWarning)

    normals, offsets = np.split(halfspaces, [-1], axis=1)

    # in our singlelayer case, the last four halfspaces are boundary halfspaces (a square)
    # for multilayer, the last six are boundaries (a cube)
    if normals.shape[1] == 2:
        interior_hs, boundaries = np.split(halfspaces, [-4], axis=0)
    elif normals.shape[1] == 3:
        interior_hs, boundaries = np.split(halfspaces, [-6], axis=0)
    else:
        raise ValueError(f"get_interior_point received unhandled dimension {normals.shape[1]}!")

    num_retries: int = 0

    while True:  # retry until success
        # randomly sample some of the halfspaces
        sample_len: int = min(initial_num_sampled, len(interior_hs))  # len(interior_hs)
        sampled_hs: FloatArray = np.vstack((
            interior_hs[choice(interior_hs.shape[0], sample_len, replace=False)], boundaries))

        # compute the Chebyshev center of the sampled halfspaces' intersection
        norm_vector: FloatArray = np.reshape(
            np.linalg.norm(sampled_hs[:, :-1], axis=1), (sampled_hs.shape[0], 1))
        c: FloatArray = np.zeros((sampled_hs.shape[1],))
        c[-1] = -1
        A: FloatArray = np.hstack((sampled_hs[:, :-1], norm_vector))
        b: FloatArray = -sampled_hs[:, -1:]

        res = linprog(c, A_ub=A, b_ub=b, bounds=(-np.inf, np.inf), method='highs-ipm')

        # check solution if optimization succeeded, even with difficulties
        #     res.status codes
        #     0 : Optimization terminated successfully.
        #     1 : Iteration limit reached.
        #     2 : Problem appears to be infeasible.
        #     3 : Problem appears to be unbounded.
        #     4 : Numerical difficulties encountered.
        if res.success or res.status in {0, 1, 4}:
            if res.x is None:
                # satisfying mypy, this should never occur
                raise ValueError(f"Scipy's linprog claimed success but returned no solution! {res}")
            intpt: FloatArray = res.x[:-1]  # res.x contains [interior_point, distance to enclosing polyhedron]

            # ensure that the computed point is actually interior to all halfspaces
            if (np.dot(normals, intpt) + np.transpose(offsets) < 0).all():
                break
        elif res.status in {2, 3}:
            # if we ever fail, the linear program seems impossible
            raise ValueError("get_interior_point problem is impossible or degenerate!")

        if num_retries > full_retry_limit:
            raise ValueError("get_interior_point is unable to find a well-conditioned solution, "
                             "but the problem does not appear impossible or degenerate?")

        if initial_num_sampled >= len(interior_hs):
            num_retries += 1
        else:
            initial_num_sampled *= 2  # try again and sample more halfspaces this time

    return intpt


def CHAMP_2D(
    G: "igraph.Graph", all_parts: Partitions, gamma_0: float, gamma_f: float,
    single_threaded: bool = False
) -> list[GammaRange]:
    r"""Calculates the pruned set of partitions from CHAMP on ``gamma_0`` :math:`\leq \gamma \leq` ``gamma_f``.

    See https://doi.org/10.3390/a10030093 for more details.

    :param G: graph of interest
    :type G: igraph.Graph
    :param all_parts: partitions to prune
    :type all_parts: Sequence[tuple[int, ...]] | set[tuple[int, ...]]
    :param gamma_0: starting gamma value for CHAMP
    :type gamma_0: float
    :param gamma_f: ending gamma value for CHAMP
    :type gamma_f: float
    :param single_threaded: if True, run in serial. Otherwise, use all CPU cores to run in parallel
    :type single_threaded: bool
    :return: list of tuples for the somewhere optimal partitions, containing (in-order)

        - starting gamma value for the partition's domain of optimality
        - ending gamma value for the partition's domain of optimality
        - community membership tuple for the partition
    :rtype: list of tuple[float, float, tuple[int, ...]]
    """

    if len(all_parts) == 0:
        return []

    all_parts = list(all_parts)
    num_partitions: int = len(all_parts)

    partition_coefficients: PartitionCoefficients2D = partition_coefficients_2D(
        G, all_parts, single_threaded=single_threaded)
    A_hats, P_hats = partition_coefficients

    # add on boundaries: bottom < Q < top, gamma_0 < gamma < gamma_f
    top: np.float64 = max(A_hats - P_hats * gamma_0)  # Could potentially be optimized
    bottom: np.float64 = min(A_hats - P_hats * gamma_f)  # Could potentially be optimized
    halfspaces: FloatArray = np.vstack((halfspaces_from_coefficients_2D(*partition_coefficients),
                                        np.array([[0, 1, -top], [0, -1, bottom],
                                                  [1, 0, -gamma_f], [-1, 0, gamma_0]])))

    # normalize halfspaces to try to improve ill-conditioned linear programs
    halfspaces = halfspaces / np.linalg.norm(halfspaces, axis=1, keepdims=True)

    # Could potentially scale axes so Chebyshev center is better for problem
    interior_point: FloatArray = get_interior_point(halfspaces)
    hs: HalfspaceIntersection = HalfspaceIntersection(halfspaces, interior_point)

    # scipy does not support facets by halfspace directly, so we must compute them
    facets_by_halfspace: defaultdict[int, list[FloatArray]] = defaultdict(list)
    for v, idx in zip(hs.intersections, hs.dual_facets):
        assert np.isfinite(v).all()
        for i in idx:
            if i < num_partitions:
                facets_by_halfspace[i].append(v)

    ranges: list[GammaRange] = []
    for i, intersections in facets_by_halfspace.items():
        x1, x2 = intersections[0][0], intersections[1][0]
        if x1 > x2:
            x1, x2 = x2, x1
        ranges.append((x1, x2, all_parts[i]))

    return sorted(ranges, key=lambda x: x[0])


def CHAMP_3D(
    G_intralayer: "igraph.Graph", G_interlayer: "igraph.Graph", layer_vec: LayerVector,
    all_parts: Partitions, gamma_0: float, gamma_f: float, omega_0: float, omega_f: float
) -> list[Domain]:
    r"""Calculates the pruned set of partitions from CHAMP on ``gamma_0`` :math:`\leq \gamma \leq` ``gamma_f`` and
    ``omega_0`` :math:`\leq \omega \leq` ``omega_f``.

    See https://doi.org/10.3390/a10030093 for more details.

    :param G_intralayer: intralayer graph of interest
    :type G_intralayer: igraph.Graph
    :param G_interlayer: interlayer graph of interest
    :type G_interlayer: igraph.Graph
    :param layer_vec: list of each vertex's layer membership
    :type layer_vec: list[int]
    :param all_parts: partitions to prune
    :type all_parts: Sequence[tuple[int, ...]] | set[tuple[int, ...]]
    :param gamma_0: starting gamma value for CHAMP
    :type gamma_0: float
    :param gamma_f: ending gamma value for CHAMP
    :type gamma_f: float
    :param omega_0: starting omega value for CHAMP
    :type omega_0: float
    :param omega_f: ending omega value for CHAMP
    :type omega_f: float
    :return: list of tuples for the somewhere optimal partitions, containing (in-order)

        - list of polygon vertices in (gamma, omega) plane for the partition's domain of optimality
        - community membership tuple for the partition
    :rtype: list of tuple[list[numpy.ndarray], tuple[int, ...]]
    """

    if len(all_parts) == 0:
        return []

    all_parts = list(all_parts)
    num_partitions: int = len(all_parts)

    partition_coefficients: PartitionCoefficients3D = partition_coefficients_3D(
        G_intralayer, G_interlayer, layer_vec, all_parts)
    A_hats, P_hats, C_hats = partition_coefficients

    # add on boundaries: bottom < Q < top, gamma_0 < gamma < gamma_f, omega_0 < omega < omega_f
    top: np.float64 = max(A_hats - P_hats * gamma_0 + C_hats * omega_f)  # Could potentially be optimized
    bottom: np.float64 = min(A_hats - P_hats * gamma_f + C_hats * omega_0)  # Could potentially be optimized
    halfspaces: FloatArray = np.vstack((halfspaces_from_coefficients_3D(*partition_coefficients),
                                        (np.array([[0, 0, 1, -top], [0, 0, -1, bottom],
                                                   [1, 0, 0, -gamma_f], [-1, 0, 0, gamma_0],
                                                   [0, 1, 0, -omega_f], [0, -1, 0, omega_0]]))))

    # normalize halfspaces to try to improve ill-conditioned linear programs
    halfspaces = halfspaces / np.linalg.norm(halfspaces, axis=1, keepdims=True)

    # Could potentially scale axes so Chebyshev center is better for problem
    interior_point: FloatArray = get_interior_point(halfspaces)
    hs: HalfspaceIntersection = HalfspaceIntersection(halfspaces, interior_point)

    # scipy does not support facets by halfspace directly, so we must compute them
    facets_by_halfspace: defaultdict[int, list[FloatArray]] = defaultdict(list)
    for v, idx in zip(hs.intersections, hs.dual_facets):
        assert np.isfinite(v).all()
        for i in idx:
            if i < num_partitions:
                facets_by_halfspace[i].append(v)

    domains: list[Domain] = []
    for i, intersections in facets_by_halfspace.items():
        # drop quality dimension / project into (gamma, omega) plane
        intersections = [x[:2] for x in intersections]
        # sort the domain's points counter-clockwise around the centroid
        centroid: FloatArray = np.mean(intersections, axis=0)
        intersections.sort(key=lambda x: np.arctan2(x[0] - centroid[0], x[1] - centroid[1]))
        domains.append((intersections, all_parts[i]))

    return domains


def partition_coefficients_2D_serial(G: "igraph.Graph", partitions: Partitions
                                     ) -> PartitionCoefficients2D:
    """Computes A_hat and P_hat for partitions of :G:"""

    weights = G.es['weight'] if G.is_weighted() else [1.0] * G.ecount()
    total_weight = float(sum(weights))
    all_edges = [(e.source, e.target, w) for e, w in zip(G.es, weights)]

    # multiply by 2 only if undirected here
    if G.is_directed():
        A_hats = np.array([sum(w for u, v, w in all_edges if membership[u] == membership[v])
                           for membership in partitions], dtype=np.float64)
    else:
        A_hats = np.array([2 * sum(w for u, v, w in all_edges if membership[u] == membership[v])
                           for membership in partitions], dtype=np.float64)

    if G.is_directed():
        # directed modularity of Leicht and Newman is actually
        #   (1/m) sum_{ij} [A_{ij} - k_i^{in} * k_j^{out} / m] delta(c_i, c_j)
        in_strength = in_strengths(G)
        out_strength = out_strengths(G)
        P_hats = np.array([sum(sum(in_strength[v] for v in vs) * sum(out_strength[v] for v in vs)
                               for vs in membership_to_communities(membership).values())
                           for membership in partitions]) / total_weight
    else:
        twom = 2 * total_weight
        strength = all_strengths(G)
        P_hats = np.array([sum(sum(strength[v] for v in vs) ** 2
                               for vs in membership_to_communities(membership).values())
                           for membership in partitions]) / twom

    return A_hats, P_hats


def partition_coefficients_2D(
    G: "igraph.Graph", partitions: Partitions, single_threaded: bool = False
) -> PartitionCoefficients2D:
    """Computes partitions coefficients in parallel by calling partition_coefficients_2D_serial"""
    partitions = list(partitions)

    if single_threaded:
        A_hats, P_hats = partition_coefficients_2D_serial(G, partitions)
    else:
        partition_chunks: list[list[Membership]] = [
            partitions[floor(i * len(partitions) / cpu_count()):floor((i + 1) * len(partitions) / cpu_count())]
            for i in range(cpu_count())
        ]

        pool: "multiprocessing.pool.Pool" = Pool(processes=cpu_count())
        results: list[PartitionCoefficients2D] = \
            pool.starmap(partition_coefficients_2D_serial,
                         [(G, partition_chunk) for partition_chunk in partition_chunks])
        pool.close()

        A_hats = np.array([v for A_hats, P_hats in results for v in A_hats])
        P_hats = np.array([v for A_hats, P_hats in results for v in P_hats])

    assert len(A_hats) == len(P_hats) == len(partitions)

    return A_hats, P_hats


def halfspaces_from_coefficients_2D(
    A_hats: FloatArray, P_hats: FloatArray
) -> FloatArray:
    """Converts partitions' coefficients to halfspace normal, offset

    Q >= -P_hat*gamma + A_hat
    -Q - P_hat*gamma + A_hat <= 0
    (-P_hat, -1) * (gamma, Q) + A_hat <= 0
    """
    return np.vstack((-P_hats, -np.ones_like(P_hats), A_hats)).T.astype(np.float64)


def halfspaces_from_coefficients_3D(
    A_hats: FloatArray, P_hats: FloatArray, C_hats: FloatArray
) -> FloatArray:
    """Converts partitions' coefficients to halfspace normal, offset

    Q >= C_hat*omega - P_hat*gamma + A_hat
    -Q + C_hat*omega - P_hat*gamma + A_hat <= 0
    (-P_hat, C_hat, -1) * (gamma, omega, Q) + A_hat <= 0
    """
    return np.vstack((-P_hats, C_hats, -np.ones_like(P_hats), A_hats)).T.astype(np.float64)


def partition_coefficients_3D_serial(G_intralayer: "igraph.Graph", G_interlayer: "igraph.Graph",
                                     layer_vec: LayerVector,
                                     partitions: list[Membership]) -> PartitionCoefficients3D:
    """Computes A_hat, P_hat, C_hat for partitions of a graph with intralayer edges given in :G_intralayer:,
    interlayer edges given in :G_interlayer:, and layer membership :layer_vec:"""

    intra_weights = G_intralayer.es['weight'] if G_intralayer.is_weighted() else [1.0] * G_intralayer.ecount()
    inter_weights = G_interlayer.es['weight'] if G_interlayer.is_weighted() else [1.0] * G_interlayer.ecount()
    all_intralayer_edges = [(e.source, e.target, w) for e, w in zip(G_intralayer.es, intra_weights)]
    all_interlayer_edges = [(e.source, e.target, w) for e, w in zip(G_interlayer.es, inter_weights)]

    # multiply by 2 only if undirected here
    if G_intralayer.is_directed():
        A_hats = np.array([sum(w for u, v, w in all_intralayer_edges if membership[u] == membership[v])
                           for membership in partitions], dtype=np.float64)
    else:
        A_hats = np.array([2 * sum(w for u, v, w in all_intralayer_edges if membership[u] == membership[v])
                           for membership in partitions], dtype=np.float64)

    num_layers = max(layer_vec) + 1
    weight_per_layer = np.zeros(num_layers)
    for e, w in zip(G_intralayer.es, intra_weights):
        weight_per_layer[layer_vec[e.source]] += w

    if G_intralayer.is_directed():
        in_strength = in_strengths(G_intralayer)
        out_strength = out_strengths(G_intralayer)
        P_hats = np.array([
            sum(sum(in_strength[v] for v in vs) * sum(out_strength[v] for v in vs) / weight_per_layer[layer]
                for (_, layer), vs in membership_to_layered_communities(membership, layer_vec).items())
            for membership in partitions
        ])
    else:
        strength = all_strengths(G_intralayer)
        P_hats = np.array([
            sum(sum(strength[v] for v in vs) ** 2 / (2 * weight_per_layer[layer])
                for (_, layer), vs in membership_to_layered_communities(membership, layer_vec).items())
            for membership in partitions
        ])

    # multiply by 2 only if undirected here
    if G_interlayer.is_directed():
        C_hats = np.array([sum(w for u, v, w in all_interlayer_edges if membership[u] == membership[v])
                           for membership in partitions], dtype=np.float64)
    else:
        C_hats = np.array([2 * sum(w for u, v, w in all_interlayer_edges if membership[u] == membership[v])
                           for membership in partitions], dtype=np.float64)

    return A_hats, P_hats, C_hats


def partition_coefficients_3D(
    G_intralayer: "igraph.Graph", G_interlayer: "igraph.Graph", layer_vec: LayerVector,
    partitions: Partitions
) -> PartitionCoefficients3D:
    """Computes partitions coefficients in parallel by calling partition_coefficients_3D_serial"""
    partitions = list(partitions)
    partition_chunks: list[list[Membership]] = [
        partitions[floor(i * len(partitions) / cpu_count()):floor((i + 1) * len(partitions) / cpu_count())]
        for i in range(cpu_count())
    ]

    pool: "multiprocessing.pool.Pool" = Pool(processes=cpu_count())
    results: list[PartitionCoefficients3D] = pool.starmap(
        partition_coefficients_3D_serial, [(G_intralayer, G_interlayer, layer_vec, partition_chunk)
                                           for partition_chunk in partition_chunks])
    pool.close()

    A_hats: FloatArray = np.array([v for A_hats, P_hats, C_hats in results for v in A_hats])
    P_hats: FloatArray = np.array([v for A_hats, P_hats, C_hats in results for v in P_hats])
    C_hats: FloatArray = np.array([v for A_hats, P_hats, C_hats in results for v in C_hats])

    assert len(A_hats) == len(P_hats) == len(C_hats) == len(partitions)

    return A_hats, P_hats, C_hats
