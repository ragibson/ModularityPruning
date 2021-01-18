from .partition_utilities import all_degrees, in_degrees, out_degrees, membership_to_communities, \
    membership_to_layered_communities
from collections import defaultdict
from champ import get_intersection
import numpy as np
from numpy import VisibleDeprecationWarning
from numpy.random import choice
from math import floor
from multiprocessing import Pool, cpu_count
from scipy.spatial import HalfspaceIntersection
from scipy.linalg import LinAlgWarning
from scipy.optimize import linprog, OptimizeWarning
import warnings


def get_interior_point(halfspaces, initial_num_sampled=50):
    """
    Find interior point of halfspaces (needed to perform halfspace intersection)

    :param halfspaces: list of halfspaces
    :param initial_num_sampled: initial number of halfspaces sampled for the linear program. If the resulting point is
                                not interior to all halfspaces, this value is doubled and the procedure is retried.
    :return: an approximation to the point most interior to the halfspace intersection polyhedron (Chebyshev center)
    """

    # We suppress these two warnings to avoid cluttering output, some of these warnings are expected as the result is
    # converged to and we've checked the consistency of results in our own tests. Moreover, we explicitly check the
    # interior point's validity prior to returning.
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    warnings.filterwarnings("ignore", category=OptimizeWarning)

    normals, offsets = np.split(halfspaces, [-1], axis=1)

    # in our singlelayer case, the last two halfspaces are boundary halfspaces
    interior_hs, boundaries = np.split(halfspaces, [-2], axis=0)

    while True:  # retry until success
        # randomly sample some of the halfspaces
        sample_len = min(initial_num_sampled, len(interior_hs))  # len(interior_hs)
        sampled_hs = np.vstack((interior_hs[choice(interior_hs.shape[0], sample_len, replace=False)], boundaries))

        # compute the Chebyshev center of the sampled halfspaces' intersection
        norm_vector = np.reshape(np.linalg.norm(sampled_hs[:, :-1], axis=1), (sampled_hs.shape[0], 1))
        c = np.zeros((sampled_hs.shape[1],))
        c[-1] = -1
        A = np.hstack((sampled_hs[:, :-1], norm_vector))
        b = -sampled_hs[:, -1:]

        res = linprog(c, A_ub=A, b_ub=b, bounds=(-np.inf, np.inf), method='interior-point')

        if res.status == 0 and res.success:
            intpt = res.x[:-1]  # res.x contains [interior_point, distance to enclosing polyhedron]

            # ensure that the computed point is actually interior to all halfspaces
            if (np.dot(normals, intpt) + np.transpose(offsets) < 0).all():
                break

        # res.status codes
        # 1: "Interior point calculation: scipy.optimize.linprog exceeded iteration limit"
        # 2: "Interior point calculation: scipy.optimize.linprog problem is infeasible"
        # 3: "Interior point calculation: scipy.optimize.linprog problem is unbounded"

        # if we failed while sampling all halfspaces, the linear program seems impossible
        assert initial_num_sampled < len(interior_hs), "get_interior_point problem is impossible or degenerate!"
        initial_num_sampled *= 2  # try again and sample more halfspaces this time

    return intpt


def CHAMP_2D(G, all_parts, gamma_0, gamma_f, single_threaded=False):
    r"""Calculates the pruned set of partitions from CHAMP on ``gamma_0`` :math:`\leq \gamma \leq` ``gamma_f``.

    See https://doi.org/10.3390/a10030093 for more details.

    :param G: graph of interest
    :type G: igraph.Graph
    :param all_parts: partitions to prune
    :type all_parts: list[tuple] or set[tuple]
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
    :rtype: list of tuple[float, float, tuple[int]]
    """

    # TODO: remove this filter once scipy updates their library
    # scipy.linprog currently uses deprecated numpy behavior, so we suppress this warning to avoid output clutter
    warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)

    if len(all_parts) == 0:
        return []

    all_parts = list(all_parts)
    num_partitions = len(all_parts)

    partition_coefficients = partition_coefficients_2D(G, all_parts, single_threaded=single_threaded)
    A_hats, P_hats = partition_coefficients

    top = max(A_hats - P_hats * gamma_0)  # Could potentially be optimized
    right = gamma_f  # Could potentially use the max intersection x value
    halfspaces = np.vstack((halfspaces_from_coefficients_2D(*partition_coefficients),
                            np.array([[0, 1, -top], [1, 0, -right]])))

    # Could potentially scale axes so Chebyshev center is better for problem
    interior_point = get_interior_point(halfspaces)
    hs = HalfspaceIntersection(halfspaces, interior_point)

    # scipy does not support facets by halfspace directly, so we must compute them
    facets_by_halfspace = defaultdict(list)
    for v, idx in zip(hs.intersections, hs.dual_facets):
        assert np.isfinite(v).all()
        for i in idx:
            if i < num_partitions:
                facets_by_halfspace[i].append(v)

    ranges = []
    for i, intersections in facets_by_halfspace.items():
        x1, x2 = intersections[0][0], intersections[1][0]
        if x1 > x2:
            x1, x2 = x2, x1
        ranges.append((x1, x2, all_parts[i]))

    return sorted(ranges, key=lambda x: x[0])


def CHAMP_3D(G_intralayer, G_interlayer, layer_vec, all_parts, gamma_0, gamma_f, omega_0, omega_f):
    r"""Calculates the pruned set of partitions from CHAMP on ``gamma_0`` :math:`\leq \gamma \leq` ``gamma_f`` and
    ``omega_0`` :math:`\leq \omega \leq` ``omega_f``.

    See https://doi.org/10.3390/a10030093 for more details.

    NOTE: This defers to the original CHAMP implementation for most of the halfspace intersection for now, so
    ``gamma_0`` and ``omega_0`` have no effect.

    :param G_intralayer: intralayer graph of interest
    :type G_intralayer: igraph.Graph
    :param G_interlayer: interlayer graph of interest
    :type G_interlayer: igraph.Graph
    :param layer_vec: list of each vertex's layer membership
    :type layer_vec: list[int]
    :param all_parts: partitions to prune
    :type all_parts: iterable[tuple]
    :param gamma_0: unused (should be the starting gamma value for CHAMP, but the original implementation seems to take this as equal to zero)
    :type gamma_0: float
    :param gamma_f: ending gamma value for CHAMP
    :type gamma_f: float
    :param omega_0: unused (should be the starting omega value for CHAMP, but the original implementation seems to take this as equal to zero)
    :type omega_0: float
    :param omega_f: ending omega value for CHAMP
    :type omega_f: float
    :return: list of tuples for the somewhere optimal partitions, containing (in-order)

        - list of polygon vertices in (gamma, omega) plane for the partition's domain of optimality
        - community membership tuple for the partition
    :rtype: list of tuple[list[float], tuple[int]]
    """

    all_parts = list(all_parts)
    A_hats, P_hats, C_hats = partition_coefficients_3D(G_intralayer, G_interlayer, layer_vec, all_parts)
    champ_coef_array = np.vstack((A_hats, P_hats, C_hats)).T

    for attempt in range(1, 10):
        try:
            champ_domains = get_intersection(champ_coef_array, max_pt=(omega_f, gamma_f))
            break
        except:  # noqa TODO: I think this is generally QhullError, but this needs to be checked
            continue
    else:
        # If this actually occurs, it's best to break your input partitions into smaller subsets
        # Then, repeatedly combine the somewhere dominant (or "admissible") domains with CHAMP
        assert False, "CHAMP failed, " \
                      "perhaps break your input partitions into smaller subsets and then combine with CHAMP?"

    domains = [([x[:2] for x in polyverts], all_parts[part_idx]) for part_idx, polyverts in champ_domains.items()]
    return domains


def partition_coefficients_2D_serial(G, partitions):
    """Computes A_hat and P_hat for partitions of :G:

    TODO: support edge weights"""

    all_edges = [(e.source, e.target) for e in G.es]

    # multiply by 2 only if undirected here
    if G.is_directed():
        A_hats = np.array([sum([membership[u] == membership[v] for u, v in all_edges])
                           for membership in partitions])
    else:
        A_hats = np.array([2 * sum([membership[u] == membership[v] for u, v in all_edges])
                           for membership in partitions])

    if G.is_directed():
        # directed modularity of Leicht and Newman is actually
        #   (1/m) sum_{ij} [A_{ij} - k_i^{in} * k_j^{out} / m] delta(c_i, c_j)
        in_degree = in_degrees(G)
        out_degree = out_degrees(G)
        P_hats = np.array([sum(sum(in_degree[v] for v in vs) * sum(out_degree[v] for v in vs)
                               for vs in membership_to_communities(membership).values())
                           for membership in partitions]) / G.ecount()
    else:
        twom = 2 * G.ecount()
        degree = all_degrees(G)
        P_hats = np.array([sum(sum(degree[v] for v in vs) ** 2 for vs in membership_to_communities(membership).values())
                           for membership in partitions]) / twom

    return A_hats, P_hats


def partition_coefficients_2D(G, partitions, single_threaded=False):
    """Computes partitions coefficients in parallel by calling partition_coefficients_2D_serial"""
    partitions = list(partitions)

    if single_threaded:
        A_hats, P_hats = partition_coefficients_2D_serial(G, partitions)
    else:
        partition_chunks = [
            partitions[floor(i * len(partitions) / cpu_count()):floor((i + 1) * len(partitions) / cpu_count())]
            for i in range(cpu_count())
        ]

        pool = Pool(processes=cpu_count())
        results = pool.starmap(partition_coefficients_2D_serial,
                               [(G, partition_chunk) for partition_chunk in partition_chunks])
        pool.close()

        A_hats = np.array([v for A_hats, P_hats in results for v in A_hats])
        P_hats = np.array([v for A_hats, P_hats in results for v in P_hats])

    assert len(A_hats) == len(P_hats) == len(partitions)

    return A_hats, P_hats


def halfspaces_from_coefficients_2D(A_hats, P_hats):
    """Converts partitions' coefficients to halfspace normal, offset

    Q >= -P_hat*gamma + A_hat
    -Q - P_hat*gamma + A_hat <= 0
    (-P_hat, -1) * (Q, gamma) + A_hat <= 0
    """
    return np.vstack((-P_hats, -np.ones_like(P_hats), A_hats)).T


def partition_coefficients_3D_serial(G_intralayer, G_interlayer, layer_vec, partitions):
    """Computes A_hat, P_hat, C_hat for partitions of a graph with intralayer edges given in :G_intralayer:,
    interlayer edges given in :G_interlayer:, and layer membership :layer_vec:

    TODO: support edge weights"""

    all_intralayer_edges = [(e.source, e.target) for e in G_intralayer.es]
    all_interlayer_edges = [(e.source, e.target) for e in G_interlayer.es]

    # multiply by 2 only if undirected here
    if G_intralayer.is_directed():
        A_hats = np.array([sum([membership[u] == membership[v] for u, v in all_intralayer_edges])
                           for membership in partitions])
    else:
        A_hats = np.array([2 * sum([membership[u] == membership[v] for u, v in all_intralayer_edges])
                           for membership in partitions])

    num_layers = max(layer_vec) + 1
    ecount_per_layer = np.zeros(num_layers)
    for e in G_intralayer.es:
        ecount_per_layer[layer_vec[e.source]] += 1

    if G_intralayer.is_directed():
        in_degree = in_degrees(G_intralayer)
        out_degree = out_degrees(G_intralayer)
        P_hats = np.array([
            sum(sum(in_degree[v] for v in vs) * sum(out_degree[v] for v in vs) / ecount_per_layer[layer]
                for (_, layer), vs in membership_to_layered_communities(membership, layer_vec).items())
            for membership in partitions
        ])
    else:
        degree = all_degrees(G_intralayer)
        P_hats = np.array([
            sum(sum(degree[v] for v in vs) ** 2 / (2 * ecount_per_layer[layer])
                for (_, layer), vs in membership_to_layered_communities(membership, layer_vec).items())
            for membership in partitions
        ])

    # multiply by 2 only if undirected here
    if G_interlayer.is_directed():
        C_hats = np.array([sum([membership[u] == membership[v] for u, v in all_interlayer_edges])
                           for membership in partitions])
    else:
        C_hats = np.array([2 * sum([membership[u] == membership[v] for u, v in all_interlayer_edges])
                           for membership in partitions])

    return A_hats, P_hats, C_hats


def partition_coefficients_3D(G_intralayer, G_interlayer, layer_vec, partitions):
    """Computes partitions coefficients in parallel by calling partition_coefficients_3D_serial"""
    partitions = list(partitions)
    partition_chunks = [
        partitions[floor(i * len(partitions) / cpu_count()):floor((i + 1) * len(partitions) / cpu_count())]
        for i in range(cpu_count())
    ]

    pool = Pool(processes=cpu_count())
    results = pool.starmap(partition_coefficients_3D_serial,
                           [(G_intralayer, G_interlayer, layer_vec, partition_chunk)
                            for partition_chunk in partition_chunks])
    pool.close()

    A_hats = np.array([v for A_hats, P_hats, C_hats in results for v in A_hats])
    P_hats = np.array([v for A_hats, P_hats, C_hats in results for v in P_hats])
    C_hats = np.array([v for A_hats, P_hats, C_hats in results for v in C_hats])

    assert len(A_hats) == len(P_hats) == len(C_hats) == len(partitions)

    return A_hats, P_hats, C_hats
