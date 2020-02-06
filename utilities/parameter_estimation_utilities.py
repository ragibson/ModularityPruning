from .louvain_utilities import num_communities
from math import log
import numpy as np
from scipy.optimize import fsolve


def gamma_estimate(G, partition):
    if 'weight' not in G.es:
        G.es['weight'] = [1.0] * G.vcount()

    m = G.ecount()
    m_in = sum(e['weight'] * (partition[e.source] == partition[e.target]) for e in G.es)
    kappa_r_list = [0] * len(partition)
    for e in G.es:
        kappa_r_list[partition[e.source]] += e['weight']
        kappa_r_list[partition[e.target]] += e['weight']
    sum_kappa_sqr = sum(x ** 2 for x in kappa_r_list)

    omega_in = (2 * m_in) / (sum_kappa_sqr / (2 * m))
    # guard for div by zero with single community partition
    omega_out = 0
    if num_communities(partition) > 1:
        omega_out = (2 * m - 2 * m_in) / (2 * m - sum_kappa_sqr / (2 * m))

    if omega_in == 0 or omega_in == 1:
        return None  # degenerate partition

    if omega_out == 0:
        return omega_in / np.log(omega_in)
    return (omega_in - omega_out) / (np.log(omega_in) - np.log(omega_out))


def gamma_omega_estimate(G_intralayer, G_interlayer, layer_vec, membership, omega_max=1000, model='temporal'):
    """Returns the (gamma, omega) estimate for a temporal network

    Relevant code copied from our parameter_estimation toolkit."""

    if 'weight' not in G_intralayer.es:
        G_intralayer.es['weight'] = [1.0] * G_intralayer.ecount()
    # else:
    #     G_intralayer.es['weight'] = [1.0] * G_intralayer.ecount()

    # TODO: non-uniform cases
    # model affects SBM parameter estimation and the updating of omega
    if model is 'temporal':
        def calculate_persistence(community):
            # ordinal persistence
            return sum(community[e.source] == community[e.target] for e in G_interlayer.es) / (N * (T - 1))
    elif model is 'multilevel':
        def calculate_persistence(community):
            # multilevel persistence
            pers_per_layer = [0] * T
            for e in G_interlayer.es:
                pers_per_layer[layer_vec[e.target]] += (community[e.source] == community[e.target])

            pers_per_layer = [pers_per_layer[l] / Nt[l] for l in range(T)]
            return sum(pers_per_layer) / (T - 1)
    elif model is 'multiplex':
        def calculate_persistence(community):
            # categorical persistence
            return sum(community[e.source] == community[e.target] for e in G_interlayer.es) / (N * T * (T - 1))
    else:
        raise ValueError("Model {} is not temporal, multilevel, or multiplex".format(model))

    def update_gamma(theta_in, theta_out):
        if theta_in == 0 or theta_in == 1:
            return None  # degenerate partition

        if theta_out == 0:
            return theta_in / log(theta_in)
        return (theta_in - theta_out) / (log(theta_in) - log(theta_out))

    if model is 'multiplex':
        def update_omega(theta_in, theta_out, p, K):
            if theta_out == 0:
                return log(1 + p * K / (1 - p)) / (T * log(theta_in)) if p < 1.0 and theta_in != 1 else omega_max
            # if p is 1, the optimal omega is infinite (here, omega_max)
            return log(1 + p * K / (1 - p)) / (T * (log(theta_in) - log(theta_out))) if p < 1.0 else omega_max
    else:
        def update_omega(theta_in, theta_out, p, K):
            if theta_out == 0:
                return log(1 + p * K / (1 - p)) / (2 * log(theta_in)) if p < 1.0 and theta_in != 1 else omega_max
            # if p is 1, the optimal omega is infinite (here, omega_max)
            return log(1 + p * K / (1 - p)) / (2 * (log(theta_in) - log(theta_out))) if p < 1.0 else omega_max

    T = max(layer_vec) + 1  # layer count
    m_t = [0] * T
    for e in G_intralayer.es:
        m_t[layer_vec[e.source]] += e['weight']

    N = G_intralayer.vcount() // T
    Nt = [0] * T
    for l in layer_vec:
        Nt[l] += 1

    K = num_communities(membership)

    m_t_in = [0] * T
    for e in G_intralayer.es:
        if membership[e.source] == membership[e.target] and layer_vec[e.source] == layer_vec[e.target]:
            m_t_in[layer_vec[e.source]] += e['weight']

    kappa_t_r_list = [[0] * K for _ in range(T)]
    for e in G_intralayer.es:
        layer = layer_vec[e.source]
        kappa_t_r_list[layer][membership[e.source]] += e['weight']
        kappa_t_r_list[layer][membership[e.target]] += e['weight']
    sum_kappa_t_sqr = [sum(x ** 2 for x in kappa_t_r_list[t]) for t in range(T)]

    theta_in = sum(2 * m_t_in[t] for t in range(T)) / sum(sum_kappa_t_sqr[t] / (2 * m_t[t]) for t in range(T))

    if any(sum_kappa_t_sqr[t] == (2 * m_t[t]) ** 2 for t in range(T)) != 0:
        # intralayer SBM is degenerate
        theta_out = 0
    else:
        # guard for div by zero with single community partition
        theta_out = sum(2 * m_t[t] - 2 * m_t_in[t] for t in range(T)) / \
                    sum(2 * m_t[t] - sum_kappa_t_sqr[t] / (2 * m_t[t]) for t in range(T)) if K > 1 else 0

    pers = calculate_persistence(membership)
    if model is 'multiplex':
        # estimate p by solving polynomial root-finding problem with starting estimate p=0.5
        def f(x):
            coeff = 2 * (1 - 1 / K) / (T * (T - 1))
            return coeff * sum((T - n) * x ** n for n in range(1, T)) + 1 / K - pers

        # guard for div by zero with single community partition
        # (in this case, all community assignments persist across layers)
        p = fsolve(f, np.array([0.5]))[0] if pers < 1.0 and K > 1 else 1.0
    else:
        # guard for div by zero with single community partition
        # (in this case, all community assignments persist across layers)
        p = max((K * pers - 1) / (K - 1), 0) if pers < 1.0 and K > 1 else 1.0

    gamma = update_gamma(theta_in, theta_out)
    omega = update_omega(theta_in, theta_out, p, K)
    return gamma, omega


def ranges_to_gamma_estimates(G, ranges):
    """Compute gamma estimates from ranges of dominance.

    Returns a list of [(gamma_start, gamma_end, membership, gamma_estimate), ...]"""

    return [(gamma_start, gamma_end, part, gamma_estimate(G, part)) for
            gamma_start, gamma_end, part in ranges]


def gamma_estimates_to_stable_partitions(gamma_estimates):
    """Computes the stable partitions from gamma estimates.

    Returns the memberships of the partitions where gamma_start <= gamma_estimate <= gamma_end."""
    return [membership for gamma_start, gamma_end, membership, gamma_estimate in gamma_estimates
            if gamma_start <= gamma_estimate <= gamma_end]


def domains_to_gamma_omega_estimates(G_intralayer, G_interlayer, layer_vec, domains, model='temporal'):
    """Compute (gamma, omega) estimates from domains of dominance.

    Returns a list of [(polygon vertices, membership, gamma_estimate, omega_estimate), ...]"""

    domains_with_estimates = []
    for polyverts, membership in domains:
        gamma_est, omega_est = gamma_omega_estimate(G_intralayer, G_interlayer, layer_vec, membership,
                                                    model=model)
        domains_with_estimates.append((polyverts, membership, gamma_est, omega_est))
    return domains_with_estimates


def gamma_omega_estimates_to_stable_partitions(domains_with_estimates):
    """Computes the stable partitions from (gamma, omega) estimates.

    Returns the memberships of the partitions where (gamma_estimate, omega_estimate) lies within the domain of
    optimality."""

    def left_or_right(x1, y1, x2, y2, x, y):
        """Returns whether the point (x,y) is to the left or right of the line between (x1, y1) and (x2, y2)."""
        return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1) >= 0

    stable_partitions = []

    for polyverts, membership, gamma_est, omega_est in domains_with_estimates:
        if gamma_est is None or omega_est is None:
            print(gamma_est, omega_est)
            continue

        centroid_x = np.mean([x[0] for x in polyverts])
        centroid_y = np.mean([x[1] for x in polyverts])
        polygon_edges = []
        for i in range(len(polyverts)):
            p1, p2 = polyverts[i], polyverts[(i + 1) % len(polyverts)]
            if left_or_right(p1[0], p1[1], p2[0], p2[1], centroid_x, centroid_y):
                p1, p2 = p2, p1
            polygon_edges.append((p1, p2))

        left_or_rights = []
        for p1, p2 in polygon_edges:
            left_or_rights.append(left_or_right(p1[0], p1[1], p2[0], p2[1], gamma_est, omega_est))

        if all(x for x in left_or_rights) or all(not x for x in left_or_rights):
            # if the (gamma, omega) estimate is on the same side of all polygon edges, it lies within the domain
            stable_partitions.append((polyverts, membership, gamma_est, omega_est))

    return stable_partitions
