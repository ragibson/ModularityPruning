from .louvain_utilities import singlelayer_louvain, multilayer_louvain
from .parameter_estimation_utilities import estimate_singlelayer_SBM_parameters, gamma_estimate_from_parameters
from .parameter_estimation_utilities import multiplex_omega_estimate_from_parameters, \
    temporal_multilevel_omega_estimate_from_parameters, ordinal_persistence, multilevel_persistence, \
    categorical_persistence
import louvain
import numpy as np
from scipy.optimize import fsolve
import warnings


def iterative_monolayer_resolution_parameter_estimation(G, gamma=1.0, tol=1e-2, max_iter=25, verbose=False,
                                                        method="louvain"):
    """
    Monolayer variant of ALG. 1 from "Relating modularity maximization and stochastic block models in multilayer
    networks." The nested functions here are just used to match the pseudocode in the paper.

    :param G: input graph
    :param gamma: starting gamma value
    :param tol: convergence tolerance
    :param max_iter: maximum number of iterations
    :param verbose: whether or not to print verbose output
    :param method: community detection method to use
    :return: gamma to which the iteration converged and the resulting partition
    """

    if 'weight' not in G.es:
        G.es['weight'] = [1.0] * G.ecount()
    m = sum(G.es['weight'])

    if method == "louvain":
        def maximize_modularity(resolution_param):
            return singlelayer_louvain(G, resolution_param, return_partition=True)
    elif method == "2-spinglass":
        def maximize_modularity(resolution_param):
            return G.community_spinglass(spins=2, gamma=resolution_param)
    else:
        raise ValueError(f"Community detection method {method} not supported")

    def estimate_SBM_parameters(partition):
        return estimate_singlelayer_SBM_parameters(G, partition, m=m)

    def update_gamma(omega_in, omega_out):
        return gamma_estimate_from_parameters(omega_in, omega_out)

    part, last_gamma = None, None
    for iteration in range(max_iter):
        part = maximize_modularity(gamma)
        omega_in, omega_out = estimate_SBM_parameters(part)

        last_gamma = gamma
        gamma = update_gamma(omega_in, omega_out)

        if gamma is None:
            raise ValueError(f"gamma={last_gamma:.3f} resulted in degenerate partition")

        if verbose:
            print(f"Iter {iteration:>2}: {len(part)} communities with Q={part.q:.3f} and "
                  f"gamma={last_gamma:.3f}->{gamma:.3f}")

        if abs(gamma - last_gamma) < tol:
            break  # gamma converged
    else:
        if verbose:
            print(f"Gamma failed to converge within {max_iter} iterations. "
                  f"Final move of {abs(gamma - last_gamma):.3f} was not within tolerance {tol}")

    if verbose:
        print(f"Returned {len(part)} communities with Q={part.q:.3f} and gamma={gamma:.3f}")

    return gamma, part


def check_multilayer_graph_consistency(G_intralayer, G_interlayer, layer_vec, model, m_t, T, N=None, Nt=None):
    """
    Checks that the structures of the intralayer and interlayer graphs are consistent and match the given model.

    :param G_intralayer: input graph containing all intra-layer edges
    :param G_interlayer: input graph containing all inter-layer edges
    :param layer_vec: vector of each vertex's layer membership
    :param model: network layer topology (temporal, multilevel, multiplex)
    :param m_t: vector of total edge weights per layer
    :param T: number of layers in input graph
    :param N: number of nodes per layer
    :param Nt: vector of nodes per layer
    """

    if G_intralayer.is_directed() != G_interlayer.is_directed():
        warnings.warn("Intralayer graph is {}, but Interlayer graph is {}."
                      "".format("directed" if G_intralayer.is_directed() else "undirected",
                                "directed" if G_interlayer.is_directed() else "undirected"),
                      RuntimeWarning)

    rules = [T > 1,
             "Graph must have multiple layers",
             G_interlayer.vcount() == G_intralayer.vcount(),
             "Inter-layer and Intra-layer graphs must be of the same size",
             len(layer_vec) == G_intralayer.vcount(),
             "Layer membership vector must have length matching graph size",
             all(m > 0 for m in m_t),
             "All layers of graph must contain edges",
             all(layer_vec[e.source] == layer_vec[e.target] for e in G_intralayer.es),
             "Intralayer graph should not contain edges across layers",
             model is not 'temporal' or G_interlayer.ecount() == N * (T - 1),
             "Interlayer temporal graph must contain (nodes per layer) * (number of layers - 1) edges",
             model is not 'temporal' or (G_interlayer.vcount() % T == 0 and G_intralayer.vcount() % T == 0),
             "Vertex count of a temporal graph should be a multiple of the number of layers",
             model is not 'temporal' or all(nt == N for nt in Nt),
             "Temporal networks must have the same number of nodes in every layer",
             model is not 'multilevel' or all(nt > 0 for nt in Nt),
             "All layers of a multilevel graph must be consecutive and nonempty",
             model is not 'multiplex' or all(nt == N for nt in Nt),
             "Multiplex networks must have the same number of nodes in every layer",
             model is not 'multiplex' or G_interlayer.ecount() == N * T * (T - 1),
             "Multiplex interlayer networks must contain edges between all pairs of layers"]

    checks, messages = rules[::2], rules[1::2]

    if not all(checks):
        raise ValueError("Input graph is malformed\n" + "\n".join(m for c, m in zip(checks, messages) if not c))


def iterative_multilayer_resolution_parameter_estimation(G_intralayer, G_interlayer, layer_vec, gamma=1.0, omega=1.0,
                                                         gamma_tol=1e-2, omega_tol=5e-2, omega_max=1000, max_iter=25,
                                                         model='temporal', verbose=False):
    """
    Multilayer variant of ALG. 1 from "Relating modularity maximization and stochastic block models in multilayer
    networks." The nested functions here are just used to match the pseudocode in the paper.

    :param G_intralayer: input graph containing all intra-layer edges
    :param G_interlayer: input graph containing all inter-layer edges
    :param layer_vec: vector of each vertex's layer membership
    :param gamma: starting gamma value
    :param omega: starting omega value
    :param gamma_tol: convergence tolerance for gamma
    :param omega_tol: convergence tolerance for omega
    :param max_iter: maximum number of iterations
    :param omega_max: maximum allowed value for omega
    :param model: network layer topology (temporal, multilevel, multiplex)
    :param verbose: whether or not to print verbose output
    :return: gamma, omega to which the iteration converged and the resulting partition
    """

    if 'weight' not in G_intralayer.es:
        G_intralayer.es['weight'] = [1.0] * G_intralayer.ecount()

    G_interlayer.es['weight'] = [omega] * G_interlayer.ecount()
    T = max(layer_vec) + 1  # layer count
    optimiser = louvain.Optimiser()
    m_t = [0] * T
    for e in G_intralayer.es:
        m_t[layer_vec[e.source]] += e['weight']

    N = G_intralayer.vcount() // T
    Nt = [0] * T
    for l in layer_vec:
        Nt[l] += 1

    check_multilayer_graph_consistency(G_intralayer, G_interlayer, layer_vec, model, m_t, T, N, Nt)

    if model is 'multiplex':
        def update_omega(theta_in, theta_out, p, K):
            return multiplex_omega_estimate_from_parameters(theta_in, theta_out, p, K, T, omega_max=omega_max)
    else:
        def update_omega(theta_in, theta_out, p, K):
            return temporal_multilevel_omega_estimate_from_parameters(theta_in, theta_out, p, K, omega_max=omega_max)

    # TODO: non-uniform cases
    # model affects SBM parameter estimation and the updating of omega
    if model is 'temporal':
        def calculate_persistence(community):
            return ordinal_persistence(G_interlayer, community)
    elif model is 'multilevel':
        def calculate_persistence(community):
            return multilevel_persistence(G_interlayer, community, layer_vec, Nt, T)
    elif model is 'multiplex':
        def calculate_persistence(community):
            return categorical_persistence(G_interlayer, community, N, T)
    else:
        raise ValueError("Model {} is not temporal, multilevel, or multiplex".format(model))

    def maximize_modularity(intralayer_resolution, interlayer_resolution):
        return multilayer_louvain(G_intralayer, G_interlayer, layer_vec, intralayer_resolution, interlayer_resolution,
                                  optimiser=optimiser, return_partition=True)

    def estimate_SBM_parameters(partition):
        # TODO: should this be removed in favor of the parameter_estimation_utilities function?
        K = len(partition)

        community = partition.membership
        m_t_in = [0] * T
        for e in G_intralayer.es:
            if community[e.source] == community[e.target] and layer_vec[e.source] == layer_vec[e.target]:
                m_t_in[layer_vec[e.source]] += e['weight']

        kappa_t_r_list = [[0] * K for _ in range(T)]
        for e in G_intralayer.es:
            layer = layer_vec[e.source]
            kappa_t_r_list[layer][community[e.source]] += e['weight']
            kappa_t_r_list[layer][community[e.target]] += e['weight']
        sum_kappa_t_sqr = [sum(x ** 2 for x in kappa_t_r_list[t]) for t in range(T)]

        theta_in = sum(2 * m_t_in[t] for t in range(T)) / sum(sum_kappa_t_sqr[t] / (2 * m_t[t]) for t in range(T))
        # guard for div by zero with single community partition
        theta_out = sum(2 * m_t[t] - 2 * m_t_in[t] for t in range(T)) / \
                    sum(2 * m_t[t] - sum_kappa_t_sqr[t] / (2 * m_t[t]) for t in range(T)) if K > 1 else 0

        pers = calculate_persistence(community)
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

        return theta_in, theta_out, p, K

    def update_gamma(theta_in, theta_out):
        return gamma_estimate_from_parameters(theta_in, theta_out)

    part, K, last_gamma, last_omega = (None,) * 4
    for iteration in range(max_iter):
        part = maximize_modularity(gamma, omega)
        theta_in, theta_out, p, K = estimate_SBM_parameters(part)

        if not 0.0 <= p <= 1.0:
            raise ValueError(f"gamma={gamma:.3f}, omega={omega:.3f} resulted in impossible estimate p={p:.3f}")

        last_gamma, last_omega = gamma, omega
        gamma = update_gamma(theta_in, theta_out)

        if gamma is None:
            raise ValueError(f"gamma={gamma:.3f}, omega={omega:.3f} resulted in degenerate partition")

        omega = update_omega(theta_in, theta_out, p, K)

        if verbose:
            print("Iter {:>2}: {} communities with Q={:.3f}, gamma={:.3f}->{:.3f}, omega={:.3f}->{:.3f}, and p={:.3f}"
                  "".format(iteration, K, part.q, last_gamma, gamma, last_omega, omega, p))

        if abs(gamma - last_gamma) < gamma_tol and abs(omega - last_omega) < omega_tol:
            break  # gamma and omega converged
    else:
        if verbose:
            print("Parameters failed to converge within {} iterations. "
                  "Final move of ({:.3f}, {:.3f}) was not within tolerance ({}, {})"
                  "".format(max_iter, abs(gamma - last_gamma), abs(omega - last_omega), gamma_tol, omega_tol))

    if verbose:
        print("Returned {} communities with Q={:.3f}, gamma={:.3f}, "
              "and omega={:.3f}".format(K, part.q, gamma, omega))

    return gamma, omega, part
