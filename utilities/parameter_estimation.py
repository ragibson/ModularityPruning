from .leiden_utilities import singlelayer_leiden, multilayer_leiden
from .parameter_estimation_utilities import leiden_part_with_membership, estimate_singlelayer_SBM_parameters, \
    gamma_estimate_from_parameters, omega_function_from_model, estimate_multilayer_SBM_parameters
from .partition_utilities import in_degrees
import leidenalg


def iterative_monolayer_resolution_parameter_estimation(G, gamma=1.0, tol=1e-2, max_iter=25, verbose=False,
                                                        method="leiden"):
    """Monolayer variant of ALG. 1 from "Relating modularity maximization and stochastic block models in multilayer
    networks." This is intended to determine an "optimal" value for gamma by repeatedly maximizing modularity and
    estimating new values for the resolution parameter.

    See https://doi.org/10.1137/18M1231304 for more details.

    :param G: graph of interest
    :type G: igraph.Graph
    :param gamma: initialization gamma value
    :type gamma: float
    :param tol: convergence tolerance
    :type tol: float
    :param max_iter: maximum number of iterations
    :type max_iter: int
    :param verbose: whether or not to print verbose output
    :type verbose: bool
    :param method: community detection method to use
    :type method: str
    :return:
        - gamma to which the iteration converged
        - the resulting partition
    :rtype: tuple[float, leidenalg.RBConfigurationVertexPartition]
    """

    if 'weight' not in G.es:
        G.es['weight'] = [1.0] * G.ecount()
    m = sum(G.es['weight'])

    if method == "louvain" or method == "leiden":
        def maximize_modularity(resolution_param):
            return singlelayer_leiden(G, resolution_param, return_partition=True)
    elif method == "2-spinglass":
        def maximize_modularity(resolution_param):
            membership = G.community_spinglass(spins=2, gamma=resolution_param).membership
            return leiden_part_with_membership(G, membership)
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

    rules = [T > 1,
             "Graph must have multiple layers",
             G_interlayer.is_directed(),
             "Interlayer graph should be directed",
             G_interlayer.vcount() == G_intralayer.vcount(),
             "Inter-layer and Intra-layer graphs must be of the same size",
             len(layer_vec) == G_intralayer.vcount(),
             "Layer membership vector must have length matching graph size",
             all(m > 0 for m in m_t),
             "All layers of graph must contain edges",
             all(layer_vec[e.source] == layer_vec[e.target] for e in G_intralayer.es),
             "Intralayer graph should not contain edges across layers",
             model != 'temporal' or G_interlayer.ecount() == N * (T - 1),
             "Interlayer temporal graph must contain (nodes per layer) * (number of layers - 1) edges",
             model != 'temporal' or (G_interlayer.vcount() % T == 0 and G_intralayer.vcount() % T == 0),
             "Vertex count of a temporal graph should be a multiple of the number of layers",
             model != 'temporal' or all(nt == N for nt in Nt),
             "Temporal networks must have the same number of nodes in every layer",
             model != 'multilevel' or all(nt > 0 for nt in Nt),
             "All layers of a multilevel graph must be consecutive and nonempty",
             model != 'multilevel' or all(in_degree <= 1 for in_degree in in_degrees(G_interlayer)),
             "Multilevel networks should have at most one interlayer in-edge per node",
             model != 'multiplex' or all(nt == N for nt in Nt),
             "Multiplex networks must have the same number of nodes in every layer",
             model != 'multiplex' or G_interlayer.ecount() == N * T * (T - 1),
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

    :param G_intralayer: intralayer graph of interest
    :type G_intralayer: igraph.Graph
    :param G_interlayer: interlayer graph of interest
    :type G_interlayer: igraph.Graph
    :param layer_vec: list of each vertex's layer membership
    :type layer_vec: list[int]
    :param gamma: starting gamma value
    :type gamma: float
    :param omega: starting omega value
    :type omega: float
    :param gamma_tol: convergence tolerance for gamma
    :type gamma_tol: float
    :param omega_tol: convergence tolerance for omega
    :type omega_tol: float
    :param omega_max: maximum allowed value for omega
    :type omega_max: float
    :param max_iter: maximum number of iterations
    :type max_iter: int
    :param model: network layer topology (temporal, multilevel, multiplex)
    :type model: str
    :param verbose: whether or not to print verbose output
    :type verbose: bool
    :return:
        - gamma to which the iteration converged
        - omega to which the iteration converged
        - the resulting partition
    :rtype: tuple[float, float, tuple[int]]
    """

    global parts
    if 'weight' not in G_intralayer.es:
        G_intralayer.es['weight'] = [1.0] * G_intralayer.ecount()

    if 'weight' not in G_interlayer.es:
        G_interlayer.es['weight'] = [1.0] * G_interlayer.ecount()

    T = max(layer_vec) + 1  # layer count
    optimiser = leidenalg.Optimiser()

    # compute total edge weights per layer
    m_t = [0] * T
    for e in G_intralayer.es:
        m_t[layer_vec[e.source]] += e['weight']

    # compute total node counts per layer
    N = G_intralayer.vcount() // T
    Nt = [0] * T
    for layer in layer_vec:
        Nt[layer] += 1

    check_multilayer_graph_consistency(G_intralayer, G_interlayer, layer_vec, model, m_t, T, N, Nt)
    update_omega = omega_function_from_model(model, omega_max, T=T)
    update_gamma = gamma_estimate_from_parameters

    def maximize_modularity(intralayer_resolution, interlayer_resolution):
        return multilayer_leiden(G_intralayer, G_interlayer, layer_vec, intralayer_resolution, interlayer_resolution,
                                 optimiser=optimiser, return_partition=True)

    def estimate_SBM_parameters(partition):
        return estimate_multilayer_SBM_parameters(G_intralayer, G_interlayer, layer_vec, partition, model,
                                                  N=N, T=T, Nt=Nt, m_t=m_t)

    part, K, last_gamma, last_omega = (None,) * 4
    for iteration in range(max_iter):
        parts = maximize_modularity(gamma, omega)
        theta_in, theta_out, p, K = estimate_SBM_parameters(parts[0])

        if not 0.0 <= p <= 1.0:
            raise ValueError(f"gamma={gamma:.3f}, omega={omega:.3f} resulted in impossible estimate p={p:.3f}")

        last_gamma, last_omega = gamma, omega
        gamma = update_gamma(theta_in, theta_out)

        if gamma is None:
            raise ValueError(f"gamma={last_gamma:.3f}, omega={last_omega:.3f} resulted in degenerate partition")

        omega = update_omega(theta_in, theta_out, p, K)

        if verbose:
            part_Q = sum(p.q for p in parts)
            print(f"Iter {iteration:>2}: {K} communities with Q={part_Q:.3f}, gamma={last_gamma:.3f}->{gamma:.3f}, "
                  f"omega={last_omega:.3f}->{omega:.3f}, and p={p:.3f}")

        if abs(gamma - last_gamma) < gamma_tol and abs(omega - last_omega) < omega_tol:
            break  # gamma and omega converged
    else:
        if verbose:
            print(f"Parameters failed to converge within {max_iter} iterations. "
                  f"Final move of ({abs(gamma - last_gamma):.3f}, {abs(omega - last_omega):.3f}) "
                  f"was not within tolerance ({gamma_tol}, {omega_tol})")

    if verbose:
        part_Q = sum(p.q for p in parts)
        print(f"Returned {K} communities with Q={part_Q:.3f}, gamma={gamma:.3f}, and omega={omega:.3f}")

    return gamma, omega, parts
