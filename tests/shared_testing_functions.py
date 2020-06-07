import igraph as ig
from random import randint, uniform


def generate_connected_ER(n, m, directed):
    G = ig.Graph.Erdos_Renyi(n=n, m=m, directed=directed)
    while not G.is_connected():
        G = ig.Graph.Erdos_Renyi(n=n, m=m, directed=directed)
    return G


def generate_random_gammas(num_gammas, gamma_start, gamma_end):
    return [uniform(gamma_start, gamma_end) for _ in range(num_gammas)]


def generate_random_partition(num_nodes, K):
    return tuple(randint(0, K - 1) for _ in range(num_nodes))


def generate_random_partitions(num_nodes, num_partitions, K_max):
    partitions = []
    for _ in range(num_partitions):
        K = randint(1, K_max)
        part = generate_random_partition(num_nodes, K)
        partitions.append(part)
    return partitions


def generate_igraph_famous():
    """Generate various famous graphs from igraph.

    In particular, we use
        Meredith (n=70, m=140): a counterexample to a conjecture regarding 4-regular 4-connected Hamiltonian graphs
        Nonline (n=50, m=72): a disconnnected graph composed of the 9 subgraphs whose presence makes a nonline graph
        Thomassen (n=34, m=52): the smallest graph without a Hamiltonian path
        Tutte (n=46, m=69): a counterexample to a conjecture regarding 3-connected 3-regular Hamiltonian graphs
        Zachary (n=34, m=78): popular network of the social interactions between 34 members of a karate club
    """

    return [ig.Graph.Famous(name) for name in ['meredith', 'nonline', 'thomassen', 'tutte', 'zachary']]
