from math import isnan
from random import randint, random, uniform

import igraph as ig
from modularitypruning.partition_utilities import num_communities


def generate_random_values(num_values, start_value, end_value):
    return [uniform(start_value, end_value) for _ in range(num_values)]


def generate_connected_ER(n, m, directed, weighted=False, low=0.1, high=10.0):
    G = ig.Graph.Erdos_Renyi(n=n, m=m, directed=directed)
    while not G.is_connected():
        G = ig.Graph.Erdos_Renyi(n=n, m=m, directed=directed)
    if weighted:
        G.es['weight'] = generate_random_values(G.ecount(), low, high)
    return G


def generate_connected_multilayer_ER(num_nodes_per_layer, m, num_layers, directed, weighted=False, low=0.1, high=10.0):
    total_num_nodes = num_nodes_per_layer * num_layers
    G = generate_connected_ER(n=total_num_nodes, m=m, directed=directed)
    layer_membership = [i // num_nodes_per_layer for i in range(total_num_nodes)]
    intralayer_edges = [(e.source, e.target) for e in G.es if layer_membership[e.source] == layer_membership[e.target]]
    interlayer_edges = [(e.source, e.target) for e in G.es if layer_membership[e.source] != layer_membership[e.target]]
    G_intralayer = ig.Graph(n=total_num_nodes, edges=intralayer_edges, directed=directed)
    G_interlayer = ig.Graph(n=total_num_nodes, edges=interlayer_edges, directed=directed)
    if weighted:
        G_intralayer.es['weight'] = generate_random_values(G_intralayer.ecount(), low, high)
        G_interlayer.es['weight'] = generate_random_values(G_interlayer.ecount(), low, high)
    return G_intralayer, G_interlayer, layer_membership


def generate_random_partition(num_nodes, K):
    partition = tuple(randint(0, K - 1) for _ in range(num_nodes))
    while len(set(partition)) != K:
        partition = tuple(randint(0, K - 1) for _ in range(num_nodes))
    return partition


def generate_random_partitions(num_nodes, num_partitions, K_max):
    partitions = []
    for _ in range(num_partitions):
        K = randint(1, K_max)
        part = generate_random_partition(num_nodes, K)
        partitions.append(part)
    return partitions


def generate_igraph_famous():
    """Generate various famous graphs from igraph as (name, graph) pairs.

    In particular, we use
        Meredith (n=70, m=140): a counterexample to a conjecture regarding 4-regular 4-connected Hamiltonian graphs
        Nonline (n=50, m=72): a disconnnected graph composed of the 9 subgraphs whose presence makes a nonline graph
        Thomassen (n=34, m=52): the smallest graph without a Hamiltonian path
        Tutte (n=46, m=69): a counterexample to a conjecture regarding 3-connected 3-regular Hamiltonian graphs
        Zachary (n=34, m=78): popular network of the social interactions between 34 members of a karate club
    """

    return [(name, ig.Graph.Famous(name)) for name in ['meredith', 'nonline', 'thomassen', 'tutte', 'zachary']]


def generate_connected_multilayer_intralayer_SBM(copying_probability, p_in, p_out, first_layer_membership, num_layers,
                                                 weighted=False, low=0.5, high=5.0, directed=False):
    num_nodes_per_layer = len(first_layer_membership)
    community_labels_per_layer = [[0] * num_nodes_per_layer for _ in range(num_layers)]
    community_labels_per_layer[0] = list(first_layer_membership)
    K = num_communities(first_layer_membership)

    # assign community labels in the higher layers
    for layer in range(1, num_layers):
        for v in range(num_nodes_per_layer):
            if random() < copying_probability:  # copy community from last layer
                community_labels_per_layer[layer][v] = community_labels_per_layer[layer - 1][v]
            else:  # assign random community
                community_labels_per_layer[layer][v] = randint(0, K - 1)

    # create intralayer edges according to an SBM
    intralayer_edges = []
    combined_community_labels = sum(community_labels_per_layer, [])
    layer_membership = [i for i in range(num_layers) for _ in range(num_nodes_per_layer)]

    if directed:
        edges_per_layer = [(v, u) for v in range(len(combined_community_labels))
                           for u in range(len(combined_community_labels))
                           if layer_membership[v] == layer_membership[u] and v != u]
    else:
        edges_per_layer = [(v, u) for v in range(len(combined_community_labels))
                           for u in range(v + 1, len(combined_community_labels))
                           if layer_membership[v] == layer_membership[u]]

    for v, u in edges_per_layer:
        if combined_community_labels[v] == combined_community_labels[u]:
            if random() < p_in:
                intralayer_edges.append((v, u))
        else:
            if random() < p_out:
                intralayer_edges.append((v, u))

    # force disconnected intralayer nodes to have at least one edge
    intralayer_degrees = [0] * len(combined_community_labels)
    for u, v in intralayer_edges:
        intralayer_degrees[u] += 1
        intralayer_degrees[v] += 1
    for v, degree in enumerate(intralayer_degrees):
        if degree == 0:
            same_layer = [u for u in range(len(combined_community_labels))
                          if layer_membership[u] == layer_membership[v] and u != v]
            intralayer_edges.append((v, same_layer[randint(0, len(same_layer) - 1)]))

    G_intralayer = ig.Graph(n=len(combined_community_labels), edges=intralayer_edges, directed=directed)

    if weighted:
        G_intralayer.es['weight'] = generate_random_values(G_intralayer.ecount(), low, high)

    return G_intralayer, layer_membership


def generate_SBM(q, directed=False, weighted=False, low=0.5, high=5.0):
    """Generate a planted-partition SBM, optionally directed and/or weighted.

    This is a setup analogous to FIG 1 of Newman's 'Equivalence between modularity optimization and maximum likelihood
    methods for community detection'.
    """
    community_sizes = [250] * q
    n = sum(community_sizes)
    p_in = 16 * n / (q * 250 * 249)  # ~16 in-edges per node
    p_out = 8 * n / (q * (q - 1) * 250 * 250)  # ~8 out-edges per node, spread over the other communities
    pref_matrix = [[p_in if i == j else p_out for j in range(q)] for i in range(q)]
    G = ig.Graph.SBM(pref_matrix, community_sizes, directed=directed)

    if weighted:
        G.es['weight'] = generate_random_values(G.ecount(), low, high)

    return G, p_in, p_out


def assert_almost_equal_or_both_none_or_nan(self, first, second, places=10):
    if first is None:
        self.assertIsNone(second)  # isnan only accepts floats
    elif isnan(first):
        self.assertTrue(isnan(second))  # NaN is != NaN according to IEEE-754
    else:
        self.assertAlmostEqual(first, second, places=places)
