import igraph as ig
from math import isnan
from modularitypruning.partition_utilities import num_communities
from random import randint, random, uniform


def generate_connected_ER(n, m, directed):
    G = ig.Graph.Erdos_Renyi(n=n, m=m, directed=directed)
    while not G.is_connected():
        G = ig.Graph.Erdos_Renyi(n=n, m=m, directed=directed)
    return G


def generate_connected_multilayer_ER(num_nodes_per_layer, m, num_layers, directed):
    total_num_nodes = num_nodes_per_layer * num_layers
    G = generate_connected_ER(n=total_num_nodes, m=m, directed=directed)
    layer_membership = [i // num_nodes_per_layer for i in range(total_num_nodes)]
    intralayer_edges = [(e.source, e.target) for e in G.es if layer_membership[e.source] == layer_membership[e.target]]
    interlayer_edges = [(e.source, e.target) for e in G.es if layer_membership[e.source] != layer_membership[e.target]]
    return (ig.Graph(n=total_num_nodes, edges=intralayer_edges, directed=directed),
            ig.Graph(n=total_num_nodes, edges=interlayer_edges, directed=directed),
            layer_membership)


def generate_random_values(num_values, start_value, end_value):
    return [uniform(start_value, end_value) for _ in range(num_values)]


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
    """Generate various famous graphs from igraph.

    In particular, we use
        Meredith (n=70, m=140): a counterexample to a conjecture regarding 4-regular 4-connected Hamiltonian graphs
        Nonline (n=50, m=72): a disconnnected graph composed of the 9 subgraphs whose presence makes a nonline graph
        Thomassen (n=34, m=52): the smallest graph without a Hamiltonian path
        Tutte (n=46, m=69): a counterexample to a conjecture regarding 3-connected 3-regular Hamiltonian graphs
        Zachary (n=34, m=78): popular network of the social interactions between 34 members of a karate club
    """

    return [ig.Graph.Famous(name) for name in ['meredith', 'nonline', 'thomassen', 'tutte', 'zachary']]


def generate_multilayer_intralayer_SBM(copying_probability, p_in, p_out, first_layer_membership, num_layers):
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

    for v in range(len(combined_community_labels)):
        for u in range(v + 1, len(combined_community_labels)):
            if layer_membership[v] == layer_membership[u]:
                if combined_community_labels[v] == combined_community_labels[u]:
                    if random() < p_in:
                        intralayer_edges.append((u, v))
                else:
                    if random() < p_out:
                        intralayer_edges.append((u, v))

    G_intralayer = ig.Graph(intralayer_edges, directed=False)

    return G_intralayer, layer_membership


def assert_almost_equal_or_both_none_or_nan(self, first, second, places=10):
    if first is None:
        self.assertIsNone(second)  # isnan only accepts floats
    elif isnan(first):
        self.assertTrue(isnan(second))  # NaN is != NaN according to IEEE-754
    else:
        self.assertAlmostEqual(first, second, places=places)
