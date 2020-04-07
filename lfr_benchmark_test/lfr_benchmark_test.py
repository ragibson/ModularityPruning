from networkx.generators.community import LFR_benchmark_graph
from networkx.exception import ExceededMaxIterations
from utilities import louvain_part_with_membership
import igraph as ig
import warnings


def LFR_benchmark_igraph(n, tau1, tau2, mu, average_degree, min_community):
    """Returns the LFR benchmark graph as discussed in "Benchmark graphs for testing community detection algorithms".

    We defer to the implementation in networkx here.

    Note real networks have typical values 2 < tau1 < 3 and 1 < tau2 < 2.

    :param n: number of nodes in the created graph
    :param tau1: power law exponent for the degree distribution
    :param tau2: power law exponent for the community size distribution
    :param mu: fraction of intra-community edges incident to each node
    :param average_degree: desired average degree of nodes in the created graph
    :param min_community: minimum size of communities in the graph
    :return: created (igraph) graph, ground truth community vector
    """

    while True:
        # we repeatedly generate graphs with the user-specified parameters until success
        # with poorly chosen values, this may take a very long time or never terminate
        try:
            G = LFR_benchmark_graph(n=n, tau1=tau1, tau2=tau2, mu=mu,
                                    average_degree=average_degree, min_community=min_community)
            communities = {frozenset(G.nodes[v]['community']) for v in G}
            break
        except ExceededMaxIterations:
            warnings.warn("LFR_benchmark generation failed. Retrying...")
            continue

    community_vector = [None for _ in range(G.number_of_nodes())]
    for i, community_set in enumerate(communities):
        assert len(community_set) > 0
        for v in community_set:
            community_vector[v] = i

    for i, community_set in enumerate(communities):
        for v in community_set:
            assert community_vector[v] == i

    G_igraph = ig.Graph(G.edges, directed=G.is_directed())
    assert not G.is_directed()
    assert all(c is not None for c in community_vector)
    assert G.number_of_nodes() == G_igraph.vcount()
    assert G.number_of_edges() == G_igraph.ecount()

    return G_igraph, community_vector


if __name__ == "__main__":
    G, community = LFR_benchmark_igraph(n=1000, tau1=2.5, tau2=1.25, mu=0.1, average_degree=5, min_community=20)
    part = louvain_part_with_membership(G, community)
    layout = G.layout_fruchterman_reingold(niter=1000)
    out = ig.plot(part, bbox=(1000, 1000), layout=layout)
