from collections import defaultdict
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score


def ami(p1, p2):
    return adjusted_mutual_info_score(p1, p2)


def nmi(p1, p2):
    return normalized_mutual_info_score(p1, p2, average_method='arithmetic')


def all_degrees(G):
    return G.degree()


def in_degrees(G):
    return G.indegree()


def out_degrees(G):
    return G.outdegree()


def membership_to_communities(membership):
    communities = defaultdict(list)
    for v, c in enumerate(membership):
        communities[c].append(v)
    return communities


def membership_to_layered_communities(membership, layer_membership):
    layered_communities = defaultdict(list)
    for v, c in enumerate(membership):
        layered_communities[(c, layer_membership[v])].append(v)
    return layered_communities


def num_communities(membership):
    n = len(set(membership))
    assert n == max(membership) + 1
    return n
