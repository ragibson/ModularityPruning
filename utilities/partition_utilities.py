from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import igraph
from collections import defaultdict

from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

from .types import IntSequence


def ami(p1: IntSequence, p2: IntSequence) -> float:
    return float(adjusted_mutual_info_score(p1, p2))


def nmi(p1: IntSequence, p2: IntSequence) -> float:
    return float(normalized_mutual_info_score(p1, p2, average_method='arithmetic'))


def all_degrees(G: "igraph.Graph") -> list[int]:
    return list(G.degree())


def in_degrees(G: "igraph.Graph") -> list[int]:
    return list(G.indegree())


def out_degrees(G: "igraph.Graph") -> list[int]:
    return list(G.outdegree())


def all_strengths(G: "igraph.Graph") -> list[float]:
    return list(G.strength(weights='weight' if G.is_weighted() else None))


def in_strengths(G: "igraph.Graph") -> list[float]:
    return list(G.strength(mode='in', weights='weight' if G.is_weighted() else None))


def out_strengths(G: "igraph.Graph") -> list[float]:
    return list(G.strength(mode='out', weights='weight' if G.is_weighted() else None))


def membership_to_communities(membership: IntSequence) -> dict[int, list[int]]:
    communities: defaultdict[int, list[int]] = defaultdict(list)
    for v, c in enumerate(membership):
        communities[int(c)].append(v)
    return communities


def membership_to_layered_communities(membership: IntSequence, layer_membership: IntSequence
                                      ) -> defaultdict[tuple[int, int], list[int]]:
    layered_communities: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for v, c in enumerate(membership):
        layered_communities[(int(c), int(layer_membership[v]))].append(v)
    return layered_communities


def num_communities(membership: IntSequence) -> int:
    n: int = len(set(membership))
    assert n == max(membership) + 1
    return n
