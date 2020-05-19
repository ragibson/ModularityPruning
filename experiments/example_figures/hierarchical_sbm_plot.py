# Generates an example figure for hierarchical community structure used in a presentation

import igraph as ig
import louvain
import pickle
import os

if __name__ == "__main__":
    p1 = 1.0
    p2 = 0.2
    p3 = 0.05
    p4 = 0.005
    B = 7
    N = B * 8

    if not os.path.exists("community_scales_graph.p"):
        pref_matrix = [[p1, p2, p3, p3, p4, p4, p4, p4],
                       [p2, p1, p3, p3, p4, p4, p4, p4],
                       [p3, p3, p1, p2, p4, p4, p4, p4],
                       [p3, p3, p2, p1, p4, p4, p4, p4],
                       [p4, p4, p4, p4, p1, p2, p3, p3],
                       [p4, p4, p4, p4, p2, p1, p3, p3],
                       [p4, p4, p4, p4, p3, p3, p1, p2],
                       [p4, p4, p4, p4, p3, p3, p2, p1]]
        block_sizes = [B] * 8
        G = ig.Graph.SBM(N, pref_matrix, block_sizes)
        pickle.dump(G, open("community_scales_graph.p", "wb"))

    G = pickle.load(open("community_scales_graph.p", "rb"))
    layout = G.layout_fruchterman_reingold(niter=10000)

    membership0 = [0] * N
    membership1 = [0] * (4 * B) + [1] * (4 * B)
    membership2 = [0] * (2 * B) + [1] * (2 * B) + [2] * (2 * B) + [3] * (2 * B)
    membership3 = [i // B for i in range(N)]

    for i, m in enumerate([membership0, membership1, membership2, membership3]):
        out = ig.plot(louvain.RBConfigurationVertexPartition(G, initial_membership=m), f"community_scales{i}.png",
                      layout=layout, bbox=(600, 600))
