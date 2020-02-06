import igraph as ig
from utilities import *

N = 600
B = N // 3
p_in1 = 10 / 99
p_in2 = p_in1 * 0.75  # 5/66
p_out1 = 0.25 / 40  # 1/160

for i, p_out2 in enumerate([0.035]):  # delta
    filename = ("{:.4f}_" * 4).format(p_in1, p_in2, p_out1, p_out2)
    print(filename)

    pref_matrix = [[p_in1, p_out1, p_out1],
                   [p_out1, p_in2, p_out2],
                   [p_out1, p_out2, p_in2]]
    block_sizes = [B] * 3
    G = ig.Graph.SBM(N, pref_matrix, block_sizes)
    assert G.is_connected()

    ground_truth = tuple(i // B for i in range(N))
    true_gamma = gamma_estimate(G, ground_truth)
    ground_truth2 = tuple(min(1, i // B) for i in range(N))
    true_gamma2 = gamma_estimate(G, ground_truth2)
    # print("mean degree is", np.mean([G.degree(v) for v in range(N)]))
    # print("'true' gamma (3 block) is", true_gamma)
    # print("'true' gamma (2 block) is", true_gamma2)

    # Store shared force-directed layout to make later plotting layouts consistent
    layout = G.layout_fruchterman_reingold(maxiter=1000)

    out2 = ig.plot(louvain.RBConfigurationVertexPartition(G, ground_truth), bbox=(1000, 1000), layout=layout)
    out2.save("bistable_sbm_delta{}_2-community.png".format(i))
    out3 = ig.plot(louvain.RBConfigurationVertexPartition(G, ground_truth2), bbox=(1000, 1000), layout=layout)
    out3.save("bistable_sbm_delta{}_3-community.png".format(i))
