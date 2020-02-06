from cProfile import run
import igraph as ig
import numpy as np
from utilities import *
import pickle
import sys

N = 300

p_in1 = 10 / 99  # 5/99 -> mean degree in community is 5
p_in2 = p_in1 * 0.75
p_out1 = 0.25 / 40  # 1/40 -> mean degree out community is 5
p_out2 = p_out1 * 3.75

filename = ("{:.4f}_" * 4).format(p_in1, p_in2, p_out1, p_out2)
print(filename)

pref_matrix = [[p_in1, p_out1, p_out1],
               [p_out1, p_in2, p_out2],
               [p_out1, p_out2, p_in2]]
block_sizes = [100] * 3
G = ig.Graph.SBM(N, pref_matrix, block_sizes)
assert G.is_connected()

print("mean degree is", np.mean([G.degree(v) for v in range(N)]))
GAMMA_START = 0.0  # 0.7
GAMMA_END = 2.0  # 1.25
ground_truth = tuple(i // 100 for i in range(N))
true_gamma = gamma_estimate(G, ground_truth)
ground_truth2 = tuple(min(1, i // 100) for i in range(N))
true_gamma2 = gamma_estimate(G, ground_truth2)
print("'true' gamma (3 block) is", true_gamma)
print("'true' gamma (2 block) is", true_gamma2)

ig.plot(louvain.RBConfigurationVertexPartition(G, ground_truth))
ig.plot(louvain.RBConfigurationVertexPartition(G, ground_truth2))

plt.close()
plot_adjacency(G.get_adjacency().data)
plt.savefig(filename + "adj.png", dpi=200)

# sys.exit(0)

all_parts = repeated_parallel_louvain(G, GAMMA_START, GAMMA_END, gamma_iters=125, repeat=4)
optimal_parts = manual_CHAMP(G, all_parts, GAMMA_START, GAMMA_END)
ranges = optimal_parts_to_ranges(optimal_parts)
gamma_estimates = ranges_to_gamma_estimates(G, ranges)

if ground_truth in all_parts:
    print("ground truth recovered")

for g_start, g_end, p in ranges:
    if p == ground_truth:
        print("ground truth recovered as optimal at gamma in [{:.3f}, {:.3f}]"
              "".format(g_start, g_end))

plt.close()
plot_estimates(gamma_estimates)
plt.axvline(true_gamma, color='red', alpha=0.5, linestyle='dashed', label="gamma for ground truth 3 block partition")
plt.axvline(true_gamma2, color='blue', alpha=0.5, linestyle='dashed', label="gamma for ground truth 2 block partition")
plt.legend()
plt.savefig(filename + "gamma_estimates.png", dpi=200)

for gamma_start, gamma_end, part in ranges:
    print("Partition optimal over gamma in [{:.3f},{:.3f}]:".format(gamma_start, gamma_end))
    print("  {} communities and AMI to 2-block: {:.3f}".format(num_communities(part), ami(ground_truth2, part)))
    print("  {} communities and AMI to 3-block: {:.3f}".format(num_communities(part), ami(ground_truth, part)))

for _ in range(10):
    print("*" * 10 + "RECORD THE AMI VALUES!!!" + "*" * 10)

pickle.dump((G, GAMMA_START, GAMMA_END,
             ground_truth, ground_truth2, true_gamma, true_gamma2,
             all_parts, optimal_parts, ranges, gamma_estimates), open(filename + "test.p", "wb"))
