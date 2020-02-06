# Generates figures 5.1 and 5.2

import igraph as ig
import louvain
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
import numpy as np
from utilities import repeated_parallel_louvain_from_gammas, CHAMP_2D, \
    ranges_to_gamma_estimates, num_communities, gamma_estimates_to_stable_partitions
from utilities import plot_estimates

GAMMA_START = 0.0
GAMMA_END = 2.0
GAMMA_ITERS = 100000

G = ig.Graph.Famous("Zachary")
gammas = np.linspace(GAMMA_START, GAMMA_END, GAMMA_ITERS)
pool = Pool(cpu_count())

all_parts = repeated_parallel_louvain_from_gammas(G, gammas, show_progress=True)

# Print details on the CHAMP sets when the number of communities is restricted
print(len(all_parts), "unique partitions in total")
for K in range(2, 9):
    restricted_parts = {p for p in all_parts if num_communities(p) == K}
    print("{} unique partitions with {} communities".format(len(restricted_parts), K))
    ranges = CHAMP_2D(G, restricted_parts, GAMMA_START, GAMMA_END)
    print("{} unique partitions in {}-community CHAMP set".format(len(ranges), K))
    print("=" * 50)

# Print details on the CHAMP set when the number of communities is not restricted
ranges = CHAMP_2D(G, all_parts, GAMMA_START, GAMMA_END)
gamma_estimates = ranges_to_gamma_estimates(G, ranges)
community_counts = [0] * 9
for _, _, membership, _ in gamma_estimates:
    community_counts[num_communities(membership)] += 1
for k, count in enumerate(community_counts):
    print("{} unique partitions with {} communities in total CHAMP set".format(count, k))

# Plot gamma estimates and domains of optimality when the number of communities is not restricted
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plot_estimates(gamma_estimates)
plt.title("Karate Club CHAMP Domains of Optimality and Gamma Estimates", fontsize=14)
plt.xlabel(r"$\gamma$", fontsize=14)
plt.ylabel("Number of communities", fontsize=14)
plt.savefig("karate_club_CHAMP_gamma_estimates.pdf")

# Store shared force-directed layout to make later plotting layouts consistent
layout = G.layout_fruchterman_reingold(maxiter=1000)

# Plot stable partitions when the number of communities is restricted
for K in range(2, 5):
    restricted_parts = {p for p in all_parts if num_communities(p) == K}

    if len(restricted_parts) > 0:
        print("Plotting stable partitions in CHAMP set with {} communities".format(K))
        ranges = CHAMP_2D(G, restricted_parts, GAMMA_START, GAMMA_END)
        gamma_estimates = ranges_to_gamma_estimates(G, ranges)
        stable_parts = gamma_estimates_to_stable_partitions(gamma_estimates)

        for p in stable_parts:
            print("Plotting {}...".format(p))
            out = ig.plot(louvain.RBConfigurationVertexPartition(G, initial_membership=p), bbox=(1000, 1000),
                          layout=layout)
            out.save("karate_club_{}_stable.png".format(K))
