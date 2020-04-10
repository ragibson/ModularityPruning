import igraph as ig
import matplotlib.pyplot as plt
import louvain
import numpy as np
from modularitypruning.champ_utilities import CHAMP_2D
from modularitypruning.louvain_utilities import singlelayer_louvain
from modularitypruning.parameter_estimation_utilities import ranges_to_gamma_estimates
from modularitypruning.plotting import plot_estimates
from modularitypruning.progress import Progress
import sys

NUM_TRIALS = 3
progress = Progress(NUM_TRIALS)

for i in range(NUM_TRIALS):
    progress.increment()

    GAMMA_START = 0
    GAMMA_END = 3
    GAMMA_ITERS = 1000

    n = np.random.randint(10, 100)
    m = np.random.randint(3 * n, 10 * n)
    G = ig.Graph.Erdos_Renyi(n=n, m=m, directed=False)

    if not G.is_connected():
        continue

    gammas = np.linspace(GAMMA_START, GAMMA_END, GAMMA_ITERS)
    all_parts = [singlelayer_louvain(G, g) for g in gammas]

    if len(all_parts) == 0:
        continue

    # Print details on the CHAMP set when the number of communities is not restricted
    ranges = CHAMP_2D(G, all_parts, GAMMA_START, GAMMA_END)
    gamma_estimates = ranges_to_gamma_estimates(G, ranges)

    for i in range(len(gamma_estimates)):
        for j in range(len(gamma_estimates)):
            if i == j:
                continue

            start1, end1, membership1, estimate1 = gamma_estimates[i]
            start2, end2, membership2, estimate2 = gamma_estimates[j]

            if estimate1 is None or estimate2 is None:
                continue

            if start1 <= estimate2 <= end1 and start2 <= estimate1 <= end2:
                # if num_communities(membership1) > 8 or num_communities(membership2) > 8:
                #     continue

                progress.done()
                print(f"n={n}, m={m}")

                print([(e.source, e.target) for e in G.es])

                layout = G.layout_fruchterman_reingold(niter=1000)
                ig.plot(louvain.RBConfigurationVertexPartition(G, initial_membership=membership1),
                        bbox=(500, 500), layout=layout)
                ig.plot(louvain.RBConfigurationVertexPartition(G, initial_membership=membership2),
                        bbox=(500, 500), layout=layout)

                plot_estimates(gamma_estimates)
                plt.show()

                sys.exit(0)

    plt.close()
    plot_estimates(gamma_estimates)
    plt.title(f"$G(n={n}, m={m})$ Realization Domains and Estimates")
    plt.xlabel("$\gamma$", fontsize=14)
    plt.ylabel("Number of communities", fontsize=14)
    plt.xlim([-0.1, 3.1])
    plt.savefig(f"G({n},{m}).pdf")
