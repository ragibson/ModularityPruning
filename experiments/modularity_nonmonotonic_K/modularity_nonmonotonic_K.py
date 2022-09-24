"""
This script demonstrates that modularity is not monotonic in the number of
communities as gamma increases.

Indeed, it is possible for the CHAMP domains / most optimal partitions to
temporarily move to a higher number of communities and then drop down to a
smaller number at an even higher choice of the resolution parameter.

In fact, if you rewrite modularity as on Wikipedia,
    Q = 1/(2m) * sum_{ij} [A_{ij} - k_i*k_j/(2m)]*delta(c_i, c_j)
      = sum_{i=1}^c (e_{ij} - a_i^2)
where
    e_{ij} = "the fraction of edges with one end vertices in community i and
              the other in community j"
           = sum_{ij} A_{ij}/(2m)
    a_i = "the fraction of ends of edges that are attached to vertices in
           community i"
        = k_i/(2m) = sum_j e_{ij}
You can see that gamma essentially applies a penalty to the group sum of
degrees. So, for example, you can get small communities that push the number
of communities temporarily higher.

Note that they're using standard modularity (i.e. gamma=1) here.
"""
import leidenalg
import igraph as ig
import matplotlib.pyplot as plt
from modularitypruning.champ_utilities import CHAMP_2D
from modularitypruning.parameter_estimation_utilities import ranges_to_gamma_estimates
from modularitypruning.partition_utilities import num_communities
from modularitypruning.plotting import plot_estimates
import numpy as np


def plot_and_exit(G, ranges):
    layout = ig.Graph.layout_fruchterman_reingold(G, niter=1000)
    print("plotting network...")
    ig.plot(G, layout=layout).save("network.png")

    print("plotting CHAMP domains and gamma estimates...")
    gamma_estimates = ranges_to_gamma_estimates(G, ranges)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_estimates(gamma_estimates)
    plt.title(r"CHAMP Domains of Optimality and $\gamma$ Estimates", fontsize=14)
    plt.xlabel(r"$\gamma$", fontsize=14)
    plt.ylabel("Number of communities", fontsize=14)
    plt.savefig("estimates.png")

    for idx, r in enumerate(ranges):
        part = r[-1]
        print(f"plotting dominant partition {idx} (K={num_communities(part)})...")
        la_part = leidenalg.RBConfigurationVertexPartition(G, initial_membership=part)
        ig.plot(la_part, layout=layout).save(f"partition{idx}.png")
    exit()


for trial_idx in range(10 ** 6):
    # just look for small examples for ease of explanation
    n = np.random.randint(7, 20)
    m = np.random.randint(n, 3 * n)
    print(f"trial {trial_idx:<10} G(n={n}, m={m})")
    G = ig.Graph.Erdos_Renyi(n=n, m=m)

    if not G.is_connected():
        continue

    # note that this experiment uses leidenalg rather than louvain
    # it also uses gamma in [0, 1] to keep the number of communities small
    parts = [
        leidenalg.find_partition(G, partition_type=leidenalg.RBConfigurationVertexPartition,
                                 resolution_parameter=gamma).membership
        for gamma in np.linspace(0, 1, 10 ** 3)
    ]
    ranges = CHAMP_2D(G, parts, 0, 1)
    Ks = [num_communities(r[-1]) for r in ranges]

    for i in range(len(Ks) - 1):
        # search for K1 -> K2 -> K3 with K3 <= K1 < K2. E.g.
        #         ---K2---
        # ---K1---        ---K3--- ...
        # or K1 -> K2 -> K3 -> K4 with K4 <= K1 < K2 < K3. E.g.
        #                 ---K3---
        #         ---K2---
        # ---K1---               ---K4--- ...
        if Ks[i + 1] < Ks[i]:
            print("Found non-monotonic K example! "
                  f"(K={Ks[i]} -> K={Ks[i + 1]})")
            plot_and_exit(G, ranges)
