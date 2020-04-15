import igraph as ig
from modularitypruning.champ_utilities import CHAMP_2D
from modularitypruning.louvain_utilities import louvain_part_with_membership, sorted_tuple
from modularitypruning.parameter_estimation_utilities import gamma_estimate, ranges_to_gamma_estimates
from modularitypruning.plotting import plot_estimates
from random import randint
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    while True:
        G = ig.Graph.Erdos_Renyi(n=10, m=20)
        while not G.is_connected():
            G = ig.Graph.Erdos_Renyi(n=10, m=20)

        p1 = sorted_tuple(tuple(randint(0, 2) for _ in range(G.vcount())))
        p2 = sorted_tuple(tuple(randint(0, 2) for _ in range(G.vcount())))

        g1 = gamma_estimate(G, p1)
        g2 = gamma_estimate(G, p2)

        if g1 is None or g2 is None or np.isnan(g1) or np.isnan(g2):
            continue

        part1 = louvain_part_with_membership(G, p1)
        part2 = louvain_part_with_membership(G, p2)

        if part1.quality(resolution_parameter=g2) > part2.quality(resolution_parameter=g2):
            if part2.quality(resolution_parameter=g1) > part1.quality(resolution_parameter=g1):
                layout = G.layout_fruchterman_reingold(maxiter=1000)
                ig.plot(part1, layout=layout)
                ig.plot(part2, layout=layout)

                ranges = CHAMP_2D(G, [p1, p2], 0, 2)
                gamma_estimates = ranges_to_gamma_estimates(G, ranges)
                plot_estimates(gamma_estimates)
                plt.show()
                break
