# TODO: combine this with boxplot.py for figure 6.4

import louvain
import numpy as np
import pickle
from time import time
from social_networks import read_graphs
from multiprocessing import Pool, cpu_count
from utilities import num_communities

Gs = read_graphs()
print("Import complete.")


def run_louvain(graphnum):
    G = Gs[graphnum]
    parts = []
    start = time()

    for gamma in np.linspace(0, 10, 1000):
        part = louvain.find_partition(G, louvain.RBConfigurationVertexPartition, resolution_parameter=gamma).membership

        if num_communities(part) > 100:
            break
        else:
            parts.append(part)

    print("Running on Graph {}, n={}, m={}: "
          "In {:.2f} s, found {} partitions at {:.2f} seconds per partition"
          "".format(graphnum, G.vcount(), G.ecount(), time() - start, len(parts), (time() - start) / len(parts)))
    return graphnum, parts


pool = Pool(cpu_count())
for graphnum, parts in pool.map(run_louvain, list(range(len(Gs)))):
    pickle.dump(parts, open("parts{}.p".format(graphnum), "wb"))
