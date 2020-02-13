from champ import create_coefarray_from_partitions, get_intersection
from champ import plot_2d_domains
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utilities import partition_coefficients_3D, all_degrees, membership_to_communities, CHAMP_3D
from champ.louvain_ext import get_expected_edges
import igraph as ig
import louvain
from time import time
from cProfile import run

G_intralayer, G_interlayer, layer_vec, champ_parts = pickle.load(open("CHAMP_3D_test.p", "rb"))

champ_parts = list(set(tuple(p) for p in champ_parts))

layer_vec = np.array(layer_vec)
nlayers = max(layer_vec) + 1
n = sum(layer_vec == 0)
champ_parts = np.array(champ_parts)

A = np.array(G_intralayer.get_adjacency().data)
C = np.array(G_interlayer.get_adjacency().data)
P = np.zeros((nlayers * n, nlayers * n))
for i in range(nlayers):
    c_degrees = np.array(G_intralayer.degree(list(range(n * i, n * i + n))))
    c_inds = np.where(layer_vec == i)[0]
    P[np.ix_(c_inds, c_inds)] = np.outer(c_degrees, c_degrees.T) / (1.0 * np.sum(c_degrees))

start = time()
coefarray = create_coefarray_from_partitions(champ_parts, A, P, C)
# A_hats, P_hats, C_hats = partition_coefficients_3D(G_intralayer, G_interlayer, layer_vec, champ_parts)
# our_coefarray = np.vstack((A_hats, P_hats, C_hats)).T
domains = get_intersection(coefarray, max_pt=(1.0, 1.0))
print("CHAMP's implementation took {:.2f} s".format(time() - start))

start = time()
domains = CHAMP_3D(G_intralayer, G_interlayer, layer_vec, champ_parts.tolist(), 0.0, 1.0, 0.0, 1.0)
print("Our implementation took {:.2f} s".format(time() - start))
