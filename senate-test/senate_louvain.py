import numpy as np
import champ
import scipy.io as scio
from utilities import repeated_parallel_louvain_from_gammas_omegas
import pickle

# def run_champ_implementation():
#     sendata = scio.loadmat('/home/ryan/PycharmProjects/mini-CHAMP-multilayer/multisenate0.5.mat')
#
#     A = sendata['A']
#     C = sendata['C']
#     sesid = sendata['Ssess'][:, 0]
#     parties = sendata['Sparty'][:, 0]
#     sessions = np.unique(sesid)
#     sess2layer = dict(zip(sessions, range(len(sessions))))
#     layer_vec = np.array(list(map(lambda x: sess2layer[x], sesid)))
#     intralayer, interlayer = champ.create_multilayer_igraph_from_adjacency(A=A, C=C, layer_vec=layer_vec)
#     parts = champ.parallel_multilayer_louvain_from_adj(intralayer_adj=A, interlayer_adj=C,
#                                                        layer_vec=layer_vec, numprocesses=8,
#                                                        gamma_range=[1, 2], omega_range=[0, 2],
#                                                        ngamma=2, nomega=4, maxpt=(2, 2))

sendata = scio.loadmat('/home/ryan/PycharmProjects/mini-CHAMP-multilayer/multisenate0.5.mat')
A = sendata['A']
C = sendata['C']
sesid = sendata['Ssess'][:, 0]
parties = sendata['Sparty'][:, 0]
sessions = np.unique(sesid)
sess2layer = dict(zip(sessions, range(len(sessions))))
layer_vec = np.array(list(map(lambda x: sess2layer[x], sesid)))
intralayer, interlayer = champ.create_multilayer_igraph_from_adjacency(A=A, C=C, layer_vec=layer_vec)

gammas = np.linspace(0, 2, 100)
omegas = np.linspace(0, 2, 100)
parts = repeated_parallel_louvain_from_gammas_omegas(intralayer, interlayer, layer_vec, gammas, omegas)
pickle.dump(parts, open("senate_10K_louvain.p", "wb"))
