from utilities import repeated_parallel_louvain_from_gammas_omegas
import numpy as np
import pickle

G_intralayer, G_interlayer, ground_truth_comms = pickle.load(open("easy_regime_multilayer.p", "rb"))
n_per_layer = 150
num_layers = 15
layer_vec = [i // n_per_layer for i in range(n_per_layer * num_layers)]

all_g0s = np.linspace(0.0, 2.0, 225)
all_o0s = np.linspace(0.0, 2.0, 225)
all_parts = repeated_parallel_louvain_from_gammas_omegas(G_intralayer, G_interlayer, layer_vec, all_g0s, all_o0s)
pickle.dump(all_parts, open("easy_regime_50K_louvain.p", "wb"))
