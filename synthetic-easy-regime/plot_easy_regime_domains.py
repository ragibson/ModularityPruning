# TODO: combine this with easy_regime_generation.py and easy_regime_louvain.py for figure 5.4

from utilities import CHAMP_3D, domains_to_gamma_omega_estimates, plot_2d_domains_with_estimates
import pickle
import matplotlib.pyplot as plt
from time import time

# Import graph and partitions
G_intralayer, G_interlayer, ground_truth = pickle.load(open("easy_regime_multilayer.p", "rb"))
n_per_layer = 150
num_layers = 15
layer_vec = [i // n_per_layer for i in range(n_per_layer * num_layers)]
gamma_start, gamma_end = 0.0, 2.0
omega_start, omega_end = 0.0, 2.0
all_parts = pickle.load(open("easy_regime_50K_louvain.p", "rb"))

# Prune partitions with CHAMP
print("Starting CHAMP...")
start = time()
domains = CHAMP_3D(G_intralayer, G_interlayer, layer_vec, all_parts, gamma_start, gamma_end, omega_start, omega_end)
print("Took {:.2f} s".format(time() - start))

# Get parameter estimates
print("Starting parameter estimation...")
start = time()
domains_with_estimates = domains_to_gamma_omega_estimates(G_intralayer, G_interlayer, layer_vec, domains)
print("Took {:.2f} s".format(time() - start))

pickle.dump(domains_with_estimates, open("synthetic_champ_domains_with_estimates.p", "wb"))
# domains_with_estimates = pickle.load(open("synthetic_champ_domains_with_estimates.p", "rb"))

# Plot domains of optimality with parameter estimates
for repeat in range(100):
    plt.close()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_2d_domains_with_estimates(domains_with_estimates, [0.4, 1.6], [0.6, 1.45], flip_axes=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title("Synthetic Network Domains and (Gamma, Omega) Estimates", fontsize=14)
    plt.xlabel(r"$\omega$", fontsize=14)
    plt.ylabel(r"$\gamma$", fontsize=14)
    plt.savefig("synthetic_network_with_gamma_omega_estimates{}.pdf".format(repeat))
