import pickle
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from random import shuffle
from utilities import ami, num_communities, nmi, plot_2d_domains_with_num_communities

# Import graph and CHAMP's pruned partitions with estimates
G_intralayer, G_interlayer, ground_truth = pickle.load(open("easy_regime_multilayer.p", "rb"))
n_per_layer = 150
num_layers = 15
layer_vec = [i // n_per_layer for i in range(n_per_layer * num_layers)]
domains_with_estimates = pickle.load(open("synthetic_champ_domains_with_estimates.p", "rb"))


def plot_2d_domains_with_ami(domains_with_estimates, xlim, ylim, flip_axes=False):
    fig, ax = plt.subplots()
    patches = []

    for polyverts, membership, gamma_est, omega_est in domains_with_estimates:
        if flip_axes:
            polyverts = [(x[1], x[0]) for x in polyverts]

        polygon = Polygon(polyverts, True)
        patches.append(polygon)

    cm = matplotlib.cm.copper
    amis = np.array([ami(membership, ground_truth) for _, membership, _, _ in domains_with_estimates] + [1.0])

    p = PatchCollection(patches, cmap=cm, alpha=1.0, edgecolors='black', linewidths=2)
    p.set_array(amis)
    ax.add_collection(p)

    cbar = plt.colorbar(p)
    cbar.set_label('AMI', fontsize=14, labelpad=15)

    plt.xlim(xlim)
    plt.ylim(ylim)


plt.close()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plot_2d_domains_with_ami(domains_with_estimates, [0.4, 1.6], [0.6, 1.45], flip_axes=True)
plt.title("AMI of Domains with Ground Truth", fontsize=14)
plt.xlabel(r"$\omega$", fontsize=14)
plt.ylabel(r"$\gamma$", fontsize=14)
plt.savefig("synthetic_network_domains_with_ground_truth_ami.pdf")

plt.close()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plot_2d_domains_with_num_communities(domains_with_estimates, [0.4, 1.6], [0.6, 1.45], flip_axes=True)
plt.title("Domains with Number of Communities", fontsize=14)
plt.xlabel(r"$\omega$", fontsize=14)
plt.ylabel(r"$\gamma$", fontsize=14)
plt.savefig("synthetic_network_domains_with_num_communities.pdf")
