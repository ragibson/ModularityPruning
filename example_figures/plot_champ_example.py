# Generates figure 3.1

import igraph as ig
import louvain
import numpy as np
from utilities import CHAMP_2D
import matplotlib.pyplot as plt

G = ig.Graph.Famous("Zachary")
twom = 2 * G.ecount()
gammas = np.linspace(0.5, 2.0, 4)
denser_gammas = np.linspace(0.25, 2.25, 10)
parts = []

for gamma in gammas:
    current_parts = [louvain.find_partition(G, louvain.RBConfigurationVertexPartition,
                                            resolution_parameter=gamma) for _ in range(10 ** 4)]
    qualities = [(p, p.quality()) for p in current_parts]
    min_Q = min(qualities, key=lambda x: x[1])[0]
    max_Q = max(qualities, key=lambda x: x[1])[0]

    if gamma == 1.0:
        parts.append(min_Q)
    else:
        parts.append(max_Q)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, axs = plt.subplots(1, 3, sharey=True, figsize=(15, 5), tight_layout=True)
ax1, ax2, ax3 = axs

ax1.scatter(gammas, [p.quality() / twom for p in parts], c=["C0", "C1", "C2", "C3"])
ax1.set_xlabel(r"$\gamma$", fontsize=14)
ax1.set_ylabel(r"$Q$", fontsize=14)
ax1.set_title("CHAMP Initial Partitions", fontsize=14)
ax1.set_xlim([0.25, 2.25])
ax1.set_ylim([0, 0.8])

ax2.scatter(gammas, [p.quality() / twom for p in parts], c=["C0", "C1", "C2", "C3"])
ys_list = []
for p in parts:
    ys_list.append([p.quality(resolution_parameter=g) / twom for g in denser_gammas])
    ax2.plot(denser_gammas, ys_list[-1])
ax2.set_xlabel(r"$\gamma$", fontsize=14)
# ax2.set_ylabel(r"$Q$", fontsize=14)
ax2.set_title("CHAMP Partition Halfspaces", fontsize=14)
ax2.set_xlim([0.25, 2.25])
ax2.set_ylim([0, 0.8])

domains = CHAMP_2D(G, [p.membership for p in parts], 0.25, 2.25)
xs1 = np.linspace(domains[0][0], domains[0][1], 10)
ys1 = [parts[0].quality(resolution_parameter=g) / twom for g in xs1]
xs3 = np.linspace(domains[1][0], domains[1][1], 10)
ys3 = [parts[2].quality(resolution_parameter=g) / twom for g in xs3]
xs4 = np.linspace(domains[2][0], domains[2][1], 10)
ys4 = [parts[3].quality(resolution_parameter=g) / twom for g in xs4]
ax3.plot(xs1, ys1, c="C0")
ax3.plot(xs3, ys3, c="C2")
ax3.plot(xs4, ys4, c="C3")
ax3.fill_between(xs1, [0] * len(xs1), [0.01] * len(xs1), color="C0")
ax3.fill_between(xs3, [0] * len(xs3), [0.01] * len(xs3), color="C2")
ax3.fill_between(xs4, [0] * len(xs4), [0.01] * len(xs4), color="C3")
ax3.axvline(xs1[-1], ymin=0, ymax=ys1[-1] / 0.8, c='black', linestyle='--')
ax3.axvline(xs3[-1], ymin=0, ymax=ys3[-1] / 0.8, c='black', linestyle='--')
ax3.set_xlabel(r"$\gamma$", fontsize=14)
# ax3.set_ylabel(r"$Q$", fontsize=14)
ax3.set_title("CHAMP Optimal Partitions and Domains", fontsize=14)
ax3.set_xlim([0.25, 2.25])
ax3.set_ylim([0, 0.8])
plt.savefig("CHAMP_example.pdf")
