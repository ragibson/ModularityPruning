# Generates figure 3.1

import igraph as ig
import louvain
import numpy as np
from modularitypruning.champ_utilities import CHAMP_2D
import matplotlib.pyplot as plt

SAMPLE_GAMMAS = [0.5, 1.0, 1.5, 2.0]
PLOT_XLIM = [0.25, 2.25]
PLOT_YLIM = [0, 0.8]
PLOT_XS = np.linspace(PLOT_XLIM[0], PLOT_XLIM[1], 10)


def sample_partitions():
    """Sample some partitions from the Karate club"""
    G = ig.Graph.Famous("Zachary")
    parts = []

    for gamma in SAMPLE_GAMMAS:
        sampled_partitions = [louvain.find_partition(G, louvain.RBConfigurationVertexPartition,
                                                     resolution_parameter=gamma) for _ in range(10 ** 4)]
        if gamma == 1.0:  # artificially make this partition low-quality
            parts.append(min(sampled_partitions, key=lambda p: p.quality()))
        else:
            parts.append(max(sampled_partitions, key=lambda p: p.quality()))

    return parts


def plot_left_panel(ax, parts):
    """Plots sampled partitions in (gamma, Q) space"""
    G = ig.Graph.Famous("Zachary")
    twom = 2 * G.ecount()

    ax.scatter(SAMPLE_GAMMAS, [p.quality() / twom for p in parts], c=["C0", "C1", "C2", "C3"])
    ax.set_xlabel(r"$\gamma$", fontsize=14)
    ax.set_ylabel(r"$Q$", fontsize=14)
    ax.set_title("CHAMP Initial Partitions", fontsize=14)
    ax.set_xlim(PLOT_XLIM)
    ax.set_ylim(PLOT_YLIM)


def plot_center_panel(ax, parts):
    """Plots sampled partitions as halfspaces in (gamma, Q) space"""
    G = ig.Graph.Famous("Zachary")
    twom = 2 * G.ecount()

    ax.scatter(SAMPLE_GAMMAS, [p.quality() / twom for p in parts], c=["C0", "C1", "C2", "C3"])
    for p in parts:
        ax.plot(PLOT_XS, [p.quality(resolution_parameter=g) / twom for g in PLOT_XS])
    ax.set_xlabel(r"$\gamma$", fontsize=14)
    ax.set_title("CHAMP Partition Halfspaces", fontsize=14)
    ax.set_xlim(PLOT_XLIM)
    ax.set_ylim(PLOT_YLIM)


def plot_right_panel(ax, parts):
    """Plots CHAMP domains and halfspace intersection from sampled partitions"""
    G = ig.Graph.Famous("Zachary")
    twom = 2 * G.ecount()

    domains = CHAMP_2D(G, [p.membership for p in parts], PLOT_XLIM[0], PLOT_XLIM[1])

    # Assume that all but the gamma=1.0 domain were optimal
    xs1 = [domains[0][0], domains[0][1]]
    xs3 = [domains[1][0], domains[1][1]]
    xs4 = [domains[2][0], domains[2][1]]
    ys1 = [parts[0].quality(resolution_parameter=g) / twom for g in xs1]
    ys3 = [parts[2].quality(resolution_parameter=g) / twom for g in xs3]
    ys4 = [parts[3].quality(resolution_parameter=g) / twom for g in xs4]

    # Plot halfspaces where the corresponding partition is optimal
    ax.plot(xs1, ys1, c="C0")
    ax.plot(xs3, ys3, c="C2")
    ax.plot(xs4, ys4, c="C3")

    # Project domains of optimality onto gamma axis
    ax.fill_between(xs1, [0] * len(xs1), [0.01] * len(xs1), color="C0")
    ax.fill_between(xs3, [0] * len(xs3), [0.01] * len(xs3), color="C2")
    ax.fill_between(xs4, [0] * len(xs4), [0.01] * len(xs4), color="C3")

    ax.axvline(xs1[-1], ymin=0, ymax=ys1[-1] / 0.8, c='black', linestyle='--')
    ax.axvline(xs3[-1], ymin=0, ymax=ys3[-1] / 0.8, c='black', linestyle='--')

    ax.set_xlabel(r"$\gamma$", fontsize=14)
    ax.set_title("CHAMP Optimal Partitions and Domains", fontsize=14)
    ax.set_xlim(PLOT_XLIM)
    ax.set_ylim(PLOT_YLIM)


if __name__ == "__main__":
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, axs = plt.subplots(1, 3, sharey="all", figsize=(15, 5), tight_layout=True)
    ax1, ax2, ax3 = axs

    parts = sample_partitions()

    plot_left_panel(ax1, parts)
    plot_center_panel(ax2, parts)
    plot_right_panel(ax3, parts)
    plt.savefig("CHAMP_example.pdf")
