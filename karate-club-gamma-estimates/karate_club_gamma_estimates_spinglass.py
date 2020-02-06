import igraph as ig
import matplotlib.pyplot as plt
from utilities import iterative_monolayer_resolution_parameter_estimation, Progress
import numpy as np


def one_step(G, gamma):
    # Assumes resolution parameter estimation is using spinglass community detection with 2 spins
    # e.g. return G.community_spinglass(spins=2, gamma=resolution_param)
    return iterative_monolayer_resolution_parameter_estimation(G, gamma=gamma, max_iter=1)[0]


G = ig.Graph.Famous("Zachary")

gamma_0s = []
gamma_fs = []
parts = []

progress = Progress(300, name="Estimation Until Convergence")
for gamma in np.linspace(0.3, 3.0, 300):
    for _ in range(1):
        try:
            final_gamma, part = iterative_monolayer_resolution_parameter_estimation(G, gamma=gamma, verbose=False)
            gamma_0s.append(gamma)
            gamma_fs.append(final_gamma)
            parts.append(parts)
        except ValueError:
            pass
    progress.increment()
progress.done()

plt.figure(figsize=(5, 5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlim([0.2, 3.1])
plt.ylim([0.2, 3.1])
plt.ylabel("Initial Gamma", size=14)
plt.xlabel("Final Gamma", size=14)

progress = Progress(50, name="One-Step Estimation")
for g0 in np.linspace(0.3, 3.0, 45):
    total = 0
    count = 0
    for _ in range(1):
        try:
            total += one_step(G, gamma=g0) - g0
            count += 1
        except ValueError:
            # print('value error on ', g0)
            pass
    if count > 0:
        average_gamma_movement = total / count
        plt.plot([g0], [g0], c='black', marker='o', markersize=2)
        plt.arrow(g0, g0, average_gamma_movement, 0, width=0.005, head_length=0.04, head_width=0.04,
                  length_includes_head=True, color="black")

    progress.increment()
progress.done()

plt.scatter(gamma_fs, gamma_0s, alpha=1.0, s=15)
plt.title(r"$\gamma$ Estimation on the Karate Club with Spinglass", fontsize=14)
plt.savefig("karate_club_gamma_estimation_spinglass.pdf")
