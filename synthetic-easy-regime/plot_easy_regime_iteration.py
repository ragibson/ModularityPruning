import numpy as np
import pickle
import matplotlib.pyplot as plt

values = pickle.load(open("easy_regime_test_results.p", "rb"))
values.sort(key=lambda x: 1000 * x[0] + x[1])

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plt.subplots()

iteration = 0
for g0, o0, gdiffs, odiffs in values:
    gdiff = np.mean(gdiffs)
    odiff = np.mean(odiffs)
    SCALE = 0.1
    plt.arrow(o0, g0, SCALE * odiff, SCALE * gdiff, width=1e-3, head_length=10e-3, head_width=15e-3,
              color="black", **{"overhang": 0.5}, alpha=0.75, length_includes_head=True)

ground_truth_gamma = 0.9357510425040243
ground_truth_omega = 0.984333998485813
plt.scatter([ground_truth_omega], [ground_truth_gamma], s=50, color='blue', edgecolor='black', linewidths=1, marker='o')

plt.title("Synthetic Network (Gamma, Omega) Estimates from Louvain", fontsize=14)
plt.xlabel(r"$\omega$", fontsize=14)
plt.ylabel(r"$\gamma$", fontsize=14)
plt.xlim([0.4, 1.6])
plt.ylim([0.6, 1.4])
plt.savefig("synthetic_network_pamfil_iteration.pdf")
