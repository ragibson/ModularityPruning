import pickle
from utilities import *
from social_networks import read_graphs
import matplotlib.pyplot as plt

K_MAX = 71

Ks = list(range(2, K_MAX))
est_means, est_maxs = pickle.load(open("maximum_gammas.p", "rb"))
est_means = est_means[2:]
est_maxs = est_maxs[2:]
assert abs(est_maxs[0] - 1.0) < 1e-4

# Gs = read_graphs()
# print("Graphs imported.")
#
# results = []
# for i in range(16):
#     print("Processing parts{}...".format(i))
#     for p in pickle.load(open("parts{}.p".format(i), "rb")):
#         K = num_communities(p)
#         g_est = gamma_estimate(Gs[i], p)
#
#         if g_est is not None and 2 <= K < K_MAX:
#             assert g_est < 15
#             results.append((K, g_est))
#
# pickle.dump(results, open("boxplot_results.p", "wb"))
results = pickle.load(open("boxplot_results.p", "rb"))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure(figsize=(10, 5))
plt.plot(Ks, est_means, '-', label=r"$\gamma_{mean}$")
plt.plot(Ks, est_maxs, '-', label=r"$\gamma_{max}$")

boxplots = []
for K in Ks:
    gamma_ests = [x[1] for x in results if x[0] == K]
    boxplots.append(gamma_ests)
    print("K={}, count={}".format(K, len(gamma_ests)))

plt.boxplot(boxplots, sym='+', flierprops={"markersize": 5}, medianprops={"color": "black"}, positions=Ks)
plt.title(r"Empirical $\gamma$ Estimates from SNAP Networks as $K$ Varies", fontsize=14)
plt.ylabel(r"$\gamma$", fontsize=14)
plt.xlabel(r"Number of Communities $K$", fontsize=14)
plt.legend(fontsize=14)
plt.ylim([0, 10])

ax = plt.axes()
xticks = ax.xaxis.get_major_ticks()
for i in range(len(xticks)):
    xticks[i].label1.set_visible(False)

for i in range(0, len(xticks), 4):
    xticks[i].label1.set_visible(True)

plt.savefig("empirical_gamma_max.pdf")
