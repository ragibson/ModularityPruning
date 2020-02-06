import pickle
import matplotlib.pyplot as plt
from utilities import gamma_omega_estimates_to_stable_partitions, plot_2d_domains_with_estimates, num_communities, ami

# Import graph and CHAMP's pruned partitions with estimates
G_intralayer, G_interlayer, ground_truth = pickle.load(open("easy_regime_multilayer.p", "rb"))
n_per_layer = 150
num_layers = 15
layer_vec = [i // n_per_layer for i in range(n_per_layer * num_layers)]
domains_with_estimates = pickle.load(open("synthetic_champ_domains_with_estimates.p", "rb"))

print(len(domains_with_estimates))

stable_domains_with_estimates = gamma_omega_estimates_to_stable_partitions(domains_with_estimates)

plt.close()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plot_2d_domains_with_estimates(stable_domains_with_estimates, [0.4, 1.6], [0.6, 1.45], flip_axes=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.title("Synthetic Network Domains and (Gamma, Omega) Estimates", fontsize=14)
plt.xlabel(r"$\omega$", fontsize=14)
plt.ylabel(r"$\gamma$", fontsize=14)
plt.show()

for polyverts, membership, gamma_est, omega_est in stable_domains_with_estimates:
    print("K={}, AMI={:.3f}, estimate=({:.3f}, {:.3f})"
          "".format(num_communities(membership), ami(membership, ground_truth), gamma_est, omega_est))
