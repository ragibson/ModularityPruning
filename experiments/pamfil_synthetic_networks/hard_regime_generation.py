from easy_regime_generation import generate_synthetic_network, run_pamfil_iteration, run_louvain_on_grid, \
    generate_domains_with_estimates, plot_pamfil_iteration, plot_domains, plot_domains_with_amis, \
    plot_domains_with_Ks, plot_domains_restricted_communities
import os
import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    graph_filename = "hard_regime_multilayer.p"
    iteration_filename = "hard_regime_test_results.p"
    louvain_filename = "hard_regime_50K_louvain.p"
    domains_filename = "synthetic_champ_domains_with_estimates_hard_regime.p"
    restricted_domains_filename = "synthetic_champ_2-community_domains_with_estimates_hard_regime.p"

    if not os.path.exists(graph_filename):
        print("Generating synthetic network...")
        G_intralayer, G_interlayer, layer_vec, comm_vec = generate_synthetic_network(eta=0.5, epsilon=0.5,
                                                                                     desired_gamma=0.96,
                                                                                     desired_omega=0.80)
        pickle.dump((G_intralayer, G_interlayer, layer_vec, comm_vec), open(graph_filename, "wb"))

    if not os.path.exists(iteration_filename):
        print("Running pamfil iteration...")
        values = run_pamfil_iteration(graph_filename)
        pickle.dump(values, open(iteration_filename, "wb"))

    if not os.path.exists(louvain_filename):
        print("Running hard regime Louvain...")
        all_parts = run_louvain_on_grid(graph_filename)
        pickle.dump(all_parts, open(louvain_filename, "wb"))

    if not os.path.exists(domains_filename):
        print("Generating CHAMP domains with estimates...")
        domains_with_estimates = generate_domains_with_estimates(graph_filename, louvain_filename)
        pickle.dump(domains_with_estimates, open(domains_filename, "wb"))

    if not os.path.exists(restricted_domains_filename):
        print("Generating CHAMP domains with estimates when K=2...")
        domains_with_estimates = generate_domains_with_estimates(graph_filename, louvain_filename,
                                                                 restrict_communities=2)
        pickle.dump(domains_with_estimates, open(restricted_domains_filename, "wb"))

    plot_pamfil_iteration(graph_filename, iteration_filename)
    plt.savefig("synthetic_network_pamfil_iteration_hard_regime.pdf")

    plot_domains(domains_filename)
    plt.savefig("synthetic_network_with_gamma_omega_estimates_hard_regime.pdf")

    plot_domains_with_amis(graph_filename, domains_filename)
    plt.savefig("synthetic_network_domains_with_ground_truth_ami_hard_regime.pdf")

    plot_domains_with_Ks(domains_filename)
    plt.savefig("synthetic_network_domains_with_num_communities_hard_regime.pdf")

    plot_domains_restricted_communities(restricted_domains_filename)
    plt.savefig("synthetic_network_with_2-community_gamma_omega_estimates_hard_regime.pdf")
