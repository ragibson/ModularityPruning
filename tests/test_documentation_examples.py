"""
This set of tests checks that the examples from the documentation still work correctly.

Sometimes this is simply checking that the code produces the intended output or runs without errors.
"""
import unittest
from random import seed, random

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from modularitypruning import prune_to_stable_partitions, prune_to_multilayer_stable_partitions
from modularitypruning.champ_utilities import CHAMP_2D, CHAMP_3D
from modularitypruning.leiden_utilities import (repeated_parallel_leiden_from_gammas,
                                                repeated_parallel_leiden_from_gammas_omegas)
from modularitypruning.parameter_estimation_utilities import domains_to_gamma_omega_estimates, ranges_to_gamma_estimates
from modularitypruning.partition_utilities import num_communities
from modularitypruning.plotting import (plot_2d_domains_with_estimates, plot_2d_domains, plot_2d_domains_with_ami,
                                        plot_2d_domains_with_num_communities, plot_estimates, plot_multiplex_community)


class TestDocumentationExamples(unittest.TestCase):
    def test_basic_singlelayer_example(self):
        """
        Taken verbatim from basic_example.rst.

        Like a lot of our other tests, this is stochastic but appears incredibly stable.
        """
        # get Karate Club graph in igraph
        G = ig.Graph.Famous("Zachary")

        # run leiden 1000 times on this graph from gamma=0 to gamma=2
        partitions = repeated_parallel_leiden_from_gammas(G, np.linspace(0, 2, 1000))

        # prune to the stable partitions from gamma=0 to gamma=2
        stable_partitions = prune_to_stable_partitions(G, partitions, 0, 2)

        intended_stable_partition = [(0, 0, 0, 0, 1, 1, 1, 0, 2, 2, 1, 0, 0, 0, 2, 2, 1,
                                      0, 2, 0, 2, 0, 2, 3, 3, 3, 2, 3, 3, 2, 2, 3, 2, 2)]
        self.assertEqual(stable_partitions, intended_stable_partition)

    @staticmethod
    def generate_basic_multilayer_network():
        """This is taken verbatim from basic_multilayer_example.rst."""
        num_layers = 3
        n_per_layer = 30
        p_in = 0.5
        p_out = 0.05
        K = 3

        # layer_vec holds the layer membership of each node
        # e.g. layer_vec[5] = 2 means that node 5 resides in layer 2 (the third layer)
        layer_vec = [i // n_per_layer for i in range(n_per_layer * num_layers)]
        interlayer_edges = [(n_per_layer * layer + v, n_per_layer * layer + v + n_per_layer)
                            for layer in range(num_layers - 1)
                            for v in range(n_per_layer)]

        # set up a community vector with
        #   three communities in layer 0 (each of size 10)
        #   three communities in layer 1 (each of size 10)
        #   one community in layer 2 (of size 30)
        comm_per_layer = [[i // (n_per_layer // K) if layer < num_layers - 1 else 0
                           for i in range(n_per_layer)] for layer in range(num_layers)]
        comm_vec = [item for sublist in comm_per_layer for item in sublist]

        # randomly connect nodes inside each layer with undirected edges according to
        # within-community probability p_in and between-community probability p_out
        intralayer_edges = [(u, v) for v in range(len(comm_vec)) for u in range(v + 1, len(comm_vec))
                            if layer_vec[v] == layer_vec[u] and (
                                    (comm_vec[v] == comm_vec[u] and random() < p_in) or
                                    (comm_vec[v] != comm_vec[u] and random() < p_out)
                            )]

        # create the networks in igraph. By Pamfil et al.'s convention, the interlayer edges
        # of a temporal network are directed (representing the "one-way" nature of time)
        G_intralayer = ig.Graph(intralayer_edges)
        G_interlayer = ig.Graph(interlayer_edges, directed=True)

        return G_intralayer, G_interlayer, layer_vec

    def test_basic_multilayer_example(self):
        """
        This is taken verbatim from basic_multilayer_example.rst.

        For simplicity and re-use, the network generation is encapsulated in generate_basic_multilayer_network().
        """
        n_per_layer = 30  # from network generation code
        G_intralayer, G_interlayer, layer_vec = self.generate_basic_multilayer_network()

        # run leidenalg on a uniform 32x32 grid (1024 samples) of gamma and omega in [0, 2]
        gamma_range = (0, 2)
        omega_range = (0, 2)
        leiden_gammas = np.linspace(*gamma_range, 32)
        leiden_omegas = np.linspace(*omega_range, 32)

        parts = repeated_parallel_leiden_from_gammas_omegas(G_intralayer, G_interlayer, layer_vec,
                                                            gammas=leiden_gammas, omegas=leiden_omegas)

        # prune to the stable partitions from (gamma=0, omega=0) to (gamma=2, omega=2)
        stable_parts = prune_to_multilayer_stable_partitions(G_intralayer, G_interlayer, layer_vec,
                                                             "temporal", parts,
                                                             *gamma_range, *omega_range)

        # check all 3-partition stable partitions closely match ground truth communities
        for membership in stable_parts:
            if num_communities(membership) != 3:
                continue

            most_common_label = []
            for chunk_idx in range(6):  # check most common label of each community (10 nodes each)
                counts = {i: 0 for i in range(max(membership) + 1)}
                for chunk_label in membership[10 * chunk_idx:10 * (chunk_idx + 1)]:
                    counts[chunk_label] += 1
                most_common_label.append(max(counts.items(), key=lambda x: x[1])[0])

            # check these communities look like the intended ground truth communities for the first layer
            #   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            self.assertNotEqual(most_common_label[0], most_common_label[1])
            self.assertNotEqual(most_common_label[1], most_common_label[2])

        # at least one partition has the last layer mostly in one community and another splits it into multiple
        unified_final_layer_count = 0
        split_final_layer_count = 0
        for membership in stable_parts:
            count_final_layer = {i: 0 for i in range(max(membership) + 1)}
            for label in membership[-n_per_layer:]:
                count_final_layer[label] += 1
            most_common_label_final_layer, most_common_label_count = max(count_final_layer.items(),
                                                                         key=lambda x: x[1])
            proportion_final_layer_having_same_label = most_common_label_count / n_per_layer

            if proportion_final_layer_having_same_label > 0.9:
                unified_final_layer_count += 1
            elif proportion_final_layer_having_same_label < 0.5:
                split_final_layer_count += 1

        self.assertGreater(unified_final_layer_count, 0)
        self.assertGreater(split_final_layer_count, 0)

    def test_plot_estimates_example(self):
        """
        This is taken (almost) verbatim from plotting_examples.rst.

        The first call to plt.rc() has usetex=False (instead of True) to avoid requiring a full LaTeX installation.
        """
        # get Karate Club graph in igraph
        G = ig.Graph.Famous("Zachary")

        # run leiden 100K times on this graph from gamma=0 to gamma=2 (takes ~2-3 seconds)
        partitions = repeated_parallel_leiden_from_gammas(G, np.linspace(0, 2, 10 ** 5))

        # run CHAMP to obtain the dominant partitions along with their regions of optimality
        ranges = CHAMP_2D(G, partitions, gamma_0=0.0, gamma_f=2.0)

        # append gamma estimate for each dominant partition onto the CHAMP domains
        gamma_estimates = ranges_to_gamma_estimates(G, ranges)

        # plot gamma estimates and domains of optimality
        plt.rc('text', usetex=False)
        plt.rc('font', family='serif')
        plot_estimates(gamma_estimates)
        plt.title(r"Karate Club CHAMP Domains of Optimality and $\gamma$ Estimates", fontsize=14)
        plt.xlabel(r"$\gamma$", fontsize=14)
        plt.ylabel("Number of communities", fontsize=14)

    def test_plot_2d_domains_examples(self):
        """
        This is taken (almost) verbatim from plotting_examples.rst.

        The first call to plt.rc() has usetex=False (instead of True) to avoid requiring a full LaTeX installation.

        The documentation explicitly shows plot_2d_domains_with_estimates() and describes other, similar functions
            * plot_2d_domains()
            * plot_2d_domains_with_ami()
            * plot_2d_domains_with_num_communities()
        As such, we test them all here.
        """
        G_intralayer, G_interlayer, layer_vec = self.generate_basic_multilayer_network()
        # run leiden on a uniform grid (10K samples) of gamma and omega (takes ~3 seconds)
        gamma_range = (0.5, 1.5)
        omega_range = (0, 2)
        parts = repeated_parallel_leiden_from_gammas_omegas(G_intralayer, G_interlayer, layer_vec,
                                                            gammas=np.linspace(*gamma_range, 100),
                                                            omegas=np.linspace(*omega_range, 100))

        # run CHAMP to obtain the dominant partitions along with their regions of optimality
        domains = CHAMP_3D(G_intralayer, G_interlayer, layer_vec, parts,
                           gamma_0=gamma_range[0], gamma_f=gamma_range[1],
                           omega_0=omega_range[0], omega_f=omega_range[1])

        # append resolution parameter estimates for each dominant partition onto the CHAMP domains
        domains_with_estimates = domains_to_gamma_omega_estimates(G_intralayer, G_interlayer, layer_vec,
                                                                  domains, model='temporal')

        # plot resolution parameter estimates and domains of optimality
        plt.rc('text', usetex=False)
        plt.rc('font', family='serif')
        plot_2d_domains_with_estimates(domains_with_estimates, xlim=omega_range, ylim=gamma_range)
        plt.title(r"CHAMP Domains and ($\omega$, $\gamma$) Estimates", fontsize=16)
        plt.xlabel(r"$\omega$", fontsize=20)
        plt.ylabel(r"$\gamma$", fontsize=20)
        plt.gca().tick_params(axis='both', labelsize=12)
        plt.tight_layout()

        # same plotting code, but with plot_2d_domains()
        plt.rc('text', usetex=False)
        plt.rc('font', family='serif')
        plot_2d_domains(domains, xlim=omega_range, ylim=gamma_range)
        plt.title(r"CHAMP Domains", fontsize=16)
        plt.xlabel(r"$\omega$", fontsize=20)
        plt.ylabel(r"$\gamma$", fontsize=20)
        plt.gca().tick_params(axis='both', labelsize=12)
        plt.tight_layout()

        # same plotting code, but with plot_2d_domains_with_ami()
        plt.rc('text', usetex=False)
        plt.rc('font', family='serif')
        ground_truth_partition = ([0] * 10 + [1] * 10 + [2] * 10) * 2 + [0] * 30
        plot_2d_domains_with_ami(domains_with_estimates, ground_truth=ground_truth_partition,
                                 xlim=omega_range, ylim=gamma_range)
        plt.title(r"CHAMP Domains, Colored by AMI with Ground Truth", fontsize=16)
        plt.xlabel(r"$\omega$", fontsize=20)
        plt.ylabel(r"$\gamma$", fontsize=20)
        plt.gca().tick_params(axis='both', labelsize=12)
        plt.tight_layout()

        # same plotting code, but with plot_2d_domains_with_num_communities()
        plt.rc('text', usetex=False)
        plt.rc('font', family='serif')
        plot_2d_domains_with_num_communities(domains_with_estimates, xlim=omega_range, ylim=gamma_range)
        plt.title(r"CHAMP Domains, Colored by Number of Communities", fontsize=16)
        plt.xlabel(r"$\omega$", fontsize=20)
        plt.ylabel(r"$\gamma$", fontsize=20)
        plt.gca().tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        plt.close()  # closing all these figures instead of showing

    def test_plot_multiplex_community(self):
        """
        This is taken (almost) verbatim from plotting_examples.rst.

        The first call to plt.rc() has usetex=False (instead of True) to avoid requiring a full LaTeX installation.
        """
        num_layers = 3
        layer_vec = [i // 71 for i in range(num_layers * 71)]
        membership = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                      2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2,
                      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

        plt.rc('text', usetex=False)
        plt.rc('font', family='serif')
        ax = plot_multiplex_community(np.array(membership), np.array(layer_vec))
        ax.set_xticks(np.linspace(0, num_layers, 2 * num_layers + 1))
        ax.set_xticklabels(["", "Advice", "", "Coworker", "", "Friend", ""], fontsize=14)
        plt.title(f"Multiplex Communities", fontsize=14)
        plt.ylabel("Node ID", fontsize=14)
        plt.close()  # closing this these figures instead of showing


if __name__ == "__main__":
    seed(0)
    unittest.main()
