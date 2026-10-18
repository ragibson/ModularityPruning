import unittest
from math import log
from random import seed

import igraph as ig
from modularitypruning.leiden_utilities import repeated_leiden_from_gammas_omegas
from modularitypruning.parameter_estimation import iterative_multilayer_resolution_parameter_estimation
from modularitypruning.parameter_estimation_utilities import gamma_omega_estimate
from modularitypruning.partition_utilities import num_communities, all_degrees, in_degrees, out_degrees
from numpy import mean

from .shared_testing_functions import assert_almost_equal_or_both_none_or_nan, generate_random_partition, \
    generate_multiplex_SBM


class TestMultiplexParameterEstimation(unittest.TestCase):
    def assert_multiplex_SBM_correct_convergence(self, first_layer_membership, copying_probability=0.75, num_layers=10,
                                                 p_in=0.25, p_out=0.05, weighted=False, directed=False):
        K = num_communities(first_layer_membership)
        G_intralayer, G_interlayer, layer_membership = generate_multiplex_SBM(copying_probability, p_in, p_out,
                                                                              first_layer_membership,
                                                                              num_layers, weighted=weighted,
                                                                              directed=directed)

        # compute ground truth gamma
        if directed:
            k_in = mean(in_degrees(G_intralayer))
            k_out = mean(out_degrees(G_intralayer))
            true_theta_in = p_in * G_intralayer.ecount() / (k_in * k_out) / num_layers
            true_theta_out = p_out * G_intralayer.ecount() / (k_in * k_out) / num_layers
        else:
            k = mean(all_degrees(G_intralayer))
            true_theta_in = p_in * (2 * G_intralayer.ecount()) / (k * k) / num_layers
            true_theta_out = p_out * (2 * G_intralayer.ecount()) / (k * k) / num_layers
        true_gamma = (true_theta_in - true_theta_out) / (log(true_theta_in) - log(true_theta_out))

        # compute ground truth omega
        true_omega = log(1 + copying_probability * K / (1 - copying_probability))
        true_omega /= (num_layers * (log(true_theta_in) - log(true_theta_out)))

        gamma, omega, part = iterative_multilayer_resolution_parameter_estimation(G_intralayer, G_interlayer,
                                                                                  layer_membership, gamma=1.0,
                                                                                  omega=0.1,
                                                                                  model='multiplex')

        # check we converged close to the ground truth "correct" values
        # the multiplex parameter estimation is much less robust and less accurate than in other models,
        # perhaps due to the copying probability approximation
        self.assertLess(abs(true_gamma - gamma), 0.1)
        self.assertLess(abs(true_omega - omega), 0.2)

    def test_multiplex_SBM_correct_convergence_varying_copying_probabilty(self):
        for weighted in [False, True]:
            for eta in [0.25, 0.5, 0.75, 0.9]:
                with self.subTest(eta=eta, weighted=weighted):
                    membership = generate_random_partition(num_nodes=200, K=2)
                    self.assert_multiplex_SBM_correct_convergence(copying_probability=eta,
                                                                  first_layer_membership=membership,
                                                                  weighted=weighted)

    def test_multiplex_SBM_correct_convergence_varying_p_in(self):
        for weighted in [False, True]:
            for p_in in [0.5, 0.4, 0.3, 0.2]:
                with self.subTest(p_in=p_in, weighted=weighted):
                    membership = generate_random_partition(num_nodes=200, K=2)
                    self.assert_multiplex_SBM_correct_convergence(p_in=p_in, p_out=0.025,
                                                                  first_layer_membership=membership,
                                                                  weighted=weighted)

    def test_multiplex_SBM_correct_convergence_varying_p_out(self):
        for weighted in [False, True]:
            for p_out in [0.05, 0.04, 0.03, 0.02]:
                with self.subTest(p_out=p_out, weighted=weighted):
                    membership = generate_random_partition(num_nodes=200, K=2)
                    self.assert_multiplex_SBM_correct_convergence(p_out=p_out, first_layer_membership=membership,
                                                                  weighted=weighted)

    def test_multiplex_SBM_correct_convergence_varying_num_communities(self):
        for weighted in [False, True]:
            for K in [2, 3, 4, 5]:
                with self.subTest(K=K, weighted=weighted):
                    membership = generate_random_partition(num_nodes=300, K=K)
                    self.assert_multiplex_SBM_correct_convergence(first_layer_membership=membership,
                                                                  weighted=weighted)

    def test_multiplex_SBM_correct_convergence_varying_num_layers(self):
        for weighted in [False, True]:
            for num_layers in [3, 5, 7, 10]:
                with self.subTest(num_layers=num_layers, weighted=weighted):
                    membership = generate_random_partition(num_nodes=300, K=2)
                    self.assert_multiplex_SBM_correct_convergence(first_layer_membership=membership,
                                                                  num_layers=num_layers, weighted=weighted)

    def test_directed_multiplex_SBM_correct_convergence(self):
        """Directed analogue of the multiplex convergence tests."""
        for weighted in [False, True]:
            with self.subTest(weighted=weighted):
                membership = generate_random_partition(num_nodes=300, K=3)
                self.assert_multiplex_SBM_correct_convergence(copying_probability=0.75, num_layers=8,
                                                              first_layer_membership=membership, weighted=weighted,
                                                              directed=True)

    def test_directed_asymmetry_multiplex(self):
        """Simple test to confirm multilayer parameter estimates depend on edge direction.

        The intralayer graph has all edges pointing from the first community to the second."""
        one_way_edges = [(0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3)]
        edge_list = one_way_edges + [(u + 4, v + 4) for (u, v) in one_way_edges]
        weights = [1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        membership = [0, 0, 1, 1, 0, 0, 1, 1]
        layer_membership = [0, 0, 0, 0, 1, 1, 1, 1]
        G_interlayer = ig.Graph([(0, 4), (4, 0), (1, 5), (5, 1), (2, 6), (6, 2), (3, 7), (7, 3)], directed=True)

        with self.subTest(model='multiplex'):
            G = ig.Graph(8, edge_list, directed=True)
            G.es['weight'] = weights
            gamma_directed, _ = gamma_omega_estimate(G, G_interlayer, layer_membership, membership, model='multiplex')

            # undirected version for comparison
            G_undirected = ig.Graph(8, edge_list, directed=False)
            G_undirected.es['weight'] = weights
            gamma_undirected_treatment, _ = gamma_omega_estimate(G_undirected, G_interlayer, layer_membership,
                                                                 membership, model='multiplex')
            self.assertGreater(abs(gamma_directed - gamma_undirected_treatment), 1e-2)

    def test_directed_consistency_multiplex_SBM_leiden(self):
        """Test parameter estimate consistency on a multiplex SBM when the intralayer edges are directed."""
        membership = [0] * 25 + [1] * 25 + [2] * 25
        G_intralayer, G_interlayer, layer_membership = generate_multiplex_SBM(copying_probability=0.9,
                                                                              p_in=0.25, p_out=0.05,
                                                                              first_layer_membership=membership,
                                                                              num_layers=25)

        partitions = repeated_leiden_from_gammas_omegas(G_intralayer, G_interlayer, layer_membership,
                                                        gammas=[0.5, 1.0, 1.5], omegas=[0.5, 1.0, 1.5])

        for i, partition in enumerate(partitions):
            with self.subTest(partition=i):
                # here, undirected/directed refers to the intralayer edges only
                # in Pamfil et al.'s multiplex networks, interlayer edges are taken to be directed
                gamma_undirected, omega_undirected = gamma_omega_estimate(G_intralayer, G_interlayer, layer_membership,
                                                                          partition, model="multiplex")

                G_intralayer.to_directed()
                gamma_directed, omega_directed = gamma_omega_estimate(G_intralayer, G_interlayer, layer_membership,
                                                                      partition, model="multiplex")

                assert_almost_equal_or_both_none_or_nan(self, gamma_undirected, gamma_directed, places=10)
                assert_almost_equal_or_both_none_or_nan(self, omega_undirected, omega_directed, places=10)


if __name__ == "__main__":
    seed(0)
    unittest.main()
