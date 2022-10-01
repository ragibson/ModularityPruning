from .shared_testing_functions import assert_almost_equal_or_both_none_or_nan, generate_random_partition, \
    generate_multilayer_intralayer_SBM
import igraph as ig
from math import log
from numpy import mean
from modularitypruning.leiden_utilities import repeated_leiden_from_gammas_omegas
from modularitypruning.parameter_estimation import iterative_multilayer_resolution_parameter_estimation
from modularitypruning.parameter_estimation_utilities import gamma_omega_estimate
from modularitypruning.partition_utilities import num_communities, all_degrees
from random import seed
import unittest


class TestMultiplexParameterEstimation(unittest.TestCase):
    def generate_multiplex_SBM(self, copying_probability, p_in, p_out, first_layer_membership, num_layers):
        G_intralayer, layer_membership = generate_multilayer_intralayer_SBM(copying_probability, p_in, p_out,
                                                                            first_layer_membership, num_layers)

        # connect each node to itself in all other layers
        num_nodes_per_layer = len(first_layer_membership)
        interlayer_edges = [(num_nodes_per_layer * layer_start + v, num_nodes_per_layer * layer_end + v)
                            for layer_start in range(num_layers) for layer_end in range(num_layers)
                            if layer_start != layer_end
                            for v in range(num_nodes_per_layer)]
        G_interlayer = ig.Graph(interlayer_edges, directed=True)

        return G_intralayer, G_interlayer, layer_membership

    def assert_multiplex_SBM_correct_convergence(self, first_layer_membership, copying_probability=0.75, num_layers=10,
                                                 p_in=0.25, p_out=0.05):
        K = num_communities(first_layer_membership)
        G_intralayer, G_interlayer, layer_membership = self.generate_multiplex_SBM(copying_probability, p_in, p_out,
                                                                                   first_layer_membership,
                                                                                   num_layers)

        # compute ground truth gamma
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
        # the multiplex omega estimation seems less accurate than in other models, perhaps due to
        # the copying probability approximation
        self.assertLess(abs(true_gamma - gamma), 0.05)
        self.assertLess(abs(true_omega - omega), 0.15)

    def test_multiplex_SBM_correct_convergence_varying_copying_probabilty(self):
        for eta in [0.25, 0.5, 0.75, 0.9]:
            membership = generate_random_partition(num_nodes=200, K=2)
            self.assert_multiplex_SBM_correct_convergence(copying_probability=eta,
                                                          first_layer_membership=membership)

    def test_multiplex_SBM_correct_convergence_varying_p_in(self):
        for p_in in [0.5, 0.4, 0.3, 0.2]:
            membership = generate_random_partition(num_nodes=200, K=2)
            self.assert_multiplex_SBM_correct_convergence(p_in=p_in, p_out=0.025, first_layer_membership=membership)

    def test_multiplex_SBM_correct_convergence_varying_p_out(self):
        for p_out in [0.05, 0.04, 0.03, 0.02]:
            membership = generate_random_partition(num_nodes=100, K=2)
            self.assert_multiplex_SBM_correct_convergence(p_out=p_out, first_layer_membership=membership)

    def test_multiplex_SBM_correct_convergence_varying_num_communities(self):
        for K in [2, 3, 4, 5]:
            membership = generate_random_partition(num_nodes=300, K=K)
            self.assert_multiplex_SBM_correct_convergence(first_layer_membership=membership)

    def test_multiplex_SBM_correct_convergence_varying_num_layers(self):
        for num_layers in [3, 5, 7, 10]:
            membership = generate_random_partition(num_nodes=300, K=2)
            self.assert_multiplex_SBM_correct_convergence(first_layer_membership=membership, num_layers=num_layers)

    def test_directed_consistency_multiplex_SBM_leiden(self):
        """Test parameter estimate consistency on a multiplex SBM when the intralayer edges are directed."""
        membership = [0] * 25 + [1] * 25 + [2] * 25
        G_intralayer, G_interlayer, layer_membership = self.generate_multiplex_SBM(copying_probability=0.9,
                                                                                   p_in=0.25, p_out=0.05,
                                                                                   first_layer_membership=membership,
                                                                                   num_layers=25)

        partitions = repeated_leiden_from_gammas_omegas(G_intralayer, G_interlayer, layer_membership,
                                                        gammas=[0.5, 1.0, 1.5], omegas=[0.5, 1.0, 1.5])

        for partition in partitions:
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
