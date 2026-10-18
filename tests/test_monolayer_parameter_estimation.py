import unittest
from math import log
from random import seed

import igraph as ig
from modularitypruning.parameter_estimation import iterative_monolayer_resolution_parameter_estimation
from modularitypruning.parameter_estimation_utilities import gamma_estimate
from modularitypruning.partition_utilities import all_degrees, all_strengths, in_strengths, out_strengths
from numpy import mean

from .shared_testing_functions import assert_almost_equal_or_both_none_or_nan, generate_SBM, \
    generate_igraph_famous, generate_random_partition


class TestMonolayerParameterEstimation(unittest.TestCase):
    def test_newman_synthetic_networks(self):
        """This mimics the synthetic test from Newman's paper on the equivalence.

        The setup here is the same as in FIG 1 of 'Equivalence between modularity optimization and maximum likelihood
        methods for community detection', albeit using Leiden for modularity maximization."""

        for weighted in [False, True]:
            for q in range(3, 15):
                with self.subTest(q=q, weighted=weighted):
                    self.assert_newman_synthetic_convergence(q, weighted=weighted)

    def assert_newman_synthetic_convergence(self, q, weighted=False):
        G, p_in, p_out = generate_SBM(q, weighted=weighted)

        # compute "ground truth" gamma estimate
        if weighted:
            k = mean(all_strengths(G))
            m = sum(G.es['weight'])
            mean_weight = m / G.ecount()
            true_omega_in = mean_weight * p_in * (2 * m) / (k * k)
            true_omega_out = mean_weight * p_out * (2 * m) / (k * k)
        else:
            k = mean(all_degrees(G))
            true_omega_in = p_in * (2 * G.ecount()) / (k * k)
            true_omega_out = p_out * (2 * G.ecount()) / (k * k)
        true_gamma = (true_omega_in - true_omega_out) / (log(true_omega_in) - log(true_omega_out))

        gamma, _ = iterative_monolayer_resolution_parameter_estimation(G, gamma=1.0)

        # check we converged close to the ground truth "correct" value
        self.assertLess(abs(true_gamma - gamma), 0.05)

    def test_directed_consistency_igraph_famous(self):
        """Test gamma estimate consistency on undirected and (symmetric) directed versions of various famous graphs."""

        for name, G in generate_igraph_famous():
            random_membership = generate_random_partition(G.vcount(), 5)

            gamma_undirected = gamma_estimate(G, random_membership)
            G.to_directed()
            gamma_directed = gamma_estimate(G, random_membership)

            with self.subTest(graph=name):
                assert_almost_equal_or_both_none_or_nan(self, gamma_undirected, gamma_directed, places=10)

    def assert_newman_directed_synthetic_convergence(self, q, weighted=False):
        """Test directed analogue of Newman's synthetic test."""
        G, p_in, p_out = generate_SBM(q, directed=True, weighted=weighted)

        # compute "ground truth" gamma estimate for a directed graph
        M = sum(G.es['weight']) if weighted else G.ecount()
        mean_weight = M / G.ecount() if weighted else 1.0
        k_in = mean(in_strengths(G))
        k_out = mean(out_strengths(G))
        true_omega_in = mean_weight * p_in * M / (k_in * k_out)
        true_omega_out = mean_weight * p_out * M / (k_in * k_out)
        true_gamma = (true_omega_in - true_omega_out) / (log(true_omega_in) - log(true_omega_out))

        gamma, _ = iterative_monolayer_resolution_parameter_estimation(G, gamma=1.0)

        # check we converged close to the ground truth "correct" value
        self.assertLess(abs(true_gamma - gamma), 0.05)

    def test_directed_newman_synthetic_networks(self):
        """Directed analogue of test_newman_synthetic_networks."""

        for weighted in [False, True]:
            for q in range(3, 15):
                with self.subTest(q=q, weighted=weighted):
                    self.assert_newman_directed_synthetic_convergence(q, weighted=weighted)

    def test_directed_difference(self):
        """Simple test to confirm parameter estimate depends on edge direction.

        The graph here has all edges pointing from the first community to the second."""
        edges = [(0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3)]
        membership = (0, 0, 1, 1)

        for i, weights in enumerate([[1.0] * 6, [1.0, 1.0, 3.0, 3.0, 3.0, 3.0]]):
            with self.subTest(weight_set=i):
                G = ig.Graph(4, edges, directed=True)
                G.es['weight'] = weights
                gamma_directed = gamma_estimate(G, membership)

                # undirected version for comparison
                G_undirected = ig.Graph(4, edges, directed=False)
                G_undirected.es['weight'] = weights
                gamma_undirected_treatment = gamma_estimate(G_undirected, membership)

                self.assertGreater(abs(gamma_directed - gamma_undirected_treatment), 1e-2)


if __name__ == "__main__":
    seed(0)
    unittest.main()
