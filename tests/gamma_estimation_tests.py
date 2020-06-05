import igraph as ig
from math import log
from numpy import mean
from modularitypruning.parameter_estimation import iterative_monolayer_resolution_parameter_estimation
from random import seed
import unittest


class TestMonolayerParameterEstimation(unittest.TestCase):
    def test_newman_synthetic_networks(self):
        """This mimics the synthetic test from Newman's paper on the equivalence.

        The setup here is the same as in FIG 1 of 'Equivalence between modularity optimization and maximum likelihood
        methods for community detection', albeit using Louvain for modularity maximization."""

        seed(0)
        for q in range(3, 15):
            community_sizes = [250] * q  # q equally sized groups of 250 nodes
            n = 250 * q
            p_in = 16 * n / (q * 250 * 249)  # ~16 in-edges per node
            p_out = 8 * n / (q * (q - 1) * 250 * 250)  # ~8 out-edges per node to each community
            pref_matrix = [[p_in if i == j else p_out for j in range(q)] for i in range(q)]
            G = ig.Graph.SBM(n, pref_matrix, community_sizes)

            # compute "ground truth" gamma estimate
            k = mean([G.degree(v) for v in range(n)])
            true_omega_in = p_in * (2 * G.ecount()) / (k * k)
            true_omega_out = p_out * (2 * G.ecount()) / (k * k)
            true_gamma = (true_omega_in - true_omega_out) / (log(true_omega_in) - log(true_omega_out))

            gamma, _ = iterative_monolayer_resolution_parameter_estimation(G, gamma=1.0)
            
            # check we converged close to the ground truth "correct" value
            self.assertLess(abs(true_gamma - gamma), 0.05)


if __name__ == "__main__":
    unittest.main()
