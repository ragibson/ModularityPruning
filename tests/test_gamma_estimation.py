import igraph as ig
from math import log
from numpy import mean
from modularitypruning.parameter_estimation import iterative_monolayer_resolution_parameter_estimation
from modularitypruning.parameter_estimation_utilities import gamma_estimate
from random import randint, seed
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

    def test_directed_consistency_igraph_famous(self):
        """Test consistency of gamma estimates on undirected and (symmetric) directed versions of various famous graphs.

        In particular, we use
            Meredith (n=70, m=140): a counterexample to a conjecture regarding 4-regular 4-connected Hamiltonian graphs
            Nonline (n=50, m=72): a disconnnected graph composed of the 9 subgraphs whose presence makes a nonline graph
            Thomassen (n=34, m=52): the smallest graph without a Hamiltonian path
            Tutte (n=46, m=69): a counterexample to a conjecture regarding 3-connected 3-regular Hamiltonian graphs
            Zachary (n=34, m=78): popular network of the social interactions between 34 members of a karate club
        """

        seed(0)
        for name in ['meredith', 'nonline', 'thomassen', 'tutte', 'zachary']:
            G = ig.Graph.Famous(name)
            random_membership = [randint(0, 5) for _ in range(G.vcount())]

            gamma_undirected = gamma_estimate(G, random_membership)
            G.to_directed()
            gamma_directed = gamma_estimate(G, random_membership)

            self.assertAlmostEqual(gamma_undirected, gamma_directed, places=10)


if __name__ == "__main__":
    unittest.main()
