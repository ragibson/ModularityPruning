import unittest
from math import log
from random import seed

import igraph as ig
from modularitypruning.leiden_utilities import repeated_leiden_from_gammas_omegas
from modularitypruning.parameter_estimation import iterative_multilayer_resolution_parameter_estimation
from modularitypruning.parameter_estimation_utilities import gamma_omega_estimate
from modularitypruning.partition_utilities import num_communities, all_strengths, in_strengths, out_strengths
from numpy import mean

from .shared_testing_functions import assert_almost_equal_or_both_none_or_nan, generate_random_partition, \
    generate_temporal_SBM


class TestTemporalAndMultilevelParameterEstimation(unittest.TestCase):
    def assert_temporal_SBM_correct_convergence(self, first_layer_membership, copying_probability=0.75, num_layers=12,
                                                p_in=0.25, p_out=0.05, tolerance=0.1, weighted=False,
                                                directed=False):
        K = num_communities(first_layer_membership)
        G_intralayer, G_interlayer, layer_membership = generate_temporal_SBM(copying_probability, p_in, p_out,
                                                                             first_layer_membership,
                                                                             num_layers, weighted=weighted,
                                                                             directed=directed)

        # compute "ground truth" gamma estimate
        # note that strengths are just degrees in unweighted graphs
        if weighted:
            total_weight = sum(G_intralayer.es['weight'])
            mean_weight = total_weight / G_intralayer.ecount()
        else:
            total_weight = G_intralayer.ecount()
            mean_weight = 1.0
        twom = total_weight if directed else 2 * total_weight

        if directed:
            k_in = mean(in_strengths(G_intralayer))
            k_out = mean(out_strengths(G_intralayer))
        else:
            k_in = k_out = mean(all_strengths(G_intralayer))
        true_theta_in = mean_weight * p_in * twom / (k_in * k_out) / num_layers
        true_theta_out = mean_weight * p_out * twom / (k_in * k_out) / num_layers
        true_gamma = (true_theta_in - true_theta_out) / (log(true_theta_in) - log(true_theta_out))

        # compute ground truth omega. For some reason, Pamfil et al. scale this by 1/2 (perhaps due to the directedness
        # of the interlayer edges), so we do the same here
        true_omega = log(1 + copying_probability * K / (1 - copying_probability))
        true_omega /= (2 * (log(true_theta_in) - log(true_theta_out)))

        gamma, omega, _ = iterative_multilayer_resolution_parameter_estimation(G_intralayer, G_interlayer,
                                                                               layer_membership, gamma=1.0, omega=1.0,
                                                                               model='temporal')

        # check we converged close to the ground truth "correct" values
        self.assertLess(abs(true_gamma - gamma), tolerance)
        self.assertLess(abs(true_omega - omega), tolerance)

        # check multilevel parameter estimation as well
        # we never use this model, but it is a slight generalization of the temporal one
        gamma, omega, _ = iterative_multilayer_resolution_parameter_estimation(G_intralayer, G_interlayer,
                                                                               layer_membership, gamma=1.0, omega=1.0,
                                                                               model='multilevel')
        self.assertLess(abs(true_gamma - gamma), tolerance)
        self.assertLess(abs(true_omega - omega), tolerance)

    def retry_test_across_graph_sizes(self, test_fn, sizes=[250, 500], retries_per_size=3):
        """
        Rerun randomized tests of asymptotic convergence across increasingly larger graph sizes until one succeeds.

        This improves the performance of the test suite by automatically using smaller graphs for "easier" tests.
        """
        last_error = None
        for n in sizes:
            for _ in range(retries_per_size):
                try:
                    test_fn(n)
                    return
                except (AssertionError, TimeoutError) as e:
                    last_error = e
                except Exception as e:
                    # test failed in unusual way (perhaps keyboard interrupt?), raising immediately
                    raise e
        assert last_error is not None
        raise last_error

    def test_temporal_SBM_correct_convergence_varying_copying_probabilty(self):
        for weighted in [False, True]:
            for eta in [0.25, 0.5, 0.75, 0.9]:
                with self.subTest(eta=eta, weighted=weighted):
                    self.retry_test_across_graph_sizes(
                        test_fn=lambda n, w=weighted: self.assert_temporal_SBM_correct_convergence(
                            copying_probability=eta,
                            first_layer_membership=generate_random_partition(num_nodes=n, K=2),
                            weighted=w
                        )
                    )

    def test_temporal_SBM_correct_convergence_varying_p_in(self):
        for weighted in [False, True]:
            for p_in in [0.5, 0.4, 0.3, 0.2]:
                with self.subTest(p_in=p_in, weighted=weighted):
                    self.retry_test_across_graph_sizes(
                        test_fn=lambda n, w=weighted: self.assert_temporal_SBM_correct_convergence(
                            p_in=p_in, p_out=0.025,
                            first_layer_membership=generate_random_partition(num_nodes=n, K=2),
                            weighted=w
                        )
                    )

    def test_temporal_SBM_correct_convergence_varying_p_out(self):
        for weighted in [False, True]:
            for p_out in [0.05, 0.04, 0.03, 0.02]:
                with self.subTest(p_out=p_out, weighted=weighted):
                    self.retry_test_across_graph_sizes(
                        test_fn=lambda n, w=weighted: self.assert_temporal_SBM_correct_convergence(
                            p_out=p_out, first_layer_membership=generate_random_partition(num_nodes=n, K=2),
                            weighted=w
                        )
                    )

    def test_temporal_SBM_correct_convergence_varying_num_communities(self):
        for weighted in [False, True]:
            for K in [2, 3, 4, 5]:
                with self.subTest(K=K, weighted=weighted):
                    self.retry_test_across_graph_sizes(
                        test_fn=lambda n, K_=K, w=weighted: self.assert_temporal_SBM_correct_convergence(
                            first_layer_membership=generate_random_partition(num_nodes=n, K=K_),
                            weighted=w
                        )
                    )

    def test_temporal_SBM_correct_convergence_varying_num_layers(self):
        for weighted in [False, True]:
            for num_layers in [8, 12, 16]:
                with self.subTest(num_layers=num_layers, weighted=weighted):
                    self.retry_test_across_graph_sizes(
                        test_fn=lambda n, nl=num_layers, w=weighted: self.assert_temporal_SBM_correct_convergence(
                            num_layers=nl,
                            first_layer_membership=generate_random_partition(num_nodes=n, K=2),
                            weighted=w
                        )
                    )

    def test_directed_temporal_SBM_correct_convergence(self):
        """Directed analogue of the temporal/multilevel convergence tests."""
        for weighted in [False, True]:
            with self.subTest(weighted=weighted):
                self.retry_test_across_graph_sizes(
                    test_fn=lambda n, w=weighted: self.assert_temporal_SBM_correct_convergence(
                        copying_probability=0.75, num_layers=8,
                        first_layer_membership=generate_random_partition(num_nodes=n, K=3), weighted=w,
                        directed=True)
                )

    def test_directed_asymmetry_temporal(self):
        """Simple test to confirm multilayer parameter estimates depend on edge direction.

        The intralayer graph has all edges pointing from the first community to the second."""
        one_way_edges = [(0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3)]
        edge_list = one_way_edges + [(u + 4, v + 4) for (u, v) in one_way_edges]
        weights = [1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        membership = [0, 0, 1, 1, 0, 0, 1, 1]
        layer_membership = [0, 0, 0, 0, 1, 1, 1, 1]
        G_interlayer = ig.Graph([(0, 4), (1, 5), (2, 6), (3, 7)], directed=True)

        for model in ['temporal', 'multilevel']:
            with self.subTest(model=model):
                G = ig.Graph(8, edge_list, directed=True)
                G.es['weight'] = weights
                gamma_directed, _ = gamma_omega_estimate(G, G_interlayer, layer_membership, membership, model=model)

                # undirected version for comparison
                G_undirected = ig.Graph(8, edge_list, directed=False)
                G_undirected.es['weight'] = weights
                gamma_undirected_treatment, _ = gamma_omega_estimate(G_undirected, G_interlayer, layer_membership,
                                                                     membership, model=model)
                self.assertGreater(abs(gamma_directed - gamma_undirected_treatment), 1e-2)

    def test_directed_consistency_temporal_SBM_leiden(self):
        """Test parameter estimate consistency on a temporal SBM when the intralayer edges are directed."""
        membership = [0] * 25 + [1] * 25 + [2] * 25
        G_intralayer, G_interlayer, layer_membership = generate_temporal_SBM(copying_probability=0.9,
                                                                             p_in=0.25, p_out=0.05,
                                                                             first_layer_membership=membership,
                                                                             num_layers=25)

        partitions = repeated_leiden_from_gammas_omegas(G_intralayer, G_interlayer, layer_membership,
                                                        gammas=[0.5, 1.0, 1.5], omegas=[0.5, 1.0, 1.5])

        for i, partition in enumerate(partitions):
            with self.subTest(partition=i):
                # here, undirected/directed refers to the intralayer edges only
                # in Pamfil et al.'s temporal networks, interlayer edges are taken to be directed
                G_intralayer.to_undirected()
                gamma_undirected, omega_undirected = gamma_omega_estimate(G_intralayer, G_interlayer, layer_membership,
                                                                          partition, model="temporal")

                G_intralayer.to_directed()
                gamma_directed, omega_directed = gamma_omega_estimate(G_intralayer, G_interlayer, layer_membership,
                                                                      partition, model="temporal")

                assert_almost_equal_or_both_none_or_nan(self, gamma_undirected, gamma_directed, places=10)
                assert_almost_equal_or_both_none_or_nan(self, omega_undirected, omega_directed, places=10)

                # check multilevel parameter estimation as well
                G_intralayer.to_undirected()
                gamma_undirected, omega_undirected = gamma_omega_estimate(G_intralayer, G_interlayer, layer_membership,
                                                                          partition, model="multilevel")

                G_intralayer.to_directed()
                gamma_directed, omega_directed = gamma_omega_estimate(G_intralayer, G_interlayer, layer_membership,
                                                                      partition, model="multilevel")

                assert_almost_equal_or_both_none_or_nan(self, gamma_undirected, gamma_directed, places=10)
                assert_almost_equal_or_both_none_or_nan(self, omega_undirected, omega_directed, places=10)


if __name__ == "__main__":
    seed(0)
    unittest.main()
