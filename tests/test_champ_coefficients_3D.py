import unittest
from random import seed

from modularitypruning.champ_utilities import partition_coefficients_3D
from modularitypruning.leiden_utilities import multilayer_leiden_part_with_membership, leiden_part_with_membership

from .shared_testing_functions import generate_connected_multilayer_ER, generate_random_partitions, \
    generate_temporal_SBM


class TestCHAMPCoefficients3D(unittest.TestCase):
    def assert_partition_coefficient_correctness(self, G_intralayer, G_interlayer, layer_membership,
                                                 partitions, coefficients):
        A_hats, P_hats, C_hats = coefficients

        for membership, A_hat, P_hat, C_hat in zip(partitions, A_hats, P_hats, C_hats):
            intralayer_parts, interlayer_part = multilayer_leiden_part_with_membership(G_intralayer, G_interlayer,
                                                                                       layer_membership,
                                                                                       community_membership=membership)

            # Q_intralayer(gamma=0) = sum_{ij} A_{ij} delta(c_i, c_j) = A_hat
            leiden_A_hat = sum(p.quality(resolution_parameter=0) for p in intralayer_parts)

            # Q_intralayer(gamma=0) - Q_intralayer(gamma=1)
            #   = sum_{ij} (A_{ij} - gamma*P_{ij} - A_{ij}) delta(c_i, c_j)
            #   = sum_{ij} P_{ij} delta(c_i, c_j)
            #   = P_hat
            leiden_P_hat = leiden_A_hat - sum(p.quality(resolution_parameter=1) for p in intralayer_parts)

            # Q_interlayer(omega=0)
            #   = sum_{ij} (C_{ij} - omega*P{ij}) delta(c_i, c_j)
            #   = sum_{ij} C_{ij} delta(c_i, c_j)
            #   = C_hat
            leiden_C_hat = interlayer_part.quality(resolution_parameter=0)

            self.assertAlmostEqual(A_hat, leiden_A_hat, places=8)
            self.assertAlmostEqual(P_hat, leiden_P_hat, places=8)
            self.assertAlmostEqual(C_hat, leiden_C_hat, places=8)

            # Also test against an alternate, per-layer calculation of P_hat
            alternate_P_hat = 0
            for layer in set(layer_membership):
                this_layer_indices = [i for i, l in enumerate(layer_membership) if layer == l]
                layer_subgraph = G_intralayer.subgraph(this_layer_indices)
                layer_community_membership = [community for i, community in enumerate(membership)
                                              if layer_membership[i] == layer]
                layer_part = leiden_part_with_membership(layer_subgraph, layer_community_membership)
                alternate_P_hat += (layer_part.quality(resolution_parameter=0.0) -
                                    layer_part.quality(resolution_parameter=1.0))
            self.assertAlmostEqual(alternate_P_hat, P_hat, places=8)

    def assert_partition_coefficient_correctness_ER(self, num_nodes_per_layer=100, m=25000, num_layers=10,
                                                    directed=False, num_partitions=10, K_max=10, weighted=False):
        G_intralayer, G_interlayer, layer_membership = generate_connected_multilayer_ER(
            num_nodes_per_layer=num_nodes_per_layer, m=m, num_layers=num_layers, directed=directed,
            weighted=weighted)
        partitions = generate_random_partitions(num_nodes=G_intralayer.vcount(), num_partitions=num_partitions,
                                                K_max=K_max)
        coefficients = partition_coefficients_3D(G_intralayer, G_interlayer, layer_membership, partitions)
        self.assert_partition_coefficient_correctness(G_intralayer, G_interlayer, layer_membership, partitions,
                                                      coefficients)

    def test_partition_coefficient_correctness_undirected_varying_num_nodes_per_layer(self):
        for num_nodes_per_layer in [50, 100, 250, 500]:
            for weighted in [False, True]:
                with self.subTest(num_nodes_per_layer=num_nodes_per_layer, weighted=weighted):
                    self.assert_partition_coefficient_correctness_ER(num_nodes_per_layer=num_nodes_per_layer,
                                                                     m=50 * num_nodes_per_layer,
                                                                     weighted=weighted)

    def test_partition_coefficient_correctness_undirected_varying_m(self):
        for m in [5000, 10000, 15000, 20000]:
            for weighted in [False, True]:
                with self.subTest(m=m, weighted=weighted):
                    self.assert_partition_coefficient_correctness_ER(m=m, weighted=weighted)

    def test_partition_coefficient_correctness_undirected_varying_num_layers(self):
        for num_layers in [5, 10, 20, 30]:
            for weighted in [False, True]:
                with self.subTest(num_layers=num_layers, weighted=weighted):
                    self.assert_partition_coefficient_correctness_ER(num_layers=num_layers, weighted=weighted)

    def test_partition_coefficient_correctness_undirected_varying_num_partitions(self):
        for num_partitions in [5, 10, 100, 250]:
            for weighted in [False, True]:
                with self.subTest(num_partitions=num_partitions, weighted=weighted):
                    self.assert_partition_coefficient_correctness_ER(num_partitions=num_partitions,
                                                                     weighted=weighted)

    def test_partition_coefficient_correctness_undirected_varying_K_max(self):
        for K_max in [2, 5, 10, 25]:
            for weighted in [False, True]:
                with self.subTest(K_max=K_max, weighted=weighted):
                    self.assert_partition_coefficient_correctness_ER(K_max=K_max, weighted=weighted)

    def test_partition_coefficient_correctness_directed_varying_num_nodes_per_layer(self):
        for num_nodes_per_layer in [50, 100, 250, 500]:
            for weighted in [False, True]:
                with self.subTest(num_nodes_per_layer=num_nodes_per_layer, weighted=weighted):
                    self.assert_partition_coefficient_correctness_ER(num_nodes_per_layer=num_nodes_per_layer,
                                                                     m=100 * num_nodes_per_layer, directed=True,
                                                                     weighted=weighted)

    def test_partition_coefficient_correctness_directed_varying_m(self):
        for m in [10000, 20000, 30000, 40000]:
            for weighted in [False, True]:
                with self.subTest(m=m, weighted=weighted):
                    self.assert_partition_coefficient_correctness_ER(m=m, directed=True, weighted=weighted)

    def test_partition_coefficient_correctness_directed_varying_num_layers(self):
        for num_layers in [5, 10, 20, 30]:
            for weighted in [False, True]:
                with self.subTest(num_layers=num_layers, weighted=weighted):
                    self.assert_partition_coefficient_correctness_ER(num_layers=num_layers, directed=True,
                                                                     weighted=weighted)

    def test_partition_coefficient_correctness_directed_varying_num_partitions(self):
        for num_partitions in [5, 10, 100, 250]:
            for weighted in [False, True]:
                with self.subTest(num_partitions=num_partitions, weighted=weighted):
                    self.assert_partition_coefficient_correctness_ER(directed=True, num_partitions=num_partitions,
                                                                     weighted=weighted)

    def test_partition_coefficient_correctness_directed_varying_K_max(self):
        for K_max in [2, 5, 10, 25]:
            for weighted in [False, True]:
                with self.subTest(K_max=K_max, weighted=weighted):
                    self.assert_partition_coefficient_correctness_ER(directed=True, K_max=K_max, weighted=weighted)

    def assert_partition_coefficient_correctness_SBM(self, num_layers, weighted):
        num_nodes_per_layer = 100
        first_layer_membership = [i // (num_nodes_per_layer // 3) for i in range(num_nodes_per_layer)]

        G_intralayer, G_interlayer, layer_membership = generate_temporal_SBM(
            copying_probability=0.9, p_in=0.25, p_out=0.05, first_layer_membership=first_layer_membership,
            num_layers=num_layers, weighted=weighted, directed=True)

        partitions = generate_random_partitions(num_nodes=G_intralayer.vcount(), num_partitions=100, K_max=3)
        coefficients = partition_coefficients_3D(G_intralayer, G_interlayer, layer_membership, partitions)

        self.assert_partition_coefficient_correctness(G_intralayer, G_interlayer, layer_membership, partitions,
                                                      coefficients)

    def test_partition_coefficient_correctness_directed_SBM(self):
        """Test on asymmetric directed SBMs, unweighted and weighted."""
        for num_layers in [4, 5]:
            for weighted in [False, True]:
                with self.subTest(num_layers=num_layers, weighted=weighted):
                    self.assert_partition_coefficient_correctness_SBM(num_layers=num_layers, weighted=weighted)

    def test_partition_coefficient_correctness_interleaved_directedness(self):
        """Test partition coefficient correctness when directedness of interlayer and intralayer edges do not match."""
        for weighted in [False, True]:
            # Intralayer directed edges, but interlayer undirected ones
            with self.subTest(intralayer_directed=True, weighted=weighted):
                G_intralayer, G_interlayer, layer_membership = generate_connected_multilayer_ER(
                    num_nodes_per_layer=100, m=5000, num_layers=10, directed=False, weighted=weighted)
                G_intralayer.to_directed()
                partitions = generate_random_partitions(num_nodes=G_intralayer.vcount(), num_partitions=10, K_max=10)
                coefficients = partition_coefficients_3D(G_intralayer, G_interlayer, layer_membership, partitions)
                self.assert_partition_coefficient_correctness(G_intralayer, G_interlayer, layer_membership, partitions,
                                                              coefficients)

            # Interlayer directed edges, but intralayer undirected ones
            with self.subTest(intralayer_directed=False, weighted=weighted):
                G_intralayer, G_interlayer, layer_membership = generate_connected_multilayer_ER(
                    num_nodes_per_layer=100, m=5000, num_layers=10, directed=False, weighted=weighted)
                G_interlayer.to_directed()
                partitions = generate_random_partitions(num_nodes=G_intralayer.vcount(), num_partitions=10, K_max=10)
                coefficients = partition_coefficients_3D(G_intralayer, G_interlayer, layer_membership, partitions)
                self.assert_partition_coefficient_correctness(G_intralayer, G_interlayer, layer_membership, partitions,
                                                              coefficients)


if __name__ == "__main__":
    seed(0)
    unittest.main()
