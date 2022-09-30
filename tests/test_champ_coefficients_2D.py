from .shared_testing_functions import generate_connected_ER, generate_random_values, generate_random_partitions, \
    generate_igraph_famous
from modularitypruning.champ_utilities import partition_coefficients_2D
from modularitypruning.leiden_utilities import leiden_part_with_membership, repeated_leiden_from_gammas
from random import seed
import unittest


class TestCHAMPCoefficients2D(unittest.TestCase):
    def assert_partition_coefficient_correctness(self, G, partitions, coefficients):
        A_hats, P_hats = coefficients

        for membership, A_hat, P_hat in zip(partitions, A_hats, P_hats):
            leiden_part = leiden_part_with_membership(G, membership)

            # Q(gamma=0) = sum_{ij} A_{ij} delta(c_i, c_j) = A_hat
            leiden_A_hat = leiden_part.quality(resolution_parameter=0)

            # Q(gamma=0) - Q(gamma=1)
            #   = sum_{ij} (A_{ij} - gamma*P_{ij} - A_{ij}) delta(c_i, c_j)
            #   = sum_{ij} P_{ij} delta(c_i, c_j)
            #   = P_hat
            leiden_P_hat = leiden_A_hat - leiden_part.quality(resolution_parameter=1)

            self.assertAlmostEqual(A_hat, leiden_A_hat, places=10)
            self.assertAlmostEqual(P_hat, leiden_P_hat, places=10)

    def assert_partition_coefficient_correctness_unweighted_ER(self, n=100, m=500, directed=False,
                                                               num_partitions=10, K_max=5):
        G = generate_connected_ER(n=n, m=m, directed=directed)
        partitions = generate_random_partitions(num_nodes=n, num_partitions=num_partitions, K_max=K_max)
        coefficients = partition_coefficients_2D(G, partitions)
        self.assert_partition_coefficient_correctness(G, partitions, coefficients)

    def test_partition_coefficient_correctness_undirected_unweighted_varying_n(self):
        for n in [50, 100, 250, 500]:
            self.assert_partition_coefficient_correctness_unweighted_ER(n=n, m=5 * n)

    def test_partition_coefficient_correctness_undirected_unweighted_varying_m(self):
        for m in [200, 500, 1000]:
            self.assert_partition_coefficient_correctness_unweighted_ER(m=m)

    def test_partition_coefficient_correctness_undirected_unweighted_varying_num_partitions(self):
        for num_partitions in [10, 100, 1000]:
            self.assert_partition_coefficient_correctness_unweighted_ER(num_partitions=num_partitions)

    def test_partition_coefficient_correctness_undirected_unweighted_varying_K_max(self):
        for K_max in [2, 5, 10, 20]:
            self.assert_partition_coefficient_correctness_unweighted_ER(num_partitions=100, K_max=K_max)

    def test_partition_coefficient_correctness_directed_unweighted_varying_n(self):
        for n in [50, 100, 250, 500]:
            self.assert_partition_coefficient_correctness_unweighted_ER(n=n, m=10 * n, directed=True)

    def test_partition_coefficient_correctness_directed_unweighted_varying_m(self):
        for m in [400, 1000, 2000]:
            self.assert_partition_coefficient_correctness_unweighted_ER(m=m, directed=True)

    def test_partition_coefficient_correctness_directed_unweighted_varying_num_partitions(self):
        for num_partitions in [10, 100, 1000]:
            self.assert_partition_coefficient_correctness_unweighted_ER(directed=True, num_partitions=num_partitions)

    def test_partition_coefficient_correctness_directed_unweighted_varying_K_max(self):
        for K_max in [2, 5, 10, 20]:
            self.assert_partition_coefficient_correctness_unweighted_ER(directed=True, num_partitions=100, K_max=K_max)

    def test_partition_coefficient_correctness_igraph_famous_leiden(self):
        """Test partition coefficient correctness on various famous graphs while obtaining partitions via Leiden.

        The correctness is checked for the original undirected and (symmetric) directed variants.
        """

        for G in generate_igraph_famous():
            gammas = generate_random_values(100, start_value=0, end_value=5)
            partitions = repeated_leiden_from_gammas(G, gammas)
            coefficients = partition_coefficients_2D(G, partitions)
            self.assert_partition_coefficient_correctness(G, partitions, coefficients)

            G.to_directed()  # check the directed version of the graph as well
            partitions = repeated_leiden_from_gammas(G, gammas)
            coefficients = partition_coefficients_2D(G, partitions)
            self.assert_partition_coefficient_correctness(G, partitions, coefficients)


if __name__ == "__main__":
    seed(0)
    unittest.main()
