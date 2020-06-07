from shared_testing_functions import generate_connected_ER, generate_random_gammas, generate_random_partitions, \
    generate_igraph_famous
from modularitypruning.champ_utilities import CHAMP_2D, partition_coefficients_2D
from modularitypruning.louvain_utilities import louvain_part_with_membership, repeated_louvain_from_gammas
from random import seed
import unittest


class TestCHAMPCoefficients2D(unittest.TestCase):
    # TODO: undirected & directed, weighted (coefficients and CHAMP domains)
    # TODO: multilayer with undirected/directed/unweighted/weighted layers (coefficients and CHAMP domains)

    def assert_partition_coefficient_correctness(self, G, partitions, coefficients):
        A_hats, P_hats = coefficients

        for membership, A_hat, P_hat in zip(partitions, A_hats, P_hats):
            louvain_part = louvain_part_with_membership(G, membership)

            # Q(gamma=0) = sum_{ij} A_{ij} delta(c_i, c_j) = A_hat
            louvain_A_hat = louvain_part.quality(resolution_parameter=0)

            # Q(gamma=0) - Q(gamma=1)
            #   = sum_{ij} (A_{ij} - gamma*P_{ij} - A_{ij}) delta(c_i, c_j)
            #   = sum_{ij} P_{ij} delta(c_i, c_j)
            #   = P_hat
            louvain_P_hat = louvain_A_hat - louvain_part.quality(resolution_parameter=1)

            self.assertAlmostEqual(A_hat, louvain_A_hat)
            self.assertAlmostEqual(P_hat, louvain_P_hat)

    def assert_partition_coefficient_correctness_unweighted_ER(self, n, m, directed, num_partitions, K_max):
        G = generate_connected_ER(n=n, m=m, directed=directed)
        partitions = generate_random_partitions(num_nodes=n, num_partitions=num_partitions, K_max=K_max)
        coefficients = partition_coefficients_2D(G, partitions)
        self.assert_partition_coefficient_correctness(G, partitions, coefficients)

    def test_partition_coefficient_correctness_undirected_unweighted_varying_n(self):
        seed(0)
        for n in [50, 100, 250, 500]:
            self.assert_partition_coefficient_correctness_unweighted_ER(n=n, m=5 * n, directed=False,
                                                                        num_partitions=10, K_max=5)

    def test_partition_coefficient_correctness_undirected_unweighted_varying_m(self):
        seed(0)
        for m in [200, 500, 1000]:
            self.assert_partition_coefficient_correctness_unweighted_ER(n=100, m=m, directed=False,
                                                                        num_partitions=10, K_max=5)

    def test_partition_coefficient_correctness_undirected_unweighted_varying_num_partitions(self):
        seed(0)
        for num_partitions in [10, 100, 1000]:
            self.assert_partition_coefficient_correctness_unweighted_ER(n=100, m=500, directed=False,
                                                                        num_partitions=num_partitions, K_max=5)

    def test_partition_coefficient_correctness_undirected_unweighted_varying_K_max(self):
        seed(0)
        for K_max in [2, 5, 10, 20]:
            self.assert_partition_coefficient_correctness_unweighted_ER(n=100, m=500, directed=False,
                                                                        num_partitions=100, K_max=K_max)

    def test_partition_coefficient_correctness_directed_unweighted_varying_n(self):
        seed(0)
        for n in [50, 100, 250, 500]:
            self.assert_partition_coefficient_correctness_unweighted_ER(n=n, m=10 * n, directed=True,
                                                                        num_partitions=10, K_max=5)

    def test_partition_coefficient_correctness_directed_unweighted_varying_m(self):
        seed(0)
        for m in [400, 1000, 2000]:
            self.assert_partition_coefficient_correctness_unweighted_ER(n=100, m=m, directed=True,
                                                                        num_partitions=10, K_max=5)

    def test_partition_coefficient_correctness_directed_unweighted_varying_num_partitions(self):
        seed(0)
        for num_partitions in [10, 100, 1000]:
            self.assert_partition_coefficient_correctness_unweighted_ER(n=100, m=1000, directed=True,
                                                                        num_partitions=num_partitions, K_max=5)

    def test_partition_coefficient_correctness_directed_unweighted_varying_K_max(self):
        seed(0)
        for K_max in [2, 5, 10, 20]:
            self.assert_partition_coefficient_correctness_unweighted_ER(n=100, m=1000, directed=True,
                                                                        num_partitions=100, K_max=K_max)

    def test_partition_coefficient_correctness_igraph_famous_louvain(self):
        """Test partition coefficient correctness on various famous graphs while obtaining partitions via Louvain.

        The correctness is checked for the original undirected and (symmetric) directed variants.
        """

        seed(0)
        for G in generate_igraph_famous():
            gammas = generate_random_gammas(100, gamma_start=0, gamma_end=5)
            partitions = repeated_louvain_from_gammas(G, gammas)
            coefficients = partition_coefficients_2D(G, partitions)
            self.assert_partition_coefficient_correctness(G, partitions, coefficients)

            G.to_directed()  # check the directed version of the graph as well
            partitions = repeated_louvain_from_gammas(G, gammas)
            coefficients = partition_coefficients_2D(G, partitions)
            self.assert_partition_coefficient_correctness(G, partitions, coefficients)


class TestCHAMP2D(unittest.TestCase):
    # TODO: undirected & directed, weighted (coefficients and CHAMP domains)
    # TODO: multilayer with undirected/directed/unweighted/weighted layers (coefficients and CHAMP domains)

    def assert_best_partitions_match_champ_set(self, G, partitions, champ_ranges, gammas):
        membership_to_louvain_partition = {p: louvain_part_with_membership(G, p) for p in partitions}

        for gamma in gammas:
            best_partition_quality = max(membership_to_louvain_partition[p].quality(resolution_parameter=gamma)
                                         for p in partitions)
            for gamma_start, gamma_end, membership in champ_ranges:
                if gamma_start <= gamma <= gamma_end:
                    # check that the best partition quality matches that of the champ domains at this gamma
                    champ_quality = membership_to_louvain_partition[membership].quality(resolution_parameter=gamma)

                    # note that this is float comparision to within ~10^{-10}
                    self.assertAlmostEqual(best_partition_quality, champ_quality, places=10,
                                           msg=f"CHAMP domain quality does not match best input partition "
                                               f"at gamma {gamma:.2f}")
                    break
            else:
                self.assertFalse(True, msg=f"gamma {gamma:.2f} was not found within any CHAMP domain")

    def assert_champ_correctness_unweighted_ER(self, n, m, directed, num_partitions, num_gammas, K_max,
                                               gamma_start, gamma_end):
        G = generate_connected_ER(n=n, m=m, directed=directed)
        partitions = generate_random_partitions(num_nodes=n, num_partitions=num_partitions, K_max=K_max)
        gammas = generate_random_gammas(num_gammas, gamma_start, gamma_end)
        champ_ranges = CHAMP_2D(G, partitions, gamma_start, gamma_end)

        self.assert_best_partitions_match_champ_set(G, partitions, champ_ranges, gammas)

    def test_champ_correctness_undirected_unweighted_varying_n(self):
        seed(0)
        for n in [50, 100, 250, 500]:
            self.assert_champ_correctness_unweighted_ER(n=n, m=5 * n, directed=False, num_partitions=10,
                                                        num_gammas=100, K_max=5, gamma_start=0, gamma_end=2)

    def test_champ_correctness_undirected_unweighted_varying_m(self):
        seed(0)
        for m in [200, 500, 1000]:
            self.assert_champ_correctness_unweighted_ER(n=100, m=m, num_partitions=10, directed=False,
                                                        num_gammas=100, K_max=5, gamma_start=0, gamma_end=2)

    def test_champ_correctness_undirected_unweighted_varying_num_partitions(self):
        seed(0)
        for num_partitions in [10, 100, 1000]:
            self.assert_champ_correctness_unweighted_ER(n=100, m=500, num_partitions=num_partitions,
                                                        directed=False, num_gammas=100, K_max=5,
                                                        gamma_start=0, gamma_end=2)

    def test_champ_correctness_undirected_unweighted_varying_K_max(self):
        seed(0)
        for K_max in [2, 5, 10, 20]:
            self.assert_champ_correctness_unweighted_ER(n=100, m=500, num_partitions=100,
                                                        directed=False, num_gammas=100, K_max=K_max,
                                                        gamma_start=0, gamma_end=2)

    def test_champ_correctness_undirected_unweighted_varying_gamma_range(self):
        seed(0)
        for gamma_start, gamma_end in zip([0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 9.0, 8.0, 7.0, 6.0]):
            self.assert_champ_correctness_unweighted_ER(n=100, m=500, num_partitions=100,
                                                        directed=False, num_gammas=100, K_max=10,
                                                        gamma_start=gamma_start, gamma_end=gamma_end)

    def test_champ_correctness_directed_unweighted_varying_n(self):
        seed(0)
        for n in [50, 100, 250, 500]:
            self.assert_champ_correctness_unweighted_ER(n=n, m=10 * n, directed=True, num_partitions=10,
                                                        num_gammas=100, K_max=5, gamma_start=0, gamma_end=2)

    def test_champ_correctness_directed_unweighted_varying_m(self):
        seed(0)
        for m in [400, 1000, 2000]:
            self.assert_champ_correctness_unweighted_ER(n=100, m=m, num_partitions=10, directed=True,
                                                        num_gammas=100, K_max=5, gamma_start=0, gamma_end=2)

    def test_champ_correctness_directed_unweighted_varying_num_partitions(self):
        seed(0)
        for num_partitions in [10, 100, 1000]:
            self.assert_champ_correctness_unweighted_ER(n=100, m=1000, num_partitions=num_partitions,
                                                        directed=True, num_gammas=100, K_max=5,
                                                        gamma_start=0, gamma_end=2)

    def test_champ_correctness_directed_unweighted_varying_K_max(self):
        seed(0)
        for K_max in [2, 5, 10, 20]:
            self.assert_champ_correctness_unweighted_ER(n=100, m=1000, num_partitions=100,
                                                        directed=True, num_gammas=100, K_max=K_max,
                                                        gamma_start=0, gamma_end=2)

    def test_champ_correctness_directed_unweighted_varying_gamma_range(self):
        seed(0)
        for gamma_start, gamma_end in zip([0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 9.0, 8.0, 7.0, 6.0]):
            self.assert_champ_correctness_unweighted_ER(n=100, m=1000, num_partitions=100,
                                                        directed=True, num_gammas=100, K_max=10,
                                                        gamma_start=gamma_start, gamma_end=gamma_end)

    def test_champ_correctness_igraph_famous_louvain(self):
        """Test CHAMP correctness on various famous graphs while obtaining partitions via Louvain.

        The correctness of the CHAMP domains are checked for the original undirected and (symmetric) directed variants.
        """

        seed(0)
        for G in generate_igraph_famous():
            gammas = generate_random_gammas(100, gamma_start=0, gamma_end=5)
            partitions = repeated_louvain_from_gammas(G, gammas)
            champ_ranges = CHAMP_2D(G, partitions, gamma_0=0, gamma_f=5)
            self.assert_best_partitions_match_champ_set(G, partitions, champ_ranges, gammas)

            G.to_directed()  # check the directed version of the graph as well
            partitions = repeated_louvain_from_gammas(G, gammas)
            champ_ranges = CHAMP_2D(G, partitions, gamma_0=0, gamma_f=5)
            self.assert_best_partitions_match_champ_set(G, partitions, champ_ranges, gammas)


if __name__ == "__main__":
    unittest.main()
