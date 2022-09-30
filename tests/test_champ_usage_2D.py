from .shared_testing_functions import generate_connected_ER, generate_random_values, generate_random_partitions, \
    generate_igraph_famous
from modularitypruning.champ_utilities import CHAMP_2D
from modularitypruning.leiden_utilities import leiden_part_with_membership, repeated_leiden_from_gammas
from random import seed
import unittest


class TestCHAMP2D(unittest.TestCase):
    def assert_best_partitions_match_champ_set(self, G, partitions, champ_ranges, gammas):
        membership_to_leiden_partition = {p: leiden_part_with_membership(G, p) for p in partitions}

        for gamma in gammas:
            best_partition_quality = max(membership_to_leiden_partition[p].quality(resolution_parameter=gamma)
                                         for p in partitions)
            for gamma_start, gamma_end, membership in champ_ranges:
                if gamma_start <= gamma <= gamma_end:
                    # check that the best partition quality matches that of the champ domains at this gamma
                    champ_quality = membership_to_leiden_partition[membership].quality(resolution_parameter=gamma)

                    # note that this is float comparision to within ~10^{-10}
                    self.assertAlmostEqual(best_partition_quality, champ_quality, places=10,
                                           msg=f"CHAMP domain quality does not match best input partition "
                                               f"at gamma {gamma:.2f}")
                    break
            else:
                self.assertFalse(True, msg=f"gamma {gamma:.2f} was not found within any CHAMP domain")

    def assert_champ_correctness_unweighted_ER(self, n=100, m=1000, directed=False, num_partitions=10, num_gammas=100,
                                               K_max=5, gamma_start=0.0, gamma_end=2.0):
        G = generate_connected_ER(n=n, m=m, directed=directed)
        partitions = generate_random_partitions(num_nodes=n, num_partitions=num_partitions, K_max=K_max)
        gammas = generate_random_values(num_gammas, gamma_start, gamma_end)
        champ_ranges = CHAMP_2D(G, partitions, gamma_start, gamma_end)

        self.assert_best_partitions_match_champ_set(G, partitions, champ_ranges, gammas)

    def test_champ_correctness_undirected_unweighted_varying_n(self):
        for n in [50, 100, 250, 500]:
            self.assert_champ_correctness_unweighted_ER(n=n, m=5 * n)

    def test_champ_correctness_undirected_unweighted_varying_m(self):
        for m in [200, 500, 1000]:
            self.assert_champ_correctness_unweighted_ER(m=m)

    def test_champ_correctness_undirected_unweighted_varying_num_partitions(self):
        for num_partitions in [10, 100, 1000]:
            self.assert_champ_correctness_unweighted_ER(num_partitions=num_partitions)

    def test_champ_correctness_undirected_unweighted_varying_K_max(self):
        for K_max in [2, 5, 10, 20]:
            self.assert_champ_correctness_unweighted_ER(K_max=K_max)

    def test_champ_correctness_undirected_unweighted_varying_gamma_range(self):
        for gamma_start, gamma_end in zip([0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 9.0, 8.0, 7.0, 6.0]):
            self.assert_champ_correctness_unweighted_ER(gamma_start=gamma_start, gamma_end=gamma_end)

    def test_champ_correctness_directed_unweighted_varying_n(self):
        for n in [50, 100, 250, 500]:
            self.assert_champ_correctness_unweighted_ER(n=n, m=10 * n, directed=True)

    def test_champ_correctness_directed_unweighted_varying_m(self):
        for m in [400, 1000, 2000]:
            self.assert_champ_correctness_unweighted_ER(m=m, directed=True)

    def test_champ_correctness_directed_unweighted_varying_num_partitions(self):
        for num_partitions in [10, 100, 1000]:
            self.assert_champ_correctness_unweighted_ER(num_partitions=num_partitions, directed=True)

    def test_champ_correctness_directed_unweighted_varying_K_max(self):
        for K_max in [2, 5, 10, 20]:
            self.assert_champ_correctness_unweighted_ER(directed=True, K_max=K_max)

    def test_champ_correctness_directed_unweighted_varying_gamma_range(self):
        for gamma_start, gamma_end in zip([0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 9.0, 8.0, 7.0, 6.0]):
            self.assert_champ_correctness_unweighted_ER(directed=True, gamma_start=gamma_start, gamma_end=gamma_end)

    def test_champ_correctness_igraph_famous_leiden(self):
        """Test CHAMP correctness on various famous graphs while obtaining partitions via Leiden.

        The correctness of the CHAMP domains are checked for the original undirected and (symmetric) directed variants.
        """

        for G in generate_igraph_famous():
            gammas = generate_random_values(100, start_value=0, end_value=5)
            partitions = repeated_leiden_from_gammas(G, gammas)
            champ_ranges = CHAMP_2D(G, partitions, gamma_0=0, gamma_f=5)
            self.assert_best_partitions_match_champ_set(G, partitions, champ_ranges, gammas)

            G.to_directed()  # check the directed version of the graph as well
            partitions = repeated_leiden_from_gammas(G, gammas)
            champ_ranges = CHAMP_2D(G, partitions, gamma_0=0, gamma_f=5)
            self.assert_best_partitions_match_champ_set(G, partitions, champ_ranges, gammas)


if __name__ == "__main__":
    seed(0)
    unittest.main()
