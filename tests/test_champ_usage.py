import igraph as ig
from modularitypruning.champ_utilities import CHAMP_2D
from modularitypruning.louvain_utilities import louvain_part_with_membership
from random import randint, uniform, seed
import unittest


class TestCHAMP2D(unittest.TestCase):
    def generate_connected_ER(self, n, m, directed):
        G = ig.Graph.Erdos_Renyi(n=n, m=m, directed=directed)
        while not G.is_connected():
            G = ig.Graph.Erdos_Renyi(n=n, m=m, directed=directed)
        return G

    def generate_random_gammas(self, num_gammas, gamma_start, gamma_end):
        return [uniform(gamma_start, gamma_end) for _ in range(num_gammas)]

    def generate_random_partitions(self, num_nodes, num_partitions, K_max):
        partitions = []
        for _ in range(num_partitions):
            K = randint(1, K_max)
            part = tuple(randint(0, K - 1) for _ in range(num_nodes))
            partitions.append(part)
        return partitions

    def assert_best_partitions_match_champ_set(self, G, partitions, champ_ranges, gammas):
        membership_to_louvain_partition = {p: louvain_part_with_membership(G, p) for p in partitions}

        for gamma in gammas:
            best_partition_quality = max(membership_to_louvain_partition[p].quality(resolution_parameter=gamma)
                                         for p in partitions)
            for gamma_start, gamma_end, membership in champ_ranges:
                if gamma_start <= gamma <= gamma_end:
                    # check that the best partition quality matches that of the champ domains at this gamma
                    champ_quality = membership_to_louvain_partition[membership].quality(resolution_parameter=gamma)

                    # note that this is exact float equality
                    self.assertEqual(best_partition_quality, champ_quality,
                                     msg=f"CHAMP domain quality does not match best input partition "
                                         f"at gamma {gamma:.2f}")
                    break
            else:
                self.assertFalse(True, msg=f"gamma {gamma:.2f} was not found within any CHAMP domain")

    def assert_champ_correctness_unweighted_ER(self, n, m, directed, num_partitions, num_gammas, K_max,
                                               gamma_start, gamma_end):
        G = self.generate_connected_ER(n=n, m=m, directed=directed)
        partitions = self.generate_random_partitions(num_nodes=n, num_partitions=num_partitions, K_max=K_max)
        gammas = self.generate_random_gammas(num_gammas, gamma_start, gamma_end)
        champ_ranges = CHAMP_2D(G, partitions, gamma_start, gamma_end)

        self.assert_best_partitions_match_champ_set(G, partitions, champ_ranges, gammas)

    def test_champ_correctness_undirected_unweighted(self):
        seed(0)
        for n in [50, 100, 250, 500]:
            self.assert_champ_correctness_unweighted_ER(n=n, m=5 * n, directed=False, num_partitions=10,
                                                        num_gammas=100, K_max=5, gamma_start=0, gamma_end=2)
        for m in [200, 500, 1000]:
            self.assert_champ_correctness_unweighted_ER(n=100, m=m, num_partitions=10, directed=False,
                                                        num_gammas=100, K_max=5, gamma_start=0, gamma_end=2)

        for num_partitions in [10, 100, 1000]:
            self.assert_champ_correctness_unweighted_ER(n=100, m=500, num_partitions=num_partitions,
                                                        directed=False, num_gammas=100, K_max=5,
                                                        gamma_start=0, gamma_end=2)

        for K_max in [2, 5, 10, 20]:
            self.assert_champ_correctness_unweighted_ER(n=100, m=500, num_partitions=1000,
                                                        directed=False, num_gammas=100, K_max=K_max,
                                                        gamma_start=0, gamma_end=2)

        for gamma_start, gamma_end in zip([0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 9.0, 8.0, 7.0, 6.0]):
            self.assert_champ_correctness_unweighted_ER(n=100, m=500, num_partitions=1000,
                                                        directed=False, num_gammas=100, K_max=10,
                                                        gamma_start=gamma_start, gamma_end=gamma_end)

    def test_champ_correctness_directed_unweighted(self):
        seed(0)
        for n in [50, 100, 250, 500]:
            self.assert_champ_correctness_unweighted_ER(n=n, m=10 * n, directed=True, num_partitions=10,
                                                        num_gammas=100, K_max=5, gamma_start=0, gamma_end=2)
        for m in [400, 1000, 2000]:
            self.assert_champ_correctness_unweighted_ER(n=100, m=m, num_partitions=10, directed=True,
                                                        num_gammas=100, K_max=5, gamma_start=0, gamma_end=2)

        for num_partitions in [10, 100, 1000]:
            self.assert_champ_correctness_unweighted_ER(n=100, m=1000, num_partitions=num_partitions,
                                                        directed=True, num_gammas=100, K_max=5,
                                                        gamma_start=0, gamma_end=2)

        for K_max in [2, 5, 10, 20]:
            self.assert_champ_correctness_unweighted_ER(n=100, m=1000, num_partitions=1000,
                                                        directed=True, num_gammas=100, K_max=K_max,
                                                        gamma_start=0, gamma_end=2)

        for gamma_start, gamma_end in zip([0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 9.0, 8.0, 7.0, 6.0]):
            self.assert_champ_correctness_unweighted_ER(n=100, m=1000, num_partitions=1000,
                                                        directed=True, num_gammas=100, K_max=10,
                                                        gamma_start=gamma_start, gamma_end=gamma_end)

    # TODO: undirected & directed, weighted
    # TODO: multilayer with all combinations of undirected/directed/unweighted/weighted layers


if __name__ == "__main__":
    unittest.main()
