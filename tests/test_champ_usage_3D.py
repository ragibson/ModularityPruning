from .shared_testing_functions import generate_connected_multilayer_ER, generate_random_values, \
    generate_random_partitions
from modularitypruning.champ_utilities import CHAMP_3D
from modularitypruning.leiden_utilities import multilayer_leiden_part_with_membership
from numpy import mean
from random import seed
import unittest


def point_is_inside_champ_domain(gamma, omega, domain_vertices):
    def left_or_right(x1, y1, x2, y2, x, y):
        """Returns whether the point (x,y) is to the left or right of the line between (x1, y1) and (x2, y2)."""
        return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1) >= 0

    centroid_x = mean([pt[0] for pt in domain_vertices])
    centroid_y = mean([pt[1] for pt in domain_vertices])

    for i in range(len(domain_vertices)):
        edge_start = domain_vertices[i]
        edge_end = domain_vertices[(i + 1) % len(domain_vertices)]

        # if (gamma, omega) is not on the same side as the (convex) polygon's centroid, it is outside the domain
        if (left_or_right(edge_start[0], edge_start[1], edge_end[0], edge_end[1], centroid_x, centroid_y) !=
                left_or_right(edge_start[0], edge_start[1], edge_end[0], edge_end[1], gamma, omega)):
            return False

    return True


class TestCHAMP3D(unittest.TestCase):
    def assert_best_partitions_match_champ_set(self, G_intralayer, G_interlayer, layer_membership, partitions,
                                               champ_domains, gammas, omegas):
        membership_to_intralayer_leiden_partitions = {}
        membership_to_interlayer_leiden_partition = {}
        for p in partitions:
            intralayer_parts, interlayer_part = multilayer_leiden_part_with_membership(G_intralayer, G_interlayer,
                                                                                       layer_membership, p)
            membership_to_intralayer_leiden_partitions[p] = intralayer_parts
            membership_to_interlayer_leiden_partition[p] = interlayer_part

        for gamma, omega in zip(gammas, omegas):
            best_partition_quality = max(
                sum(p_intra.quality(resolution_parameter=gamma)
                    for p_intra in membership_to_intralayer_leiden_partitions[p]) +
                omega * membership_to_interlayer_leiden_partition[p].quality(resolution_parameter=0)
                for p in partitions)

            for domain_vertices, membership in champ_domains:
                if point_is_inside_champ_domain(gamma, omega, domain_vertices):
                    # check that the best partition quality matches that of the champ domains at this gamma
                    champ_quality = (
                            sum(p_intra.quality(resolution_parameter=gamma)
                                for p_intra in membership_to_intralayer_leiden_partitions[membership])
                            + omega *
                            membership_to_interlayer_leiden_partition[membership].quality(resolution_parameter=0)
                    )

                    # note that this is float comparision to within ~10^{-10}
                    self.assertAlmostEqual(best_partition_quality, champ_quality, places=10,
                                           msg=f"CHAMP domain quality does not match best input partition "
                                               f"at gamma {gamma:.2f} omega {omega:.2f}")
                    break
            else:
                self.assertFalse(True, msg=f"gamma {gamma:.2f}, omega {omega:.2f} "
                                           f"was not found within any CHAMP domain")

    def assert_champ_correctness_unweighted_ER(self, num_nodes_per_layer=100, m=10000, num_layers=10, directed=False,
                                               num_partitions=10, num_sample_points=100, K_max=10,
                                               gamma_start=0.0, gamma_end=2.0, omega_start=0.0, omega_end=2.0):
        G_intralayer, G_interlayer, layer_membership = generate_connected_multilayer_ER(
            num_nodes_per_layer=num_nodes_per_layer, m=m, num_layers=num_layers, directed=directed)
        partitions = generate_random_partitions(num_nodes=G_intralayer.vcount(), num_partitions=num_partitions,
                                                K_max=K_max)
        gammas = generate_random_values(num_sample_points, start_value=gamma_start, end_value=gamma_end)
        omegas = generate_random_values(num_sample_points, start_value=omega_start, end_value=omega_end)
        champ_domains = CHAMP_3D(G_intralayer, G_interlayer, layer_membership, partitions,
                                 gamma_start, gamma_end, omega_start, omega_end)

        self.assert_best_partitions_match_champ_set(G_intralayer, G_interlayer, layer_membership, partitions,
                                                    champ_domains, gammas, omegas)

    def test_partition_coefficient_correctness_undirected_unweighted_varying_num_nodes_per_layer(self):
        for num_nodes_per_layer in [50, 100, 250, 500]:
            self.assert_champ_correctness_unweighted_ER(num_nodes_per_layer=num_nodes_per_layer,
                                                        m=50 * num_nodes_per_layer)

    def test_partition_coefficient_correctness_undirected_unweighted_varying_m(self):
        for m in [5000, 10000, 15000, 20000]:
            self.assert_champ_correctness_unweighted_ER(m=m)

    def test_partition_coefficient_correctness_undirected_unweighted_varying_num_layers(self):
        for num_layers in [5, 10, 20, 30]:
            self.assert_champ_correctness_unweighted_ER(m=15000, num_layers=num_layers)

    def test_partition_coefficient_correctness_undirected_unweighted_varying_num_partitions(self):
        for num_partitions in [5, 10, 100, 250]:
            self.assert_champ_correctness_unweighted_ER(num_partitions=num_partitions)

    def test_partition_coefficient_correctness_undirected_unweighted_varying_K_max(self):
        for K_max in [2, 5, 10, 25]:
            self.assert_champ_correctness_unweighted_ER(K_max=K_max)

    def test_partition_coefficient_correctness_undirected_unweighted_varying_gamma_range(self):
        for gamma_start, gamma_end in zip([0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 9.0, 8.0, 7.0, 6.0]):
            self.assert_champ_correctness_unweighted_ER(gamma_start=gamma_start, gamma_end=gamma_end)

    def test_partition_coefficient_correctness_undirected_unweighted_varying_omega_range(self):
        for omega_start, omega_end in zip([0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 9.0, 8.0, 7.0, 6.0]):
            self.assert_champ_correctness_unweighted_ER(omega_start=omega_start, omega_end=omega_end)

    def test_partition_coefficient_correctness_directed_unweighted_varying_num_nodes_per_layer(self):
        for num_nodes_per_layer in [50, 100, 250, 500]:
            self.assert_champ_correctness_unweighted_ER(num_nodes_per_layer=num_nodes_per_layer,
                                                        m=100 * num_nodes_per_layer, directed=True)

    def test_partition_coefficient_correctness_directed_unweighted_varying_m(self):
        for m in [10000, 20000, 30000, 40000]:
            self.assert_champ_correctness_unweighted_ER(m=m, directed=True)

    def test_partition_coefficient_correctness_directed_unweighted_varying_num_layers(self):
        for num_layers in [5, 10, 20, 30]:
            self.assert_champ_correctness_unweighted_ER(m=30000, num_layers=num_layers, directed=True)

    def test_partition_coefficient_correctness_directed_unweighted_varying_num_partitions(self):
        for num_partitions in [5, 10, 100, 250]:
            self.assert_champ_correctness_unweighted_ER(directed=True, num_partitions=num_partitions)

    def test_partition_coefficient_correctness_directed_unweighted_varying_K_max(self):
        for K_max in [2, 5, 10, 25]:
            self.assert_champ_correctness_unweighted_ER(directed=True, K_max=K_max)

    def test_partition_coefficient_correctness_directed_unweighted_varying_gamma_range(self):
        for gamma_start, gamma_end in zip([0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 9.0, 8.0, 7.0, 6.0]):
            self.assert_champ_correctness_unweighted_ER(directed=True,
                                                        gamma_start=gamma_start, gamma_end=gamma_end)

    def test_partition_coefficient_correctness_directed_unweighted_varying_omega_range(self):
        for omega_start, omega_end in zip([0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 9.0, 8.0, 7.0, 6.0]):
            self.assert_champ_correctness_unweighted_ER(directed=True,
                                                        omega_start=omega_start, omega_end=omega_end)


if __name__ == "__main__":
    seed(0)
    unittest.main()
