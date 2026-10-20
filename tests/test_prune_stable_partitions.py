"""Simple tests of main pruning functions beyond the other testing scripts.

These cover:
- prune_to_stable_partitions
- prune_to_multilayer_stable_partitions
"""
import unittest
from random import seed

from modularitypruning import prune_to_multilayer_stable_partitions, prune_to_stable_partitions
from modularitypruning.leiden_utilities import multilayer_leiden, singlelayer_leiden

from .shared_testing_functions import generate_connected_ER, generate_temporal_SBM


class TestPruneToStablePartitions(unittest.TestCase):
    def test_prune_leiden_partitions_and_membership_tuples_agree(self):
        """Check that passing a list of leidenalg partitions operates the same as a list of membership tuples."""
        G = generate_connected_ER(n=50, m=200, directed=False)
        partitions = [singlelayer_leiden(G, gamma, return_partition=True)
                      for gamma in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]]
        membership_tuples = [tuple(part.membership) for part in partitions]

        stable_from_tuples = prune_to_stable_partitions(G, membership_tuples, 0, 2)
        stable_from_partitions = prune_to_stable_partitions(G, partitions, 0, 2)

        self.assertEqual(stable_from_partitions, stable_from_tuples)
        self.assertTrue(stable_from_tuples)  # non-empty


class TestPruneToMultilayerStablePartitions(unittest.TestCase):
    def test_prune_leiden_partitions_and_membership_tuples_agree(self):
        """Check that passing a list of leidenalg partitions operates the same as a list of membership tuples."""
        G_intralayer, G_interlayer, layer_vec = generate_temporal_SBM(
            copying_probability=0.75, p_in=0.6, p_out=0.05,
            first_layer_membership=[0] * 10 + [1] * 10, num_layers=2)

        partition_runs = [multilayer_leiden(G_intralayer, G_interlayer, layer_vec, gamma, omega, return_partition=True)
                          for gamma in [0.5, 1.0, 1.5, 2.0] for omega in [0.5, 1.0, 1.5, 2.0]]

        partitions = [run[0] for run in partition_runs]
        membership_tuples = [tuple(part.membership) for part in partitions]

        stable_from_tuples = prune_to_multilayer_stable_partitions(
            G_intralayer, G_interlayer, layer_vec, 'temporal', membership_tuples, 0, 2, 0, 2)
        stable_from_partitions = prune_to_multilayer_stable_partitions(
            G_intralayer, G_interlayer, layer_vec, 'temporal', partitions, 0, 2, 0, 2)

        self.assertEqual(stable_from_partitions, stable_from_tuples)
        self.assertTrue(stable_from_tuples)  # non-empty


if __name__ == "__main__":
    seed(0)
    unittest.main()
