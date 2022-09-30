import pytest
import unittest


class TestDeprecatedLouvainNames(unittest.TestCase):
    def test_deprecated_louvain_module(self):
        with pytest.warns(DeprecationWarning):
            from modularitypruning import louvain_utilities  # noqa

    def test_shimmed_louvain_functions_in_old_module(self):
        # these only include the single-layer functions
        with pytest.warns(DeprecationWarning):
            from modularitypruning.louvain_utilities import louvain_part, louvain_part_with_membership, \
                repeated_louvain_from_gammas, repeated_parallel_louvain_from_gammas, singlelayer_louvain
        from modularitypruning.leiden_utilities import leiden_part, leiden_part_with_membership, \
            repeated_leiden_from_gammas, repeated_parallel_leiden_from_gammas, singlelayer_leiden

        # check that the deprecated module is returning the new module's functions
        self.assertEqual(louvain_part, leiden_part)
        self.assertEqual(louvain_part_with_membership, leiden_part_with_membership)
        self.assertEqual(repeated_louvain_from_gammas, repeated_leiden_from_gammas)
        self.assertEqual(repeated_parallel_louvain_from_gammas, repeated_parallel_leiden_from_gammas)
        self.assertEqual(singlelayer_louvain, singlelayer_leiden)


if __name__ == "__main__":
    unittest.main()
