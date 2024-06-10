import functools
import unittest
import warnings
from multiprocessing import Pool, cpu_count
from random import seed
from time import time

import igraph as ig
import numpy as np
import psutil
import pytest
from modularitypruning.leiden_utilities import (repeated_leiden_from_gammas, repeated_parallel_leiden_from_gammas,
                                                repeated_leiden_from_gammas_omegas,
                                                repeated_parallel_leiden_from_gammas_omegas)

from .shared_testing_functions import generate_connected_ER, generate_multilayer_intralayer_SBM

# this set of tests ensures that we achieve >= 75% parallel performance compared to perfect scaling of
# single-threaded jobs to multiple cores (with no memory contention). This threshold will be decreased in
# determine_target_parallelization_speedup() if the background CPU utilization exceeds 20%.
PERFORMANCE_TARGET_RELATIVE_TO_PERFECT_SCALING = 0.75


def mock_calculation(_):
    """A mock calculation that provides enough work to make serialization overhead negligible."""
    return sum(range(10 ** 7))


@functools.lru_cache(maxsize=1)
def determine_target_parallelization_speedup(num_calculations=32):
    """
    Calculate the parallelization speedup on mock_calculation to benchmark our implementation against.

    This performs
      * ``num_calculations`` function calls in the single-threaded case, and
      * ``num_calculations * cpu_count()`` calls in the multi-processed case

    Due in part to frequency scaling and simple memory contention, leidenalg over multiple processes (completely
    outside of Python or multiprocessing.Pool) seems to run at around (90% * core count) speedup on modern systems when
    hyper-threading is disabled.
    """
    global PERFORMANCE_TARGET_RELATIVE_TO_PERFECT_SCALING

    cpu_utilization = psutil.cpu_percent(interval=5)
    if cpu_utilization > 20:
        PERFORMANCE_TARGET_RELATIVE_TO_PERFECT_SCALING = 0.5
        warnings.warn(f"System CPU utilization is non-negligible during parallel performance test! "
                      f"Dropping performance scaling target to 50%.")

    start_time = time()
    _ = [mock_calculation(i) for i in range(num_calculations)]
    base_duration = time() - start_time

    num_pool_calculations = num_calculations * cpu_count()
    with Pool(processes=cpu_count()) as pool:
        pool.map(mock_calculation, range(cpu_count()))  # force pool initialization and basic burn-in

        start_time = time()
        pool.map(mock_calculation, range(num_pool_calculations))
        pool_duration = time() - start_time

    return num_pool_calculations / num_calculations * base_duration / pool_duration


@pytest.mark.serial  # these tests have to run serially for the parallel performance comparisons to make sense
class TestParallelLeidenPerformance(unittest.TestCase):
    @staticmethod
    def run_singlelayer_graph_parallelization(G, gammas):
        target_speedup = determine_target_parallelization_speedup()

        start_time = time()
        _ = repeated_leiden_from_gammas(G, gammas)
        duration = time() - start_time

        pool_gammas = np.linspace(min(gammas), max(gammas), len(gammas) * cpu_count())
        start_time = time()
        _ = repeated_parallel_leiden_from_gammas(G, pool_gammas)
        pool_duration = time() - start_time

        speedup = len(pool_gammas) / len(gammas) * duration / pool_duration
        return speedup / target_speedup

    @staticmethod
    def run_multilayer_graph_parallelization(G_intralayer, G_interlayer, layer_membership, gammas, omegas):
        target_speedup = determine_target_parallelization_speedup()

        start_time = time()
        _ = repeated_leiden_from_gammas_omegas(G_intralayer, G_interlayer, layer_membership, gammas, omegas)
        duration = time() - start_time

        pool_gammas = np.linspace(min(gammas), max(gammas), int(len(gammas) * np.sqrt(cpu_count())))
        pool_omegas = np.linspace(min(omegas), max(omegas), int(len(omegas) * np.sqrt(cpu_count())))
        start_time = time()
        _ = repeated_parallel_leiden_from_gammas_omegas(
            G_intralayer, G_interlayer, layer_membership, pool_gammas, pool_omegas
        )
        pool_duration = time() - start_time

        speedup = len(pool_gammas) * len(pool_omegas) / len(gammas) / len(omegas) * duration / pool_duration
        return speedup / target_speedup

    def test_tiny_singlelayer_graph_many_runs(self):
        """Single-threaded equivalent is 25k runs on G(n=34, m=78)."""
        G = ig.Graph.Famous("Zachary")
        gammas = np.linspace(0.0, 4.0, 25000)
        parallelization = self.run_singlelayer_graph_parallelization(G, gammas)
        self.assertGreater(parallelization, PERFORMANCE_TARGET_RELATIVE_TO_PERFECT_SCALING)

    def test_larger_singlelayer_graph_few_runs(self):
        """Single-threaded equivalent is 50 runs on G(n=10000, m=40000)."""
        G = generate_connected_ER(n=10000, m=40000, directed=False)
        gammas = np.linspace(0.0, 2.0, 50)
        parallelization = self.run_singlelayer_graph_parallelization(G, gammas)
        self.assertGreater(parallelization, PERFORMANCE_TARGET_RELATIVE_TO_PERFECT_SCALING)

    def test_tiny_multilayer_graph_many_runs(self):
        """Single-threaded equivalent is 10k runs on G(n=50, m=150)."""
        G_intralayer, layer_membership = generate_multilayer_intralayer_SBM(
            copying_probability=0.9, p_in=0.8, p_out=0.2, first_layer_membership=[0] * 5 + [1] * 5, num_layers=5
        )
        interlayer_edges = [(10 * layer + v, 10 * layer + v + 10)
                            for layer in range(5 - 1) for v in range(10)]
        G_interlayer = ig.Graph(interlayer_edges, directed=True)

        gammas = np.linspace(0.0, 2.0, 100)
        omegas = np.linspace(0.0, 2.0, 100)
        parallelization = self.run_multilayer_graph_parallelization(G_intralayer, G_interlayer,
                                                                    layer_membership, gammas, omegas)
        self.assertGreater(parallelization, PERFORMANCE_TARGET_RELATIVE_TO_PERFECT_SCALING)

    def test_larger_multilayer_graph_few_runs(self):
        """Single-threaded equivalent is 49 runs on approximately G(n=2500, m=15000)."""
        G_intralayer, layer_membership = generate_multilayer_intralayer_SBM(
            copying_probability=0.9, p_in=0.15, p_out=0.05, first_layer_membership=[0] * 50 + [1] * 50, num_layers=25
        )
        interlayer_edges = [(100 * layer + v, 100 * layer + v + 100)
                            for layer in range(25 - 1) for v in range(100)]
        G_interlayer = ig.Graph(interlayer_edges, directed=True)

        gammas = np.linspace(0.0, 2.0, 7)
        omegas = np.linspace(0.0, 2.0, 7)
        parallelization = self.run_multilayer_graph_parallelization(G_intralayer, G_interlayer,
                                                                    layer_membership, gammas, omegas)
        self.assertGreater(parallelization, PERFORMANCE_TARGET_RELATIVE_TO_PERFECT_SCALING)


if __name__ == "__main__":
    seed(0)
    unittest.main()
