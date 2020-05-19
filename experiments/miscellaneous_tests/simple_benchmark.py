# This shows that in 60 seconds, our parallel implementation of Louvain runs about 10x faster than
# CHAMP's (~2.1M vs. ~207K partitions) when running on the Karate Club with 8 cores

from champ.louvain_ext import parallel_louvain
import igraph as ig
from time import time
from modularitypruning.champ_utilities import CHAMP_2D
from modularitypruning.louvain_utilities import repeated_parallel_louvain_from_gammas
from multiprocessing import cpu_count
import numpy as np

G = ig.Graph.Famous("Zachary")


def test(gamma_0=0.0, gamma_f=2.0, louvain_iterations=100, timeout=60):
    """Runs a simple benchmark on the Karate Club of our parallel louvain implementation vs. CHAMP's

    :param num_processors: number of CPUs to use
    :param gamma_0: starting gamma
    :param gamma_f: final gamma
    :param louvain_iterations: number of initial louvain iterations
    :param timeout: maximum allowed time (in seconds) for a set of louvain runs
    """

    xs = []
    ys1 = []
    ys2 = []
    our_duration = 0
    champ_duration = 0

    num_processors = cpu_count()
    print(f"Test with {num_processors} CPUs")
    print(f'{"# partitions":>15} {"Our time (s)":>15} {"CHAMP time (s)":>15}')

    while our_duration < timeout:
        xs.append(louvain_iterations)
        print(f"{louvain_iterations:>15} ", end='', flush=True)

        start = time()
        all_parts = repeated_parallel_louvain_from_gammas(G, gammas=np.linspace(gamma_0, gamma_f, louvain_iterations),
                                                          show_progress=False)
        _ = CHAMP_2D(G, all_parts, gamma_0, gamma_f)
        our_duration = time() - start

        ys1.append(our_duration)
        print(f"{our_duration:>15.2f} ", end='', flush=True)

        if champ_duration < timeout:
            start = time()
            _ = parallel_louvain(G, gamma_0, gamma_f, numruns=louvain_iterations, numprocesses=num_processors)
            champ_duration = time() - start
            ys2.append(champ_duration)
            print(f"{champ_duration:>15.2f}")
        else:
            print(f"{0:>15.2f}")

        # the number of louvain iterations roughly increases by 1.5x each iteration
        louvain_iterations = louvain_iterations + louvain_iterations // 2


if __name__ == "__main__":
    test()
