from champ.louvain_ext import parallel_louvain
import igraph as ig
from time import time
from utilities import CHAMP_2D, repeated_parallel_louvain
from multiprocessing import cpu_count

G = ig.Graph.Famous("Zachary")


def test(PROCESSORS):
    GAMMA_0 = 0.0
    GAMMA_F = 2.0
    ITERATIONS = 100
    TIMEOUT = 60

    xs = []
    ys1 = []
    ys2 = []
    MC_duration = 0
    C_duration = 0

    while MC_duration < TIMEOUT:
        xs.append(ITERATIONS)
        print("{:>15} ".format(ITERATIONS), end='', flush=True)

        start = time()
        all_parts = repeated_parallel_louvain(G, GAMMA_0, GAMMA_F, gamma_iters=ITERATIONS // PROCESSORS,
                                              repeat=PROCESSORS,
                                              show_progress=False)
        results1 = CHAMP_2D(G, all_parts, GAMMA_0, GAMMA_F)
        MC_duration = time() - start

        ys1.append(MC_duration)
        print("{:>15.2f} ".format(MC_duration), end='', flush=True)

        if C_duration < TIMEOUT:
            start = time()
            results2 = parallel_louvain(G, GAMMA_0, GAMMA_F, numruns=ITERATIONS, numprocesses=PROCESSORS)
            C_duration = time() - start
            ys2.append(C_duration)
            print("{:>15.2f}".format(C_duration))
        else:
            print("{:>15.2f}".format(0))

        ITERATIONS = ITERATIONS + ITERATIONS // 2

    print(xs)
    print(ys1)
    print(ys2)


test(1)
test(cpu_count())
