from random import random, randint
from collections import Counter
import igraph as ig
from champ.parameter_estimation import iterative_multilayer_resolution_parameter_estimation
import numpy as np
import pickle
from utilities import Progress


def generate_synthetic_network():
    eta = 0.7  # copying probability
    epsilon = 0.4  # p_in/p_out ratio
    p_in = 20 / 75
    p_out = epsilon * p_in

    n_per_layer = 150
    num_layers = 15
    K = 2

    comm_per_layer = [[0] * n_per_layer for _ in range(num_layers)]
    comm_per_layer[0] = [i // (n_per_layer // K) for i in range(n_per_layer)]

    layer_vec = [i // n_per_layer for i in range(n_per_layer * num_layers)]

    comm_counts = Counter(comm_per_layer[0])
    assert all(v == comm_counts[0] for v in comm_counts.values())

    for layer in range(1, num_layers):
        for v in range(n_per_layer):
            p = random()
            if p < eta:  # copy community from last layer
                comm_per_layer[layer][v] = comm_per_layer[layer - 1][v]
            else:  # assign random community
                comm_per_layer[layer][v] = randint(0, K - 1)

    comm_vec = [item for sublist in comm_per_layer for item in sublist]

    intralayer_edges = []
    interlayer_edges = [(n_per_layer * l + v, n_per_layer * l + v + n_per_layer)
                        for l in range(num_layers - 1)
                        for v in range(n_per_layer)]

    for v in range(len(comm_vec)):
        for u in range(v + 1, len(comm_vec)):
            if layer_vec[v] == layer_vec[u]:
                p = random()
                if comm_vec[v] == comm_vec[u]:
                    if p < p_in:
                        intralayer_edges.append((u, v))
                else:
                    if p < p_out:
                        intralayer_edges.append((u, v))

    G_intralayer = ig.Graph(intralayer_edges)
    G_interlayer = ig.Graph(interlayer_edges, directed=True)
    pickle.dump((G_intralayer, G_interlayer, comm_vec), open("easy_regime_multilayer.p", "wb"))


def run_pamfil_iteration():
    G_intralayer, G_interlayer, _ = pickle.load(open("easy_regime_multilayer.p", "rb"))

    n_per_layer = 150
    num_layers = 15
    layer_vec = [i // n_per_layer for i in range(n_per_layer * num_layers)]

    def one_step(gamma, omega):
        try:
            g_new, o_new, _ = iterative_multilayer_resolution_parameter_estimation(G_intralayer, G_interlayer,
                                                                                   layer_vec,
                                                                                   gamma=gamma, omega=omega, max_iter=1)
            return g_new, o_new
        except ValueError:
            return None, None

    values = []
    all_g0s = np.linspace(0.6, 1.45, 15)
    all_o0s = np.linspace(0.4, 1.6, 15)
    progress = Progress(len(all_g0s) * len(all_o0s))
    for g0 in all_g0s:
        for o0 in all_o0s:
            gdiffs = []
            odiffs = []
            REPEAT = 5
            for repeat in range(REPEAT):
                # print("\rgamma={:.3f}, omega={:.3f} Progress: [".format(g0, o0) +
                #       "#" * repeat + "-" * (REPEAT - repeat - 1) + "] ", end='', flush=True)
                g1, o1 = one_step(g0, o0)
                if g1 is not None and o1 is not None:
                    gdiffs.append(g1 - g0)
                    odiffs.append(o1 - o0)
            else:
                # if len(odiffs) > 0:
                #     print("has movement ({:.3f},{:.3f}) with count {}"
                #           "".format(np.mean(gdiffs), np.mean(odiffs), len(gdiffs)))
                # else:
                #     print()
                values.append((g0, o0, gdiffs, odiffs))

            progress.increment()

    progress.done()
    pickle.dump(values, open("easy_regime_test_results.p", "wb"))
    print(values)


run_pamfil_iteration()
