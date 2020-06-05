# Compares the NMI with ground truth from our pruning strategy to Louvain baselines on LFR benchmark networks

import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
from networkx.exception import ExceededMaxIterations
import numpy as np
from modularitypruning.louvain_utilities import repeated_louvain_from_gammas
from modularitypruning.parameter_estimation_utilities import gamma_estimate, prune_to_stable_partitions
from modularitypruning.partition_utilities import nmi, num_communities
from modularitypruning.progress import Progress
import igraph as ig
import pickle
from functools import wraps
import errno
import os
import signal


# Note: this only works on UNIX
def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


def LFR_benchmark_igraph(n, tau1, tau2, mu, average_degree):
    """Returns the LFR benchmark graph as discussed in "Benchmark graphs for testing community detection algorithms".

    We defer to the implementation in networkx here.

    Note real networks have typical values 2 < tau1 < 3 and 1 < tau2 < 2.

    :param n: number of nodes in the created graph
    :param tau1: power law exponent for the degree distribution
    :param tau2: power law exponent for the community size distribution
    :param mu: fraction of intra-community edges incident to each node
    :param average_degree: desired average degree of nodes in the created graph
    :return: created (igraph) graph, ground truth community vector
    """

    # this generation regularly fails (exceeds max iterations) and calling code must handle this
    # networkx's implementation also seems prone to hanging occasionally
    # (at the very least, the worst-case running time here appears to be more than 100x the average case)
    try:
        G = LFR_benchmark_graph(n=n, tau1=tau1, tau2=tau2, mu=mu, average_degree=average_degree, max_iters=100)
        communities = {frozenset(G.nodes[v]['community']) for v in G}
        if not nx.is_connected(G):
            return None, None
    except ExceededMaxIterations:
        return None, None

    community_vector = [None for _ in range(G.number_of_nodes())]
    for i, community_set in enumerate(communities):
        assert len(community_set) > 0
        for v in community_set:
            community_vector[v] = i

    for i, community_set in enumerate(communities):
        for v in community_set:
            assert community_vector[v] == i

    G_igraph = ig.Graph(edges=list(G.edges), directed=G.is_directed())
    assert not G.is_directed()
    assert all(c is not None for c in community_vector)
    assert G.number_of_nodes() == G_igraph.vcount()
    assert G.number_of_edges() == G_igraph.ecount()

    return G_igraph, community_vector


# timeout is necessary to work around pathological generation cases
# (see comment in LFR_benchmark_igraph)
@timeout(10)
def trial(mu):
    """Run one realization of our LFR benchmark graph generation

    :param mu: fraction of intra-community edges incident to each node in LFR graph
    :return: created (igraph) graph, ground truth community vector
    """
    G, ground_truth_communities = LFR_benchmark_igraph(n=500, tau1=2.5, tau2=1.5, mu=mu, average_degree=20)
    return G, ground_truth_communities


def run_method(G, ground_truth_communities, num_louvain_runs, method, gamma_sweep_min=0.5, gamma_sweep_max=2.0):
    """
    Run one trial of comparing our benchmark to typical Louvain strategies

    :param G: graph of interest
    :param ground_truth_communities: ground truth community vector
    :param method: "modularity pruning", "modularity pruning ground truth K", "gamma sweep" or "ground truth gamma"
    :return: list of NMIs compared to the ground truth communities
    """
    ground_truth_gamma = gamma_estimate(G, ground_truth_communities)

    if ground_truth_gamma > gamma_sweep_max:
        print(f"Ground truth gamma {ground_truth_gamma:.2f} is large")

    if ground_truth_gamma is None:
        raise ValueError("Cannot use a graph with degenerate ground truth communities")

    if method == "modularity pruning" or method == "modularity pruning ground truth K" or method == "gamma sweep":
        gammas = np.linspace(gamma_sweep_min, gamma_sweep_max, num_louvain_runs)
    elif method == "ground truth gamma":
        gammas = np.linspace(ground_truth_gamma, ground_truth_gamma, num_louvain_runs)
    else:
        raise ValueError(f"Option {method} is not valid")

    parts = repeated_louvain_from_gammas(G, gammas)

    if method == "modularity pruning":
        stable_parts = prune_to_stable_partitions(G, parts, gamma_start=gammas[0], gamma_end=gammas[-1],
                                                  single_threaded=True)
        nmis = [nmi(ground_truth_communities, p) for p in stable_parts]
    elif method == "modularity pruning ground truth K":
        ground_truth_K = num_communities(ground_truth_communities)
        stable_parts = prune_to_stable_partitions(G, parts, gamma_start=gammas[0], gamma_end=gammas[-1],
                                                  restrict_num_communities=ground_truth_K,
                                                  single_threaded=True)
        nmis = [nmi(ground_truth_communities, p) for p in stable_parts]
    else:  # method == "gamma sweep" or method == "ground truth gamma":
        nmis = [nmi(ground_truth_communities, p) for p in parts]

    return nmis


def generate_LFR_graphs(num_graphs, mu):
    filename = f"LFR_num_graphs={num_graphs}_mu={mu:.3f}.p"

    if not os.path.exists(filename):
        LFR_graphs = []
        while len(LFR_graphs) < num_graphs:
            pool = Pool(processes=cpu_count())
            try:
                for G, ground_truth_communities in pool.map(trial, [mu for _ in range(cpu_count())]):
                    if G is not None:
                        LFR_graphs.append((G, ground_truth_communities))
            except TimeoutError:
                # timeout is necessary to work around pathological generation cases
                # (see comment in LFR_benchmark_igraph)
                # print("TIMEOUT!!!")
                pass
            pool.close()
            # print(f"\r{filename} currently at length {len(LFR_graphs)}", end="", flush=True)

        # print()
        LFR_graphs = LFR_graphs[:num_graphs]
        assert len(LFR_graphs) == num_graphs
        pickle.dump(LFR_graphs, open(filename, "wb"))

    return pickle.load(open(filename, "rb"))


def run_experiment(num_louvain_runs, num_graphs_per_mu, mus):
    progress = Progress(len(mus))

    plot1_data = {}
    plot2_data = {}
    plot3_data = {}
    plot4_data = {}

    if os.path.exists("lfr_benchmark_results_partial.p"):
        plot1_data, plot2_data, plot3_data, plot4_data = pickle.load(open("lfr_benchmark_results_partial.p", "rb"))

    for mu in mus:
        if mu in plot1_data:
            progress.increment()
            continue

        plot1_data[mu] = []
        plot2_data[mu] = []
        plot3_data[mu] = []
        plot4_data[mu] = []

        LFR_graphs = generate_LFR_graphs(num_graphs_per_mu, mu)

        pool = Pool(processes=cpu_count())
        for data1 in pool.starmap(run_method, [(G, ground_truth_communities, num_louvain_runs,
                                                "modularity pruning")
                                               for G, ground_truth_communities in LFR_graphs]):
            plot1_data[mu].append(data1)

        for data2 in pool.starmap(run_method, [(G, ground_truth_communities, num_louvain_runs,
                                                "gamma sweep")
                                               for G, ground_truth_communities in LFR_graphs]):
            plot2_data[mu].append(data2)

        for data3 in pool.starmap(run_method, [(G, ground_truth_communities, num_louvain_runs,
                                                "modularity pruning ground truth K")
                                               for G, ground_truth_communities in LFR_graphs]):
            plot3_data[mu].append(data3)

        for data4 in pool.starmap(run_method, [(G, ground_truth_communities, num_louvain_runs,
                                                "ground truth gamma")
                                               for G, ground_truth_communities in LFR_graphs]):
            plot4_data[mu].append(data4)
        pool.close()

        progress.increment()
        pickle.dump((plot1_data, plot2_data, plot3_data, plot4_data), open("lfr_benchmark_results_partial.p", "wb"))

    progress.done()
    pickle.dump((plot1_data, plot2_data, plot3_data, plot4_data), open("lfr_benchmark_results.p", "wb"))


if __name__ == "__main__":
    num_graphs_per_mu = 100
    mus = np.linspace(0.1, 0.5, 17)

    for mu in mus:
        generate_LFR_graphs(num_graphs_per_mu, mu)

    if not os.path.exists("lfr_benchmark_results.p"):
        run_experiment(num_louvain_runs=500, num_graphs_per_mu=num_graphs_per_mu, mus=mus)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    ax_idx = 0

    plot1_data, plot2_data, plot3_data, plot4_data = pickle.load(open("lfr_benchmark_results.p", "rb"))
    for data_dict, label in [(plot1_data, "Modularity Pruning"),
                             (plot2_data, r"Louvain running with $\gamma \in [0.5, 2.0]$"),
                             (plot3_data, r"Modularity Pruning with ground truth $K$"),
                             (plot4_data, r"Louvain running with ground truth $\gamma$")]:
        mus = []
        medians = []
        yerrs = []
        fraction_with_data = []

        for mu, data_lists in data_dict.items():
            data = []
            num_runs_nonzero_data = 0
            for data_list in data_lists:
                data.extend(data_list)

                if len(data_list) > 0:
                    num_runs_nonzero_data += 1

            fraction_with_data.append(num_runs_nonzero_data / len(data_lists))

            if len(data) > 0:
                mus.append(mu)
                medians.append(np.median(data))
                yerrs.append(np.array([np.median(data) - np.percentile(data, 25),
                                       np.percentile(data, 75) - np.median(data)]))

        yerrs = np.array(yerrs).T

        ax = axs[1 - ax_idx % 2][ax_idx // 2]
        ax_idx += 1
        ax.set_title(label, fontsize=14)
        ax.errorbar(mus, medians, yerr=yerrs, label="NMI with ground truth", fmt='-o', markersize=5, capsize=5)
        ax.set_xlim([0.09, 0.51])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel(r"mixing parameter $\mu$", fontsize=14)

        if "Pruning" in label:
            ax.plot(list(data_dict.keys()), fraction_with_data, label="probability of finding stable partitions")

        ax.legend()

    plt.tight_layout()
    plt.savefig("lfr_benchmark_results.pdf")
