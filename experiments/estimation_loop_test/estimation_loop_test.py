# Attempts to generate a parameter estimation loop example with Louvain

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from modularitypruning.champ_utilities import CHAMP_2D
from modularitypruning.louvain_utilities import singlelayer_louvain
from modularitypruning.parameter_estimation_utilities import ranges_to_gamma_estimates
from modularitypruning.plotting import plot_estimates
from modularitypruning.progress import Progress
import sys

if __name__ == "__main__":
    NUM_TRIALS = int(sys.argv[1])
    progress = Progress(NUM_TRIALS)
    num_trials_completed = 0

    for i in range(NUM_TRIALS):
        progress.increment()

        GAMMA_START = 0
        GAMMA_END = 3
        GAMMA_ITERS = 1000

        n = np.random.randint(10, 100)
        m = np.random.randint(3 * n, 10 * n)
        try:
            G = ig.Graph.Erdos_Renyi(n=n, m=m, directed=False)
        except ig.InternalError:
            continue

        if not G.is_connected():
            continue

        gammas = np.linspace(GAMMA_START, GAMMA_END, GAMMA_ITERS)
        all_parts = [singlelayer_louvain(G, g) for g in gammas]

        if len(all_parts) == 0:
            continue

        # Print details on the CHAMP set when the number of communities is not restricted
        ranges = CHAMP_2D(G, all_parts, GAMMA_START, GAMMA_END, single_threaded=True)
        gamma_estimates = ranges_to_gamma_estimates(G, ranges)

        def gamma_to_domain(gamma):
            for gamma_start, gamma_end, part, gamma_est in gamma_estimates:
                if gamma_start <= gamma <= gamma_end:
                    return part, gamma_est
            return None, None

        for i in range(len(gamma_estimates)):
            _, _, old_membership, old_estimate = gamma_estimates[i]

            for update_iteration in range(1000):
                if old_estimate is None:
                    break

                new_membership, new_estimate = gamma_to_domain(old_estimate)
                if old_membership == new_membership:
                    break

                old_membership, old_estimate = new_membership, new_estimate
            else:
                print(f"loop detected with n={n}, m={m}")
                plot_estimates(gamma_estimates)
                plt.show()
                progress.done()
                sys.exit(0)

        num_trials_completed += 1

        # plt.close()
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        # plot_estimates(gamma_estimates)
        # plt.title(f"$G(n={n}, m={m})$ Realization Domains and Estimates")
        # plt.xlabel("$\gamma$", fontsize=14)
        # plt.ylabel("Number of communities", fontsize=14)
        # plt.xlim([-0.1, 3.1])
        # plt.savefig(f"G({n},{m}).pdf")

    progress.done()
    print(f"Actually completed {num_trials_completed} trials (out of {NUM_TRIALS})")
