import igraph as ig
import math
from utilities import *
from scipy.special import comb
import sys

N = int(sys.argv[1])
B = N // 3
m_out2 = float(sys.argv[2])
print(N, m_out2)

while True:
    p_in1 = 10 * B / (2 * comb(B, 2))
    p_in2 = 7.5 * B / (2 * comb(B, 2))
    p_out1 = 1.25 / (2 * B)
    p_out2 = m_out2 / B

    pref_matrix = [[p_in1, p_out1, p_out1],
                   [p_out1, p_in2, p_out2],
                   [p_out1, p_out2, p_in2]]
    block_sizes = [N // 3] * 3
    G = ig.Graph.SBM(N, pref_matrix, block_sizes)

    if not G.is_connected():
        print("\rDisconnected graph. Skipping...", end='', flush=True)
        continue

    GAMMA_START = 0.0
    GAMMA_END = 2.0

    gamma2s = []
    gamma3s = []
    gamma2_errs = [[], []]
    gamma3_errs = [[], []]
    ITERATION_LIST = [int(x) for x in np.linspace(50, 500, 20)]
    REPEAT = 50

    for iterations in ITERATION_LIST:
        progress = Progress(REPEAT, name="iters={}".format(iterations))
        g2s = []
        g3s = []
        for repeat in range(REPEAT):
            all_parts = repeated_louvain(G, GAMMA_START, GAMMA_END,
                                         gamma_iters=iterations, repeat=1,
                                         show_progress=False)
            ranges = CHAMP_2D(G, all_parts, GAMMA_START, GAMMA_END)
            gamma_estimates = ranges_to_gamma_estimates(G, ranges)

            for g_start, g_end, membership, g_est in gamma_estimates:
                if g_est is not None:
                    if num_communities(membership) == 2:
                        g2s.append(g_est)
                    elif num_communities(membership) == 3:
                        g3s.append(g_est)

            progress.increment()

        gamma2s.append(np.mean(g2s))
        gamma3s.append(np.mean(g3s))
        gamma2_errs[0].append(np.mean(g2s) - min(g2s))
        gamma3_errs[0].append(np.mean(g3s) - min(g3s))
        gamma2_errs[1].append(max(g2s) - np.mean(g2s))
        gamma3_errs[1].append(max(g3s) - np.mean(g3s))
        progress.done()

    print(ITERATION_LIST)
    print(gamma2s)
    print(gamma2_errs)
    print(gamma3s)
    print(gamma3_errs)
    # plt.errorbar(ITERATION_LIST, gamma2s, gamma2_errs, label="2 community gamma estimates")
    # plt.errorbar(ITERATION_LIST, gamma3s, gamma3_errs, label="3 community gamma estimates")
    # plt.xlabel("number of louvain runs")
    # plt.ylabel("gamma")
    # plt.legend()
    # plt.title("???")
    # plt.show()
    break
