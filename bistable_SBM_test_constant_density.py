import igraph as ig
from utilities import *
import pickle
from scipy.special import comb
import sys

ITERATIONS = 100
m2s = np.linspace(1, 5, 100).tolist()
c2s = []
c3s = []
cboths = []
progress = Progress(ITERATIONS * len(m2s))

N = int(sys.argv[1])
B = N // 3
print(N)

for m_out2 in m2s:
    total = 0
    count_2stable = 0
    count_3stable = 0
    count_both_stable = 0

    while total < ITERATIONS:
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

        # print("mean degree is", np.mean([G.degree(v) for v in range(N)]))
        ground_truth = tuple(i // block_sizes[0] for i in range(N))
        ground_truth2 = tuple(min(1, i // block_sizes[0]) for i in range(N))
        true_gamma = gamma_estimate(G, ground_truth)
        true_gamma2 = gamma_estimate(G, ground_truth2)

        if true_gamma is None or true_gamma2 is None:
            print("\rDegenerate ground truth estimate. Skipping...", end='', flush=True)
            continue

        # print("'true' gamma (3 block) is", true_gamma)
        # print("'true' gamma (2 block) is", true_gamma2)
        GAMMA_START = 0.0
        GAMMA_END = 2.0

        all_parts = repeated_parallel_louvain(G, GAMMA_START, GAMMA_END, gamma_iters=1000 // cpu_count() + 1,
                                              repeat=cpu_count(), show_progress=False)
        ranges = CHAMP_2D(G, all_parts, GAMMA_START, GAMMA_END)
        gamma_estimates = ranges_to_gamma_estimates(G, ranges)

        stable2, stable3 = False, False
        for g_start, g_end, membership, g_est in gamma_estimates:
            if g_est is not None and g_start <= g_est <= g_end:
                if num_communities(membership) == 2:
                    stable2 = True
                elif num_communities(membership) == 3:
                    stable3 = True

        if stable2:
            count_2stable += 1
        if stable3:
            count_3stable += 1
        if stable2 and stable3:
            count_both_stable += 1
        total += 1

        # plt.close()
        # plot_estimates(gamma_estimates)
        # plt.axvline(true_gamma, color='red', alpha=0.5, linestyle='dashed',
        #             label="gamma for ground truth 3 block partition")
        # plt.axvline(true_gamma2, color='blue', alpha=0.5, linestyle='dashed',
        #             label="gamma for ground truth 2 block partition")
        # plt.legend()
        # plt.show()

        progress.increment()

    c2s.append(count_2stable / total)
    c3s.append(count_3stable / total)
    cboths.append(count_both_stable / total)

progress.done()
print(m2s)
print(c2s)
print(c3s)
print(cboths)
# pickle.dump((p2s, c2s, c3s), open("stability_results.p", "wb"))
# plt.scatter(m2s, c2s, label="Probability of stable 2 community")
# plt.scatter(m2s, c3s, label="Probability of stable 3 community")
# plt.scatter(m2s, cboths, label="Probability of stable 2 and 3 community")
# plt.xlabel("delta")
# plt.ylabel("Probability")
# plt.title("Bistability test for {} node SBM (constant mean degree)".format(N))
# plt.legend()
# plt.show()
