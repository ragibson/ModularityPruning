import igraph as ig
from utilities import *

ITERATIONS = 1000

for N in [150, 300, 600]:
    for p_out2 in np.linspace(0.01, 0.05, 2):
        progress = Progress(ITERATIONS, name="N={}, delta={:.2f}:".format(N, p_out2))
        theory_g2s = []
        theory_g3s = []
        observed_g2s = []
        observed_g3s = []
        total = 0
        count_2stable = 0
        count_3stable = 0
        count_both_stable = 0

        while total < ITERATIONS:
            p_in1 = 10 / 99  # m_in = 10/3
            p_in2 = p_in1 * 0.75  # m_in = 5/2 for each block
            p_out1 = 0.25 / 40  # m_out = 1.25
            # p_out2 in outer loop

            pref_matrix = [[p_in1, p_out1, p_out1],
                           [p_out1, p_in2, p_out2],
                           [p_out1, p_out2, p_in2]]
            block_sizes = [N // 3] * 3
            G = ig.Graph.SBM(N, pref_matrix, block_sizes)

            if not G.is_connected():
                # print("\rDisconnected graph. Skipping...", end='', flush=True)
                continue

            # print("mean degree is", np.mean([G.degree(v) for v in range(N)]))
            ground_truth = tuple(i // block_sizes[0] for i in range(N))
            ground_truth2 = tuple(min(1, i // block_sizes[0]) for i in range(N))
            true_gamma = gamma_estimate(G, ground_truth)
            true_gamma2 = gamma_estimate(G, ground_truth2)

            if true_gamma is None or true_gamma2 is None:
                # print("\rDegenerate ground truth estimate. Skipping...", end='', flush=True)
                continue

            GAMMA_START = 0.0
            GAMMA_END = 2.0

            all_parts = repeated_parallel_louvain(G, GAMMA_START, GAMMA_END, gamma_iters=1000 // cpu_count() + 1,
                                                  repeat=cpu_count(), show_progress=False)
            ranges = CHAMP_2D(G, all_parts, GAMMA_START, GAMMA_END)
            gamma_estimates = ranges_to_gamma_estimates(G, ranges)

            for g_start, g_end, membership, g_est in gamma_estimates:
                if g_est is not None and num_communities(membership) == 2:
                    observed_g2s.append(g_est)
                if g_est is not None and num_communities(membership) == 3:
                    observed_g3s.append(g_est)

            theory_g2s.append(true_gamma2)
            theory_g3s.append(true_gamma)
            total += 1
            progress.increment()

        progress.done()
        print(observed_g2s)
        print(theory_g2s)
        print(observed_g3s)
        print(theory_g3s)
        # low = min(min(observed_g2s), min(theory_g2s))
        # high = max(max(observed_g3s), max(theory_g3s))
        # shared_bins = np.linspace(low, high, 50)
        # plt.hist(observed_g2s, bins=shared_bins, label="observed 2-community gamma estimates", alpha=0.5,
        #          weights=np.ones_like(observed_g2s) / float(len(observed_g2s)))
        # plt.hist(theory_g2s, bins=shared_bins, label="ground truth 2-community gamma estimates", alpha=0.5,
        #          weights=np.ones_like(theory_g2s) / float(len(theory_g2s)))
        # plt.hist(observed_g3s, bins=shared_bins, label="observed 3-community gamma estimates", alpha=0.5,
        #          weights=np.ones_like(observed_g3s) / float(len(observed_g3s)))
        # plt.hist(theory_g3s, bins=shared_bins, label="ground truth 3-community gamma estimates", alpha=0.5,
        #          weights=np.ones_like(theory_g3s) / float(len(theory_g3s)))
        # plt.legend()
        # plt.ylabel("frequency")
        # plt.xlabel("gamma")
        # plt.title("\"Are you sure?\" -- gamma range, N={}, delta={:.2f}".format(N, p_out2))
        # plt.show()
