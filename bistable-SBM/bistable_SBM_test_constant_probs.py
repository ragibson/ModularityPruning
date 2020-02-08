# Generates figure B.2

from utilities import *
import os


def generate_bistable_SBM_test_output():
    print("Running bistable SBM test...")

    N = 2400
    p2s = np.linspace(0.01, 0.02, 5).tolist() + \
          np.linspace(0.02, 0.025, 15).tolist() + \
          np.linspace(0.025, 0.0475, 5).tolist() + \
          np.linspace(0.0475, 0.0525, 15).tolist() + \
          np.linspace(0.0525, 0.06, 5).tolist()

    DELTA_ITERATIONS = 100
    LOUVAIN_ITERATIONS_PER_DELTA = 1000

    c2s = []
    c3s = []
    cboths = []
    progress = Progress(DELTA_ITERATIONS * len(p2s))

    for p_out2 in p2s:
        total = 0
        count_2stable = 0
        count_3stable = 0
        count_both_stable = 0

        while total < DELTA_ITERATIONS:
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
            gammas = np.linspace(GAMMA_START, GAMMA_END, LOUVAIN_ITERATIONS_PER_DELTA)

            all_parts = repeated_parallel_louvain_from_gammas(G, gammas, show_progress=False)
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

    with open("SBM_constant_probs.out", "w") as f:
        print(N, file=f)
        print(p2s, file=f)
        print(c2s, file=f)
        print(c3s, file=f)
        print(cboths, file=f)


def plot_bistable_SBM_empirical_results():
    output = open("SBM_constant_probs.out", "r").read()

    run = output.split("\n\n")[0]
    lines = run.split("\n")
    N = float(lines[0])
    p2s = eval(lines[1])
    c2s = eval(lines[2])
    c3s = eval(lines[3])
    cboths = eval(lines[4])

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.scatter(p2s, c2s, label="Probability of $K=2$ stability", s=20)
    plt.scatter(p2s, c3s, label="Probability of $K=3$ stability", s=20)
    plt.plot(p2s, cboths, label="Probability of bistability", alpha=0.75, color="C2")
    plt.xlabel(r"$\delta$", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.title(f"Simulation of Example Bistable SBM with $N={N}$", fontsize=14)
    plt.xlim((-0.003, 0.063))
    plt.legend()
    plt.savefig("simulation_bistable_sbm.pdf")


if not os.path.exists("SBM_constant_probs.out"):
    generate_bistable_SBM_test_output()

plot_bistable_SBM_empirical_results()
