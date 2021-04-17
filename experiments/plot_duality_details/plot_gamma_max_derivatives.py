# plot large-K gamma_max behavior and derivatives
import functools
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from math import log
from modularitypruning.progress import Progress


def f(K, x):
    if not 0 < x < 1:
        return 0

    return K * (1 / x - 1) / (1 / x + K - 1) / log(1 / x)


@functools.lru_cache(maxsize=None)
def gamma_max(K, eps=1e-16):
    return -minimize_scalar(lambda x: -f(K=K, x=x), [eps, 1 - eps]).fun


def fprime_estimate(K):
    return (gamma_max(K + 1) - gamma_max(K - 1)) / 2


def fprime2_estimate(K):
    return (gamma_max(K + 1) - 2 * gamma_max(K) + gamma_max(K - 1)) / 1


def fprime3_estimate(K):
    return (gamma_max(K + 2) - 2 * gamma_max(K + 1)
            + 2 * gamma_max(K - 1) - gamma_max(K - 2)) / 2


def plot_gamma_max(Ks):
    progress = Progress(len(Ks))
    maxes = []
    for K in Ks:
        maxes.append(gamma_max(K))
        progress.increment()
    progress.done()

    plt.close()
    plt.figure()
    plt.plot(Ks, maxes)
    plt.xlabel('$K$', fontsize=14)
    plt.ylabel(r'$\gamma_{max}$', fontsize=14)
    plt.title(r"$\gamma_{max}$ vs. $K$")
    plt.gca().ticklabel_format(style='plain')
    plt.tight_layout()
    plt.savefig("gamma_max_plot_to_K=1M.pdf")


def plot_gamma_max_derivatives(Ks):
    plt.close()
    fig, ax = plt.subplots(nrows=3, figsize=(6, 12))
    ax[0].plot(Ks[1:], [fprime_estimate(K) for K in Ks[1:]])
    ax[0].set_title(r"$\gamma_{max}$ $f'$ approximation: $\frac{f(K+1)-f(K-1)}{2}$", fontsize=14)
    ax[0].set_xscale('log')
    ax[1].plot(Ks[1:], [fprime2_estimate(K) for K in Ks[1:]])
    ax[1].plot(Ks[1:], [0] * len(Ks[1:]), linestyle="dashed", color="black", alpha=0.75)
    ax[1].set_title(r"$\gamma_{max}$ $f''$ approximation: $\frac{f(K+1)-2f(K)+f(K-1)}{1}$", fontsize=14)
    ax[1].set_xscale('log')
    ax[2].plot(Ks[2:], [fprime2_estimate(K) for K in Ks[2:]])
    ax[2].plot(Ks[2:], [0] * len(Ks[2:]), linestyle="dashed", color="black", alpha=0.75)
    ax[2].set_title(r"$\gamma_{max}$ $f'''$ approximation: $\frac{f(K+2)-2f(K+1)+2f(K-1)-f(K-2)}{2}$", fontsize=14)
    ax[2].set_xscale('log')

    for axis in ax:
        axis.set_xlabel('$K$', fontsize=14)
        axis.set_ylabel('derivative approximation', fontsize=14)
    plt.tight_layout()
    plt.savefig("gamma_max_finite_differences.pdf")


def plot_gamma_max_tangent_examples():
    # tangent at K=100, plot to K=250
    Ks = list(range(2, 251))
    tangent_location = 100
    tangent_slope = fprime_estimate(K=tangent_location)
    tangent_intercept = gamma_max(K=tangent_location) - tangent_slope * tangent_location

    plt.close()
    plt.plot(Ks, [gamma_max(K) for K in Ks], label=r"$\gamma_{max}$")
    plt.plot(Ks, [tangent_intercept + tangent_slope * K for K in Ks], linestyle="dashed",
             label=f"${tangent_intercept:.4f}+{tangent_slope:.4f}K$")
    plt.xlabel('$K$', fontsize=14)
    plt.ylabel(r'$\gamma_{max}$', fontsize=14)
    plt.title(rf"$\gamma_{{max}}$ vs. $K$, example tangent at $K={tangent_location}$")
    plt.gca().ticklabel_format(style='plain')
    plt.legend()
    plt.tight_layout()
    plt.savefig("gamma_max_tangent_example1.pdf")

    # tangent at K=1000, plot to K=5000
    Ks = list(range(2, 5001))
    tangent_location = 1000
    tangent_slope = fprime_estimate(K=tangent_location)
    tangent_intercept = gamma_max(K=tangent_location) - tangent_slope * tangent_location

    plt.close()
    plt.plot(Ks, [gamma_max(K) for K in Ks], label=r"$\gamma$ max")
    plt.plot(Ks, [tangent_intercept + tangent_slope * K for K in Ks], linestyle="dashed",
             label=f"${tangent_intercept:.4f}+{tangent_slope:.4f}K$")
    plt.xlabel('$K$', fontsize=14)
    plt.ylabel(r'$\gamma_{max}$', fontsize=14)
    plt.title(rf"$\gamma_{{max}}$ vs. $K$, example tangent at $K={tangent_location}$")
    plt.gca().ticklabel_format(style='plain')
    plt.legend()
    plt.tight_layout()
    plt.savefig("gamma_max_tangent_example2.pdf")


if __name__ == "__main__":
    # show first two gamma max plots to K=1M
    Ks = list(range(2, 10 ** 6 + 1))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_gamma_max(Ks)
    plot_gamma_max_derivatives(Ks)
    plot_gamma_max_tangent_examples()
