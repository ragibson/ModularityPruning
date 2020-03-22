# ModularityPruning

- [X] Functionality checking of figure generation scripts
- [X] Cleanup pass of implementation
- [X] Cleanup pass of figure generation scripts
- [X] Initial performance pass of implementation

## Initial performance pass of implementation

On a machine with

 - CPU: i7-9700K (stock, up to 4.9 GHz)
 - RAM: 16 GB DDR4 3200 MHz

the figure generation runtimes are as follows.

| Figures | Runtime | Saves Progress/Results? |
|---------|---------|-----------------|
| [2.1](karate_club_gamma_estimates/karate_club_gamma_estimates.py) | ~50 s | No
| [2.2](karate_club_gamma_estimates/karate_club_estimates_per_community.py) | ~40 s | No
| [3.1](example_figures/plot_champ_example.py) | ~10 s | No
| [5.1 - 5.2](karate_club_test/karate_club_test.py) | ~5 m | No
| [5.3 - 5.6](synthetic_easy_regime/easy_regime_generation.py) | ~60 m | Yes
| [5.7 - 5.10](lazega_law_firm/lazega_figures.py) | ~25 m | Yes
| [6.1](plot_duality_details/plot_maximum_gamma_estimates.py) | ~1 s | No
| [6.2](plot_duality_details/plot_maximum_gamma_estimates_in_omega_space.py) | ~1 s | No
| [6.3](plot_duality_details/plot_maximum_gamma_estimates_general.py) | ~1 s | No
| [6.4](social_networks/SNAP_boxplot.py) | ~55 m | Yes
| [A.1 - A.2](plot_duality_details/plot_gamma_omega_duality.py) | ~2 s | No
| [B.1](bistable_SBM/plot_bistable_SBM_analytic.py) | ~1 s | No
| [B.2](bistable_SBM/bistable_SBM_test_constant_probs.py) | ~2 d | Yes
| [B.3](bistable_SBM/plot_bistable_SBM_realizations.py) | ~4 s | No
