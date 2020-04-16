## Performance pass of implementation

On a machine with

 - CPU: i7-9700K (8 cores, stock, up to 4.9 GHz)
 - RAM: 16 GB DDR4 3200 MHz

Some figure generation runtimes are as follows.

| Figures | Runtime | Saves Progress/Results? |
|---------|---------|-----------------|
| [karate_club_gamma_estimates](karate_club_gamma_estimates/karate_club_gamma_estimates.py) | ~50 s | No
| [karate_club_estimates_per_community](karate_club_gamma_estimates/karate_club_estimates_per_community.py) | ~40 s | No
| [plot_champ_example](example_figures/plot_champ_example.py) | ~10 s | No
| [karate_club_test](karate_club_test/karate_club_test.py) | ~5 m | No
| [easy_regime_generation](synthetic_easy_regime/easy_regime_generation.py) | ~60 m | Yes
| [lazega_figures](lazega_law_firm/lazega_figures.py) | ~25 m | Yes
| [plot_maximum_gamma_estimates](plot_duality_details/plot_maximum_gamma_estimates.py) | ~1 s | No
| [plot_maximum_gamma_estimates_in_omega_space](plot_duality_details/plot_maximum_gamma_estimates_in_omega_space.py) | ~1 s | No
| [plot_maximum_gamma_estimates_general](plot_duality_details/plot_maximum_gamma_estimates_general.py) | ~1 s | No
| [SNAP_boxplot](social_networks/SNAP_boxplot.py) | ~55 m | Yes
| [plot_gamma_omega_duality](plot_duality_details/plot_gamma_omega_duality.py) | ~2 s | No
| [plot_bistable_SBM_analytic](bistable_SBM/plot_bistable_SBM_analytic.py) | ~1 s | No
| [bistable_SBM_test_constant_probs](bistable_SBM/bistable_SBM_test_constant_probs.py) | ~2 d | Yes
| [plot_bistable_SBM_realizations](bistable_SBM/plot_bistable_SBM_realizations.py) | ~4 s | No