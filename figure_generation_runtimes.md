## Figure Generation Runtimes

On a machine with

 - CPU: i7-9700K (8 cores, stock, up to 4.9 GHz)
 - RAM: 16 GB DDR4 3200 MHz

the figure generation runtimes that exceed ~5 seconds are as follows.

| Figures | Runtime | Saves Progress/Results? |
|---------|---------|-----------------|
| [karate_club_gamma_estimates](experiments/karate_club_gamma_estimates/karate_club_gamma_estimates.py) | ~50 s | No
| [karate_club_estimates_per_community](experiments/karate_club_gamma_estimates/karate_club_estimates_per_community.py) | ~40 s | No
| [plot_champ_example](experiments/example_figures/plot_champ_example.py) | ~10 s | No
| [modularity_nonmonotonic_K](experiments/modularity_nonmonotonic_K/modularity_nonmonotonic_K.py) | ~1 m | No
| [karate_club_test](experiments/karate_club_test/karate_club_test.py) | ~5 m | No
| [plot_gamma_max_derivatives](experiments/plot_duality_details/plot_gamma_max_derivatives.py) | ~5 m | No
| [easy_regime_generation](experiments/pamfil_synthetic_networks/easy_regime_generation.py) | ~55 m | Yes
| [hard_regime_generation](experiments/pamfil_synthetic_networks/hard_regime_generation.py) | ~60 m | Yes
| [lazega_figures](experiments/lazega_law_firm/lazega_figures.py) | ~25 m | Yes
| [SNAP_boxplot](experiments/social_networks/SNAP_boxplot.py) | ~55 m | Yes
| [bistable_SBM_test_constant_probs](experiments/bistable_SBM/bistable_SBM_test_constant_probs.py) | ~2 d | Yes
| [lfr_benchmark_test](experiments/lfr_benchmark_test/lfr_benchmark_test.py) | ~3.0 h | Yes
| [hierarchical_SBM_test](experiments/hierarchical_SBM_test/hierarchical_SBM_test.py) | ~1.5 h | Yes
| [runtime_comparison_with_louvain](experiments/miscellaneous_tests/runetime_comparison_with_louvain.py) | ~25 m | Yes