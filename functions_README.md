# Functions

This README contains a list of some of the useful functions from the package, broken down by subpackage. For now, this
list only contains methods intended for the analysis of single-layer networks.

# Table of Contents
  * [modularitypruning](#modularitypruning)
  * [modularitypruning.champ_utilities](#champ_utilities)
  * [modularitypruning.louvain_utilities](#louvain_utilities)
  * [modularitypruning.parameter_estimation_utilities](#parameter_estimation_utilities)
  * [modularitypruning.plotting](#plotting)

<a name = "modularitypruning"></a>
## modularitypruning

    prune_to_stable_partitions(G, parts, gamma_start, gamma_end, restrict_num_communities=None, single_threaded=False)

This runs our full pruning pipeline on a singlelayer network.

Parameters
 * `G`: graph of interest
 * `parts`: partitions to prune
 * `gamma_start`: starting gamma value for CHAMP
 * `gamma_end`: ending gamma value for CHAMP
 * `restrict_num_communities`: if not None, only use input partitions of this many communities
 * `single_threaded`: if True, run the CHAMP step without parallelization

Returns the pruned set of stable partitions.

<a name = "champ_utilities"></a>
## modularitypruning.champ_utilities

    CHAMP_2D(G, all_parts, gamma_0, gamma_f, single_threaded=False)

This calculates the pruned set of partitions from CHAMP on gamma_0 <= gamma <= gamma_f.

Parameters
 * `G`: graph of interest
 * `all_parts`: partitions to prune
 * `gamma_0`: starting gamma value
 * `gamma_f`: ending gamma value
 * `single_threaded`: if True, run without parallelization

Returns a list `(domain_gamma_start, domain_gamma_end, membership)` tuples.

<a name = "louvain_utilities"></a>
## modularitypruning.louvain_utilities

    sorted_tuple(t)

This returns a canonical representation of a membership tuple in which the labels' first occurrences are sorted (e.g.
label 1 will always occur before label 2 as the node numbers increase).

    repeated_parallel_louvain_from_gammas(G, gammas, show_progress=True, chunk_dispatch=True)

This runs the louvain modularity maximization heuristic at each gamma in gammas, using all CPU cores available.

Parameters
 * `G`: input graph
 * `gammas`: list of gammas (resolution parameters) to run louvain at
 * `show_progress`: if True, render a progress bar
 * `chunk_dispatch`: if True, dispatch parallel work in chunks. Setting this to False may increase performance, but can
 lead to out-of-memory issues

Returns a set of all unique partitions encountered.

<a name = "parameter_estimation_utilities"></a>
## modularitypruning.parameter_estimation_utilities

    gamma_estimate(G, partition)

Parameters
 * `G`: input graph
 * `partition`: membership vector of the partition of interest

Returns the gamma estimate for a graph and a partition.

    gamma_estimate_from_parameters(omega_in, omega_out)

Parameters
 * `omega_in`: within community edge propensity
 * `omega_out`: between community edge propensity

Returns the gamma estimate derived from these SBM parameters.

    ranges_to_gamma_estimates(G, ranges)

This adds gamma estimates onto a list of domains of optimality.

Parameters
 * `G`: input graph
 * `ranges`: list of `(gamma_start, gamma_end, membership)` tuples

Returns a list of `(gamma_start, gamma_end, membership, gamma_estimate)` tuples.

    gamma_estimates_to_stable_partitions(gamma_estimates)

Parameters
 * `gamma_estimates`: list of `(gamma_start, gamma_end, membership, gamma_estimate)` tuples

Returns a list of stable partitions, i.e. the membership vectors of the partitions where `gamma_start <= gamma_estimate
<= gamma_end`.

    prune_to_stable_partitions(G, parts, gamma_start, gamma_end, restrict_num_communities=None, single_threaded=False)

See description in [modularitypruning](#modularitypruning).

<a name = "plotting"></a>
## modularitypruning.plotting

Documentation and examples forthcoming.
