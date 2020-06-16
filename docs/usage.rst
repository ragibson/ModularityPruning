Usage
=====

Note that this documentation only includes the functions intended for single-layer network analysis at the moment. We
expect that this will be the most common use case for our package, but the package also supports multi-layer networks.

.. _modularitypruning:

modularitypruning
-----------------

.. function:: prune_to_stable_partitions(G, parts, gamma_start, gamma_end, restrict_num_communities=None, single_threaded=False)

    This runs our full pruning pipeline on a singlelayer network. Returns the pruned list of stable partitions.

    See **[CITATION FORTHCOMING]** for more details.

    :param G: graph of interest
    :type G: igraph.Graph
    :param parts: partitions to prune
    :type parts: list[tuple]
    :param gamma_start: starting gamma value for CHAMP
    :type gamma_start: float
    :param gamma_end: ending gamma value for CHAMP
    :type gamma_end: float
    :param restrict_num_communities: if not None, only use input partitions of this many communities
    :type restrict_num_communities: int or None
    :param single_threaded: if True, run the CHAMP step in serial
    :type single_threaded: bool
    :return: list of community membership tuples (tuple[int])

modularitypruning.champ_utilities
---------------------------------

These functions provide access to the `CHAMP <https://doi.org/10.3390/a10030093>`_ method of Weir et al.

.. function:: CHAMP_2D(G, all_parts, gamma_0, gamma_f, single_threaded=False)

    Calculates the pruned set of partitions from CHAMP on ``gamma_0`` :math:`\leq \gamma \leq` ``gamma_f``.

    See https://doi.org/10.3390/a10030093 for more details.

    :param G: graph of interest
    :type G: igraph.Graph
    :param all_parts: partitions to prune
    :type all_parts: list[tuple] or set[tuple]
    :param gamma_0: starting gamma value for CHAMP
    :type gamma_0: float
    :param gamma_f: ending gamma value for CHAMP
    :type gamma_f: float
    :param single_threaded: if True, run in serial. Otherwise, use all CPU cores to run in parallel
    :type single_threaded: bool
    :return: list of tuples for the somewhere optimal partitions, containing (in-order)

        - starting gamma value for the partition's domain of optimality
        - ending gamma value for the partition's domain of optimality
        - community membership tuple (tuple[int]) for the partition

modularitypruning.louvain_utilities
-----------------------------------

These functions provide access to the `Louvain <https://doi.org/10.1088%2F1742-5468%2F2008%2F10%2FP10008>`_ modularity
maximization algorithm and some related utilities. The implementation here is provided by `louvain-igraph
<https://github.com/vtraag/louvain-igraph>`_.

.. function:: sorted_tuple(t)

    Returns a canonical representation of a membership tuple in which the labels' first occurrences are sorted (e.g.
    label 0 will always occur before label 1 in the tuple).

    :param t: community membership of a partition
    :type t: tuple[int]
    :rtype: tuple[int]

.. function:: singlelayer_louvain(G, gamma, return_partition=False)

    Run the Louvain modularity maximization algorithm at a single gamma value.

    :param G: graph of interest
    :type G: igraph.Graph
    :param gamma: gamma (resolution parameter) to run Louvain at
    :type gamma: float
    :param return_partition: if True, return a louvain partition. Otherwise, return a community membership tuple
    :type return_partition: bool
    :rtype: tuple[int] or louvain.RBConfigurationVertexPartition

.. function:: repeated_parallel_louvain_from_gammas(G, gammas, show_progress=True, chunk_dispatch=True)

    Runs the Louvain modularity maximization algorithm at each provided gamma value, using all CPU cores.

    :param G: graph of interest
    :type G: igraph.Graph
    :param gammas: list of gammas (resolution parameters) to run Louvain at
    :type gammas: list[float]
    :param show_progress: if True, render a progress bar. This will only work if ``chunk_dispatch`` is also True
    :type show_progress: bool
    :param chunk_dispatch: if True, dispatch parallel work in chunks. Setting this to False may increase performance,
                           but can lead to out-of-memory issues
    :type chunk_dispatch: bool
    :return: a set of all unique partitions (tuple[int]) returned by the Louvain algorithm

modularitypruning.parameter_estimation
--------------------------------------

These functions provide the ability to iteratively estimate "correct" values for the resolution parameter in modularity
as discussed by `Newman <https://doi.org/10.1103/PhysRevE.94.052315>`_ and `Pamfil et al.
<https://doi.org/10.1137/18M1231304>`_ Here, we maximize modularity via the Louvain algorithm.

.. function:: iterative_monolayer_resolution_parameter_estimation(G, gamma=1.0, tol=1e-2, max_iter=25, verbose=False)

    Monolayer variant of ALG. 1 from "Relating modularity maximization and stochastic block models in multilayer
    networks." This is intended to determine an "optimal" value for gamma by repeatedly maximizing modularity and
    estimating new values for the resolution parameter.

    See https://doi.org/10.1137/18M1231304 for more details.

    :param G: graph of interest
    :type G: igraph.Graph
    :param gamma: initialization gamma value
    :type gamma: float
    :param tol: convergence tolerance
    :type tol: float
    :param max_iter: maximum number of iterations
    :type max_iter: int
    :param verbose: whether or not to print verbose output
    :type verbose: bool
    :return: tuple containing

        - gamma (float) to which the iteration converged
        - the resulting partition (`louvain.RBConfigurationVertexPartition`)

modularitypruning.parameter_estimation_utilities
------------------------------------------------

These functions provide utilities related to the parameter estimation of `Newman
<https://doi.org/10.1103/PhysRevE.94.052315>`_ and `Pamfil et al. <https://doi.org/10.1137/18M1231304>`_

.. function:: estimate_singlelayer_SBM_parameters(G, partition)

    Estimate singlelayer SBM parameters from a graph and a partition.

    See https://doi.org/10.1103/PhysRevE.94.052315 for more details.

    :param G: graph of interest
    :type G: igraph.Graph
    :param partition: partition of interest
    :type partition: louvain.RBConfigurationVertexPartition
    :return: tuple(float, float) of SBM parameter estimates :math:`(\omega_{in}, \omega_{out})`

.. function:: gamma_estimate(G, partition)

    Compute the "correct" value of gamma where modularity maximization becomes equivalent to maximum likelihood methods
    on a degree-corrected, planted partition stochastic block model.

    See https://doi.org/10.1103/PhysRevE.94.052315 for more details.

    :param G: graph of interest
    :type G: igraph.Graph
    :param partition: partition of interest
    :type partition: tuple[int] or louvain.RBConfigurationVertexPartition
    :rtype: float

.. function:: gamma_estimate_from_parameters(omega_in, omega_out)

    Compute the "correct" value of gamma (as in :meth:`gamma_estimate`) from SBM parameters.

    :param omega_in: within-community edge propensity of a degree-corrected, planted partition SBM
    :type omega_in: float
    :param omega_out: within-community edge propensity of a degree-corrected, planted partition SBM
    :type omega_out: float
    :rtype: float

.. function:: ranges_to_gamma_estimates(G, ranges)

    Compute gamma estimates (as in :meth:`gamma_estimate`), given domains of optimality from :meth:`CHAMP_2D`.

    :param G: graph of interest
    :type G: igraph.Graph
    :param ranges: list of ``(gamma_start, gamma_end, membership)`` tuples (as returned from :meth:`CHAMP_2D`)
    :type ranges: list[tuple]
    :return: a copy of ``ranges`` with the corresponding gamma estimate (float) appended to each tuple


.. function:: gamma_estimates_to_stable_partitions(gamma_estimates)

    Computes the stable partitions (i.e. those whose gamma estimates are within their domains of optimality), given
    domains of optimality and gamma estimates from :meth:`ranges_to_gamma_estimates`.

    See **[CITATION FORTHCOMING]** for more details.

    :param gamma_estimates: list of ``(gamma_start, gamma_end, membership, gamma_estimate)`` tuples (as returned from
                            :meth:`ranges_to_gamma_estimates`)
    :type gamma_estimates: list[tuple]
    :return: list of community membership tuples (tuple[int]) of the stable partitions

.. function:: prune_to_stable_partitions(G, parts, gamma_start, gamma_end, restrict_num_communities=None, single_threaded=False)
    :noindex:

    See description in :ref:`modularitypruning`.

modularitypruning.plotting
--------------------------

Documentation and examples forthcoming.