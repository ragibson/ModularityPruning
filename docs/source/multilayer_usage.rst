Multi-Layer Usage
=================

Note that this documentation page only includes the functions intended for multi-layer network analysis.

For single-layer usage, see :doc:`usage`.

.. _modularitypruning:
  :noindex:

modularitypruning
-----------------

.. currentmodule:: modularitypruning
.. autofunction:: prune_to_multilayer_stable_partitions

modularitypruning.champ_utilities
---------------------------------

These functions provide access to the `CHAMP <https://doi.org/10.3390/a10030093>`_ method of Weir et al.

.. currentmodule:: modularitypruning.champ_utilities
.. autofunction:: CHAMP_3D

modularitypruning.leiden_utilities
-----------------------------------

These functions provide access to the `Leiden <https://doi.org/10.1038/s41598-019-41695-z>`_ modularity
maximization algorithm and some related utilities. The implementation here is provided by `leidenalg
<https://github.com/vtraag/leidenalg>`_.

.. currentmodule:: modularitypruning.leiden_utilities
.. autofunction:: sorted_tuple
  :noindex:
.. autofunction:: multilayer_leiden
.. autofunction:: repeated_parallel_leiden_from_gammas_omegas

modularitypruning.parameter_estimation
--------------------------------------

These functions provide the ability to iteratively estimate "correct" values for the resolution parameter in modularity
as discussed by `Newman <https://doi.org/10.1103/PhysRevE.94.052315>`_ and `Pamfil et al.
<https://doi.org/10.1137/18M1231304>`_ Here, we maximize modularity via the Leiden algorithm.

.. currentmodule:: modularitypruning.parameter_estimation
.. autofunction:: iterative_multilayer_resolution_parameter_estimation

modularitypruning.parameter_estimation_utilities
------------------------------------------------

These functions provide utilities related to the parameter estimation of `Newman
<https://doi.org/10.1103/PhysRevE.94.052315>`_ and `Pamfil et al. <https://doi.org/10.1137/18M1231304>`_

.. currentmodule:: modularitypruning.parameter_estimation_utilities
.. autofunction:: estimate_multilayer_SBM_parameters
.. autofunction:: gamma_omega_estimate
.. autofunction:: omega_function_from_model
.. autofunction:: temporal_multilevel_omega_estimate_from_parameters
.. autofunction:: multiplex_omega_estimate_from_parameters
.. autofunction:: domains_to_gamma_omega_estimates
.. autofunction:: gamma_omega_estimates_to_stable_partitions
.. function:: prune_to_multilayer_stable_partitions(G_intralayer, G_interlayer, layer_vec, model, parts, gamma_start, gamma_end, omega_start, omega_end, restrict_num_communities=None, single_threaded=False)
    :noindex:

    See description in :ref:`modularitypruning`.

modularitypruning.plotting
--------------------------

See :doc:`plotting_examples`.