.. ModularityPruning documentation master file, created by
   sphinx-quickstart on Tue Jun 16 12:28:29 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ModularityPruning's documentation!
=============================================

ModularityPruning is a pruning tool to identify small subsets of network partitions that are significant from the
perspective of stochastic block model inference.

This method works for single-layer and multi-layer networks, as well as for restricting focus to a fixed number of
communities when desired.

Prior to version v1.3.0, we used the Louvain algorithm for modularity maximization instead of Leiden. The deprecated
module ``modularitypruning.louvain_utilities`` now shims single-layer functions to their corresponding Leiden versions
in ``modularitypruning.leiden_utilities`` (though it still contains the legacy multi-layer functions since they can be
faster in general -- leidenalg does not efficiently implement multilayer optimization).

.. toctree::
   source/installation
   source/basic_example
   source/basic_multilayer_example
   source/usage
   source/multilayer_usage
   source/plotting_examples
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
