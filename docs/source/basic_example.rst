Basic Singlelayer Example
=========================

The simplest entry point to the package is the ``prune_to_stable_partitions`` function.

See :doc:`usage` for complete documentation on this function, but a simple example of its usage is

.. code-block:: python

   import igraph as ig
   from modularitypruning import prune_to_stable_partitions
   from modularitypruning.leiden_utilities import repeated_leiden_from_gammas
   import numpy as np

   # get Karate Club graph in igraph
   G = ig.Graph.Famous("Zachary")

   # run leiden 1000 times on this graph from gamma=0 to gamma=2
   partitions = repeated_leiden_from_gammas(G, np.linspace(0, 2, 1000))

   # prune to the stable partitions from gamma=0 to gamma=2
   stable_partitions = prune_to_stable_partitions(G, partitions, 0, 2)
   print(stable_partitions)

This prints

.. code-block:: python

   [(0, 0, 0, 0, 1, 1, 1, 0, 2, 2, 1, 0, 0, 0, 2, 2, 1, 0, 2, 0, 2, 0, 2, 3, 3, 3, 2, 3, 3, 2, 2, 3, 2, 2)]

which is the stable 4-community split of the Karate Club network.

It is often useful to set ``restrict_num_communities`` in order to restrict focus to a specific number of communities.
For the Karate Club, ``restrict_num_communities=2`` yields the more standard, 2-community split.