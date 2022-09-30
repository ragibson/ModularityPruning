Basic Multilayer Example
========================

Once the graph is set up, the multilayer usage of modularitypruning is very similar to that of singlelayer networks.

First, we'll need to construct a multilayer network. Here, we create a simple, 3-community multilayer SBM that merges
all communities into one in the final layer.

.. code-block:: python

  from random import random
  import igraph as ig

  num_layers = 3
  n_per_layer = 30
  p_in = 0.5
  p_out = 0.05
  K = 3

  # layer_vec holds the layer membership of each node
  # e.g. layer_vec[5] = 2 means that node 5 resides in layer 2 (the third layer)
  layer_vec = [i // n_per_layer for i in range(n_per_layer * num_layers)]
  interlayer_edges = [(n_per_layer * layer + v, n_per_layer * layer + v + n_per_layer)
                      for layer in range(num_layers - 1)
                      for v in range(n_per_layer)]

  # set up a community vector with
  #   three communities in layer 0 (each of size 10)
  #   three communities in layer 1 (each of size 10)
  #   one community in layer 2 (of size 30)
  comm_per_layer = [[i // (n_per_layer // K) if layer < num_layers - 1 else 0
                     for i in range(n_per_layer)] for layer in range(num_layers)]
  comm_vec = [item for sublist in comm_per_layer for item in sublist]

  # randomly connect nodes inside each layer with undirected edges according to
  # within-community probability p_in and between-community probability p_out
  intralayer_edges = [(u, v) for v in range(len(comm_vec)) for u in range(v + 1, len(comm_vec))
                      if layer_vec[v] == layer_vec[u] and (
                              (comm_vec[v] == comm_vec[u] and random() < p_in) or
                              (comm_vec[v] != comm_vec[u] and random() < p_out)
                      )]

  # create the networks in igraph. By Pamfil et al.'s convention, the interlayer edges
  # of a temporal network are directed (representing the "one-way" nature of time)
  G_intralayer = ig.Graph(intralayer_edges)
  G_interlayer = ig.Graph(interlayer_edges, directed=True)

Now that we have a multilayer network, we can run leidenalg across a grid of gamma and omega values and prune the
resulting partitions down to a small subset of stable partitions.

.. code-block:: python

  from modularitypruning import prune_to_multilayer_stable_partitions
  from modularitypruning.leiden_utilities import repeated_leiden_from_gammas_omegas
  import numpy as np

  # run leidenalg on a uniform 32x32 grid (1024 samples) of gamma and omega in [0, 2]
  gamma_range = (0, 2)
  omega_range = (0, 2)
  leiden_gammas = np.linspace(*gamma_range, 32)
  leiden_omegas = np.linspace(*omega_range, 32)

  parts = repeated_leiden_from_gammas_omegas(G_intralayer, G_interlayer, layer_vec,
                                             gammas=leiden_gammas,
                                             omegas=leiden_omegas)

  # prune to the stable partitions from (gamma=0, omega=0) to (gamma=2, omega=2)
  stable_parts = prune_to_multilayer_stable_partitions(G_intralayer, G_interlayer, layer_vec,
                                                       "temporal", parts,
                                                       *gamma_range, *omega_range)

  for p in stable_parts:
      # instead of print(p), we use a more condensed format for the membership vector here
      print(" ".join(str(x) for x in p))


In an example run, we find the following two stable partitions:

.. code-block:: python

  0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 0 0 0 1 1 1 2 0 0 2 1 1 0 2 2 1 0 2 1 1 0 0 2 0 2 1 0 2 0 2

Here, we can see that modularitypruning essentially identifies the ground truth membership vector (the first partition
output) and a second partition that follows the ground truth for the first two layers, but splits the third into
roughly equal size communities.

This is perhaps reasonable since the final layer has no significant community structure whatsoever (it is comprised of
a single community).

As in the singlelayer case, it is often useful to set ``restrict_num_communities`` in order to restrict focus to a
specific number of communities. However, the community structure is so strong in this example that we find similar
results without this restriction.