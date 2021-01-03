Basic Multilayer Example
========================

.. code-block:: python

  from modularitypruning import prune_to_multilayer_stable_partitions
  from modularitypruning.louvain_utilities import repeated_louvain_from_gammas_omegas
  from random import random
  import igraph as ig
  import numpy as np

  num_layers = 2
  n_per_layer = 30
  layer_vec = [i // n_per_layer for i in range(n_per_layer * num_layers)]
  interlayer_edges = [(n_per_layer * layer + v, n_per_layer * layer + v + n_per_layer)
                      for layer in range(num_layers - 1)
                      for v in range(n_per_layer)]

  p_in = 0.5
  p_out = 0.05
  K = 3

  comm_per_layer = [[i // (n_per_layer // K) for i in range(n_per_layer)] for _ in range(num_layers)]
  comm_vec = [item for sublist in comm_per_layer for item in sublist]
  intralayer_edges = [(u, v) for v in range(len(comm_vec)) for u in range(v + 1, len(comm_vec))
                      if layer_vec[v] == layer_vec[u] and (
                              (comm_vec[v] == comm_vec[u] and random() < p_in) or
                              (comm_vec[v] != comm_vec[u] and random() < p_out)
                      )]

  G_intralayer = ig.Graph(intralayer_edges)
  G_interlayer = ig.Graph(interlayer_edges, directed=True)

  gamma_range = (0, 2)
  omega_range = (0, 2)
  louvain_gammas = np.linspace(*gamma_range, 10)
  louvain_omegas = np.linspace(*omega_range, 10)

  parts = repeated_louvain_from_gammas_omegas(G_intralayer, G_interlayer, layer_vec,
                                              gammas=louvain_gammas,
                                              omegas=louvain_omegas)

  stable_parts = prune_to_multilayer_stable_partitions(G_intralayer, G_interlayer, layer_vec, "temporal",
                                                       parts, *gamma_range, *omega_range)
  print(stable_parts)
