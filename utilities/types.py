"""Type aliases for mypy to make the code more readable."""

from collections.abc import Sequence
from typing import Callable, TypeAlias

import leidenalg
import numpy as np


# numpy arrays
FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

# mypy doesn't consider numpy arrays to be a sequence
IntSequence = Sequence[int] | IntArray
FloatSequence = Sequence[float] | FloatArray

# community membership vector and layer membership vector
# note: some parts of the code do actually support list[int] for membership, but this is simpler and more consistent
Membership = tuple[int, ...]
LayerVector = list[int]
LayerTuple = tuple[int, ...]  # needed for hashing

# collection of partitions for CHAMP
Partitions = Sequence[Membership] | set[Membership]

# Leiden partitions
LeidenPartition: TypeAlias = leidenalg.RBConfigurationVertexPartition
MultilayerLeidenParts: TypeAlias = tuple[list[LeidenPartition], leidenalg.CPMVertexPartition]
SingleLayerLeidenResult: TypeAlias = LeidenPartition | Membership
MultilayerLeidenResult: TypeAlias = list[LeidenPartition] | Membership

# collections of partitions for pruning (membership vectors or leiden partitions)
PrunablePartitions = Sequence[LeidenPartition | Membership] | set[LeidenPartition | Membership]

# CHAMP 2D domains and parameter estimates
GammaEstimateValue = np.float64 | None
GammaEstimate = tuple[np.float64, np.float64, Membership, GammaEstimateValue]
GammaRange = tuple[np.float64, np.float64, Membership]

# CHAMP 3D domains and parameter estimates
DomainVertices = list[FloatArray]
Domain = tuple[DomainVertices, Membership]  # plus membership
GammaOmegaEstimate = tuple[GammaEstimateValue, float]
DomainWithEstimates = tuple[DomainVertices, Membership, GammaEstimateValue, float]

# (A_hats, P_hats) and (A_hats, P_hats, C_hats) coefficients from CHAMP
PartitionCoefficients2D = tuple[FloatArray, FloatArray]
PartitionCoefficients3D = tuple[FloatArray, FloatArray, FloatArray]

# SBM parameter estimates
SinglelayerSBMParameters = tuple[float, float]
MultilayerSBMParameters = tuple[float, float, float, int]

# parameter estimation functions
OmegaUpdateFunc = Callable[[float, float, float, int], float]
PersistenceFunc = Callable[[list[int]], float]
GammaUpdateFunc = Callable[[float, float], GammaEstimateValue]

# more flexible plotting equivalents (allows tuples, etc.)
FlexibleDomainVertices = Sequence[FloatArray] | Sequence[tuple[float, ...]]
FlexibleDomain = tuple[FlexibleDomainVertices, Membership]
FlexibleDomainWithEstimates = tuple[FlexibleDomainVertices, Membership, GammaEstimateValue, float]
PlotColor = tuple[float, ...]
AxisLimits = tuple[float, float]
DomainPoint = tuple[float, ...]
