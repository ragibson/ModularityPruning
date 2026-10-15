"""
This is a deprecated version of leiden_utilities. Usage will be shimmed to use Leiden instead of Louvain when possible.

Prior to version v1.3.0, we used the Louvain algorithm for modularity maximization instead of Leiden. The deprecated
module ``modularitypruning.louvain_utilities`` now shims Louvain function names to their corresponding Leiden versions
in ``modularitypruning.leiden_utilities``.
"""
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    import igraph
    import leidenalg
import warnings

from . import leiden_utilities

warnings.simplefilter('always', DeprecationWarning)
warnings.warn("modularitypruning.louvain_utilities has been replaced. "
              "Please use leiden_utilities in the future when possible.", DeprecationWarning)


# TODO: simplify this type, it's not meaningful to do this for such a shim
def __getattr__(
      name: str,
) -> (
    "Callable[..., set[tuple[int, ...]]]|"
    "Callable[[igraph.Graph, list[float]], set[tuple[int, ...]]]|"
    "Callable[[igraph.Graph, list[int]|tuple[int, ...]], leidenalg.RBConfigurationVertexPartition]|"
    "Callable[[igraph.Graph], leidenalg.RBConfigurationVertexPartition]|None"
):
    if "louvain" in name:
        leiden_name: str = name.replace("louvain", "leiden")
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(f"The Louvain functions have been deprecated. Replacing {repr(name)} with {repr(leiden_name)}.",
                      DeprecationWarning)
        return getattr(leiden_utilities, leiden_name)
