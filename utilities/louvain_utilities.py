"""
This is a deprecated version of leiden_utilities. Usage will be shimmed to use Leiden instead of Louvain when possible.

Prior to version v1.3.0, we used the Louvain algorithm for modularity maximization instead of Leiden. The deprecated
module ``modularitypruning.louvain_utilities`` now shims Louvain function names to their corresponding Leiden versions
in ``modularitypruning.leiden_utilities``.
"""
from typing import Any
import warnings

from . import leiden_utilities

warnings.simplefilter('always', DeprecationWarning)
warnings.warn("modularitypruning.louvain_utilities has been replaced. "
              "Please use leiden_utilities in the future when possible.", DeprecationWarning)


def __getattr__(name: str) -> Any:  # there is no meaningful type for this style of universal shim
    leiden_name: str = name
    if "louvain" in name:
        leiden_name = name.replace("louvain", "leiden")
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(f"The Louvain functions have been deprecated. Replacing {repr(name)} with {repr(leiden_name)}.",
                      DeprecationWarning)
    return getattr(leiden_utilities, leiden_name)
