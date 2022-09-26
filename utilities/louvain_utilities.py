"""This is a deprecated version of leiden_utilities. Usage will be shimmed to use Leiden instead of Louvain."""
import warnings

warnings.simplefilter('always', DeprecationWarning)
warnings.warn("modularitypruning.louvain_utilities has been replaced. Please use leiden_utilities in the future.",
              DeprecationWarning)

# TODO: ensure remaining references to Louvain (especially in docstrings, documentation, etc.) are removed
from .leiden_utilities import *
from . import leiden_utilities


def __getattr__(name):
    if "louvain" in name:
        leiden_name = name.replace("louvain", "leiden")
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(f"The Louvain functions have been deprecated. Replacing {repr(name)} with {repr(leiden_name)}.",
                      DeprecationWarning)
        return getattr(leiden_utilities, leiden_name)
