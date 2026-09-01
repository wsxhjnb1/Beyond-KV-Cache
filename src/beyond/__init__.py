"""Beyond's stable Python package.

The package intentionally performs no CUDA, model, dataset, or vLLM imports at
module import time. Runtime-specific components live in explicit subpackages.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("beyond-kv")
except PackageNotFoundError:  # Source checkout without an editable install.
    __version__ = "0.1.0"


__all__ = ["__version__"]
