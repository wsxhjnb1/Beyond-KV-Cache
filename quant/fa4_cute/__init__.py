"""Beyond's BSD-licensed SM103 non-uniform packed-attention kernel."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("beyond-cute")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
