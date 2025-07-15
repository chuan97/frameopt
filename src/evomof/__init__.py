"""evomof package initialisation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("evomof")
except PackageNotFoundError:
    __version__ = "0.1.0"

# src/evomof/__init__.py
from .core.frame import Frame

__all__ = ["Frame"]
