"""frameopt package initialisation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("frameopt")
except PackageNotFoundError:
    __version__ = "0.1.0"
