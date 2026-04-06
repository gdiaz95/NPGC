from importlib.metadata import PackageNotFoundError, version

from .core import NPGC

try:
    __version__ = version("npgc")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["NPGC", "__version__"]
