"""India Forecast App"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("india-forecast-app")
except PackageNotFoundError:
    __version__ = "v?"
