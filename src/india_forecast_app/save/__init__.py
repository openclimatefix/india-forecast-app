"""Public API for the save subpackage."""

from india_forecast_app.save.data_platform import build_dp_location_map
from india_forecast_app.save.save import save_forecast

__all__ = [
    "build_dp_location_map",
    "save_forecast",
]
