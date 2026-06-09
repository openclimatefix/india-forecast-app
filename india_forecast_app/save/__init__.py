"""Public API for the save subpackage."""

import ocf.dp as dp  # noqa: F401
from grpclib.client import Channel  # noqa: F401

from india_forecast_app.save.data_platform import (
    DataPlatformClient,
    build_dp_location_map,
    fetch_dp_location_map,
    get_dataplatform_client,
    make_forecaster_adjuster,
    save_forecast_to_dataplatform,
    save_to_dataplatform,
)
from india_forecast_app.save.save import save_forecast
from india_forecast_app.save.utils import limit_adjuster  # noqa: F401

__all__ = [
    "Channel",
    "DataPlatformClient",
    "build_dp_location_map",
    "dp",
    "fetch_dp_location_map",
    "get_dataplatform_client",
    "limit_adjuster",
    "make_forecaster_adjuster",
    "save_forecast",
    "save_forecast_to_dataplatform",
    "save_to_dataplatform",
]
