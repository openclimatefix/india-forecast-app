import os

import fsspec

from india_forecast_app.models.pvnet.consts import satellite_path
from india_forecast_app.models.pvnet.utils import log


def download_satellite_data(satellite_source_file_path: str) -> None:
    """Download the sat data"""

    # download satellite data
    fs = fsspec.open(satellite_source_file_path).fs
    if fs.exists(satellite_source_file_path):
        log.info(
            f"Downloading satellite data from {satellite_source_file_path} "
            f"to sat_15_min.zarr.zip"
        )
        fs.get(satellite_source_file_path, "sat_15_min.zarr.zip")
        log.info(f"Unzipping sat_15_min.zarr.zip to {satellite_path}")
        os.system(f"unzip -qq sat_15_min.zarr.zip -d {satellite_path}")
    else:
        log.error(f"Could not find satellite data at {satellite_source_file_path}")


def check_satellite_data() -> None:
    """Check if satellite data exists"""

    if not os.path.exists(satellite_path):
        log.error(f"Satellite data not found at {satellite_path}")
        raise FileNotFoundError(f"Satellite data not found at {satellite_path}")
    log.info(f"Satellite data found at {satellite_path}")

    # opent he data and log the timestamps
    log.info("Opening satellite data")
    