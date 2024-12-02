import xarray as xr
import os
import logging

logger = logging.getLogger(__name__)


def regrid_nwp_data(nwp_ds: xr.Dataset, target_coords_path: str) -> xr.Dataset:
    """This function loads the  NWP data, then regrids and saves it back out if the data is not
    on the same grid as expected. The data is resaved in-place.
    """

    logger.info(f"Regridding NWP data to expected grid to {target_coords_path}")

    ds_raw = nwp_ds

    # These are the coords we are aiming for
    ds_target_coords = xr.load_dataset(target_coords_path)

    # Check if regridding step needs to be done
    needs_regridding = not (
        ds_raw.latitude.equals(ds_target_coords.latitude)
        and ds_raw.longitude.equals(ds_target_coords.longitude)
    )

    if not needs_regridding:
        logger.info(f"No NWP regridding required - skipping this step")
        return ds_raw

    # flip latitude, so its in ascending order
    if ds_raw.latitude[0] > ds_raw.latitude[-1]:
        ds_raw = ds_raw.reindex(latitude=ds_raw.latitude[::-1])

    # clip to india coordindates
    ds_raw = ds_raw.sel(
        latitude=slice(0, 40),
        longitude=slice(65, 100),
    )

    # regrid
    logger.info(f"Regridding NWP to expected grid")
    ds_regridded = ds_raw.interp(
        latitude=ds_target_coords.latitude, longitude=ds_target_coords.longitude
    )

    # rechunking
    ds_regridded["variable"] = ds_regridded["variable"].astype(str)

    # Rechunk to these dimensions when saving
    save_chunk_dict = {
        "step": 5,
        "latitude": 100,
        "longitude": 100,
        "x": 100,
        "y": 100,
    }

    ds_regridded = ds_regridded.chunk(
        {k: save_chunk_dict[k] for k in list(ds_raw.xindexes) if k in save_chunk_dict}
    )

    return ds_regridded
