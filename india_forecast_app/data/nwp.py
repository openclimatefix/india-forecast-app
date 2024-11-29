import xarray as xr
import os
import logging

logger = logging.getLogger(__name__)


def regrid_nwp_data(nwp_zarr: str, target_coords_path: str, nwp_zarr_save: str):
    """This function loads the  NWP data, then regrids and saves it back out if the data is not
    on the same grid as expected. The data is resaved in-place.

    method can be 'conservative' or 'bilinear'
    """

    logger.info(f"Regridding NWP data {nwp_zarr} to expected grid to {target_coords_path}")

    ds_raw = xr.open_zarr(nwp_zarr)

    # These are the coords we are aiming for
    ds_target_coords = xr.load_dataset(target_coords_path)

    # Check if regridding step needs to be done
    needs_regridding = not (
        ds_raw.latitude.equals(ds_target_coords.latitude)
        and ds_raw.longitude.equals(ds_target_coords.longitude)
    )

    if not needs_regridding:
        logger.info(f"No NWP regridding required for {nwp_zarr} - skipping this step")
        return

    logger.info(f"Regridding NWP {nwp_zarr} to expected grid")

    # Pull the raw data into RAM
    ds_raw = ds_raw.compute()

    # regrid
    ds_regridded = ds_raw.interp(
        latitude=ds_target_coords.latitude, longitude=ds_target_coords.longitude
    )

    # Re-save - including rechunking
    os.system(f"rm -rf {nwp_zarr_save}")
    ds_regridded["variable"] = ds_regridded["variable"].astype(str)

    # Rechunk to these dimensions when saving
    save_chunk_dict = {
        "step": 5,
        "latitude": 100,
        "longitude": 100,
        "x": 100,
        "y": 100,
    }

    ds_regridded.chunk(
        {k: save_chunk_dict[k] for k in list(ds_raw.xindexes) if k in save_chunk_dict}
    ).to_zarr(nwp_zarr_save)
