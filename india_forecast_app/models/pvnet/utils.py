"""Useful functions for setting up PVNet model"""
import datetime as dt
import logging

import fsspec
import pandas as pd
import xarray as xr
import yaml

from .consts import nwp_path, wind_metadata_path, wind_netcdf_path

log = logging.getLogger(__name__)


def worker_init_fn(worker_id):
    """
    Clear reference to the loop and thread.

    This is a nasty hack that was suggested but NOT recommended by the lead fsspec developer!
    This appears necessary otherwise gcsfs hangs when used after forking multiple worker processes.
    Only required for fsspec >= 0.9.0
    See:
    - https://github.com/fsspec/gcsfs/issues/379#issuecomment-839929801
    - https://github.com/fsspec/filesystem_spec/pull/963#issuecomment-1131709948
    TODO: Try deleting this two lines to make sure this is still relevant.
    """
    fsspec.asyn.iothread[0] = None
    fsspec.asyn.loop[0] = None


def populate_data_config_sources(input_path, output_path):
    """Re-save the data config and replace the source filepaths

    Args:
        input_path: Path to input datapipes configuration file
        output_path: Location to save the output configuration file
    """
    with open(input_path) as infile:
        config = yaml.load(infile, Loader=yaml.FullLoader)

    production_paths = {
        "wind": {
            "filename": wind_netcdf_path,
            "metadata_filename": wind_metadata_path
        },
        "nwp": {
            "ecmwf": nwp_path
        }
    }

    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config.keys():
            if nwp_config[nwp_source]["nwp_zarr_path"] != "":
                assert "nwp" in production_paths, "Missing production path: nwp"
                assert nwp_source in production_paths["nwp"], f"Missing NWP path: {nwp_source}"
                nwp_config[nwp_source]["nwp_zarr_path"] = production_paths["nwp"][nwp_source]

    if "wind" in config["input_data"]:
        wind_config = config["input_data"]["wind"]
        assert "wind" in production_paths, "Missing production path: wind"
        wind_config["wind_files_groups"][0]["wind_filename"] = production_paths["wind"]['filename']
        wind_config["wind_files_groups"][0]["wind_metadata_filename"] = (
            production_paths)["wind"]['metadata_filename']

    log.debug(config)

    with open(output_path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def reset_stale_nwp_timestamps_and_rename_t(source_nwp_path: str):
    """Resets the init_time values of the NWP zarr to more recent timestamps"""

    # Load dataset from source
    ds = xr.open_zarr(source_nwp_path)

    # Set t0 now and floor to 3-hour interval
    t0_datetime_utc = (pd.Timestamp.now(tz=None).floor(dt.timedelta(hours=3)))
    t0_datetime_utc = t0_datetime_utc - dt.timedelta(hours=1)
    ds.init_time.values[:] = pd.date_range(
        t0_datetime_utc - dt.timedelta(hours=3 * (len(ds.init_time) - 1)),
        t0_datetime_utc,
        freq=dt.timedelta(hours=3),
    )

    # This is important to avoid saving errors
    for v in list(ds.coords.keys()):
        if ds.coords[v].dtype == object:
            ds[v].encoding.clear()

    for v in list(ds.variables.keys()):
        if ds[v].dtype == object:
            ds[v].encoding.clear()

    # get list of variables
    variables = list(ds.variable.values)
    new_variables = []
    for var in variables:
        if 't2m' == var:
            new_variables.append('t')
        else:
            new_variables.append(var)
    ds.__setitem__('variables', new_variables)

    # Save back down to source path
    ds.to_zarr(source_nwp_path, mode="a")


