"""Useful functions for setting up PVNet model"""
import logging
import os
from typing import Optional

import fsspec
import numpy as np
import torch
import xarray as xr
import yaml
from ocf_datapipes.batch import BatchKey
from ocf_datapipes.config.model import NWP
from ocf_datapipes.utils.consts import ELEVATION_MEAN, ELEVATION_STD
from pydantic import BaseModel

from india_forecast_app.data.nwp import regrid_nwp_data
from .consts import (
    nwp_ecmwf_path,
    nwp_gfs_path,
    nwp_mo_global_path,
    pv_metadata_path,
    pv_netcdf_path,
    satellite_path,
    wind_metadata_path,
    wind_netcdf_path,
)

log = logging.getLogger(__name__)


class NWPProcessAndCacheConfig(BaseModel):
    """Configuration for processing and caching NWP data"""

    source_nwp_path: str
    dest_nwp_path: str
    source: str
    config: Optional[NWP] = None


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
        "wind": {"filename": wind_netcdf_path, "metadata_filename": wind_metadata_path},
        "pv": {"filename": pv_netcdf_path, "metadata_filename": pv_metadata_path},
        "nwp": {"ecmwf": nwp_ecmwf_path, "gfs": nwp_gfs_path, "mo_global": nwp_mo_global_path},
        "satellite": {"filepath": satellite_path},
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
        wind_config["wind_files_groups"][0]["wind_filename"] = production_paths["wind"]["filename"]
        wind_config["wind_files_groups"][0]["wind_metadata_filename"] = (production_paths)["wind"][
            "metadata_filename"
        ]
    if "pv" in config["input_data"]:
        pv_config = config["input_data"]["pv"]
        assert "pv" in production_paths, "Missing production path: pv"
        pv_config["pv_files_groups"][0]["pv_filename"] = production_paths["pv"]["filename"]
        pv_config["pv_files_groups"][0]["pv_metadata_filename"] = (production_paths)["pv"][
            "metadata_filename"
        ]
    if "satellite" in config["input_data"]:
        satellite_config = config["input_data"]["satellite"]
        assert "satellite" in production_paths, "Missing production path: satellite"
        satellite_config["satellite_zarr_path"] = production_paths["satellite"]["filepath"]

    log.debug(config)

    with open(output_path, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    return config


def process_and_cache_nwp(nwp_config: NWPProcessAndCacheConfig):
    """Reads zarr file, renames t variable to t2m and saves zarr to new destination"""

    source_nwp_path = nwp_config.source_nwp_path
    dest_nwp_path = nwp_config.dest_nwp_path

    log.info(f"Processing and caching NWP data for {source_nwp_path} "
             f"and saving to {dest_nwp_path} for {nwp_config.source}")

    if os.path.exists(dest_nwp_path):
        log.info(f"File already exists at {dest_nwp_path}")
        return

    # Load dataset from source
    ds = xr.open_zarr(source_nwp_path)

    # This is important to avoid saving errors
    for v in list(ds.coords.keys()):
        if ds.coords[v].dtype == object:
            ds[v].encoding.clear()

    for v in list(ds.variables.keys()):
        if ds[v].dtype == object:
            ds[v].encoding.clear()

    if nwp_config.source == "ecmwf":
        # Rename t variable to t2m
        variables = list(ds.variable.values)
        new_variables = []
        for var in variables:
            if "t" == var:
                new_variables.append("t2m")
                log.debug(f"Renamed t to t2m in NWP data {ds.variable.values}")
            elif "clt" == var:
                new_variables.append("tcc")
                log.debug(f"Renamed clt to tcc in NWP data {ds.variable.values}")
            else:
                new_variables.append(var)
        ds.__setitem__("variable", new_variables)

    # Hack to resolve some NWP data format differences between providers
    elif nwp_config.source == "gfs":
        data_var = ds[list(ds.data_vars.keys())[0]]
        # # Use .to_dataset() to split the data variable based on 'variable' dim
        ds = data_var.to_dataset(dim="variable")
        ds = ds.rename({"t2m": "t"})

    if nwp_config.source == "mo_global":

        # only select the variables we need
        nwp_channels = nwp_config.config.nwp_channels
        ds = ds.sel(variable=nwp_channels)

        # regrid data
        ds = regrid_nwp_data(
            ds, "india_forecast_app/data/mo_global/india_coords.nc"
        )

    # Save destination path
    log.info(f"Saving NWP data to {dest_nwp_path}")
    ds.to_zarr(dest_nwp_path, mode="a")


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


def set_night_time_zeros(batch, preds, sun_elevation_limit=0.0):
    """
    Set all predictions to zero for night time values
    """

    log.debug("Setting night time values to zero")
    # get sun elevation values and if less 0, set to 0
    if BatchKey.wind_solar_elevation in batch.keys():
        key = BatchKey.wind_solar_elevation
        t0_key = BatchKey.wind_t0_idx
    elif BatchKey.pv_solar_elevation in batch.keys():
        key = BatchKey.pv_solar_elevation
        t0_key = BatchKey.pv_t0_idx
    else:
        log.warning(
            f'Could not find "wind_solar_elevation" or "pv_solar_elevation" '
            f"key in {batch.keys()}"
        )
        raise Exception('Could not find "wind_solar_elevation" or "pv_solar_elevation" ')

    sun_elevation = batch[key]
    if not isinstance(sun_elevation, np.ndarray):
        sun_elevation = sun_elevation.detach().cpu().numpy()

    # un normalize elevation
    sun_elevation = sun_elevation * ELEVATION_STD + ELEVATION_MEAN

    # expand dimension from (1,197) to (1,197,7), 7 is due to the number plevels
    n_plevels = preds.shape[2]
    sun_elevation = np.repeat(sun_elevation[:, :, np.newaxis], n_plevels, axis=2)
    # only take future time steps
    sun_elevation = sun_elevation[:, batch[t0_key] + 1 :, :]
    preds[sun_elevation < sun_elevation_limit] = 0

    return preds


def save_batch(batch, i: int, model_name, site_uuid, save_batches_dir: Optional[str] = None):
    """
    Save batch to SAVE_BATCHES_DIR if set

    Args:
        batch: The batch to save
        i: The index of the batch
        model_name: The name of the
        site_uuid: The site_uuid of the site
        save_batches_dir: The directory to save the batch to,
            defaults to environment variable SAVE_BATCHES_DIR
    """

    if save_batches_dir is None:
        save_batches_dir = os.getenv("SAVE_BATCHES_DIR", None)

    if save_batches_dir is not None:
        log.info(f"Saving batch {i} to {save_batches_dir}")

        local_filename = f"batch_{i}_{model_name}_{site_uuid}.pt"
        torch.save(batch, local_filename)

        fs = fsspec.open(save_batches_dir).fs
        fs.put(local_filename, f"{save_batches_dir}/{local_filename}")
