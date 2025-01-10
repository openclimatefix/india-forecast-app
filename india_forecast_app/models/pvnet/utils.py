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

from india_forecast_app.data import nwp

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

    log.info(
        f"Processing and caching NWP data for {source_nwp_path} "
        f"and saving to {dest_nwp_path} for {nwp_config.source}"
    )

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

        if "HRES-IFS_india" in ds.data_vars:
            # rename from HRES-IFS_india to ECMWF_INDIA
            ds = ds.rename({"HRES-IFS_india": "ECMWF_INDIA"})

            # rename variable names in the variable coordinate
            # This is a renaming from ECMWF variables to what we use in the ML Model
            # This change happened in the new nwp-consumer>=1.0.0
            # Ideally we won't need this step in the future
            variable_coords = ds.variable.values
            rename = {'cloud_cover_high': 'hcc',
                      'cloud_cover_low': 'lcc',
                      'cloud_cover_medium': 'mcc',
                      'cloud_cover_total': 'tcc',
                      'snow_depth_gl': 'sde',
                      'direct_shortwave_radiation_flux_gl': 'sr',
                      'downward_longwave_radiation_flux_gl': 'dlwrf',
                      'downward_shortwave_radiation_flux_gl': 'dswrf',
                      'downward_ultraviolet_radiation_flux_gl': 'duvrs',
                      'temperature_sl': 't',
                      'total_precipitation_rate_gl': 'prate',
                      'visibility_sl': 'vis',
                      'wind_u_component_100m': 'u100',
                      'wind_u_component_10m': 'u10',
                      'wind_u_component_200m': 'u200',
                      'wind_v_component_100m': 'v100',
                      'wind_v_component_10m': 'v10',
                      'wind_v_component_200m': 'v200'}

            for k, v in rename.items():
                variable_coords[variable_coords == k] = v

            # assign the new variable names
            ds = ds.assign_coords(variable=variable_coords)

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

        # COMMENTED this out for the moment, as different models use different mo global variables
        # only select the variables we need
        # nwp_channels = list(nwp_config.config.nwp_channels)
        # log.info(f"Selecting NWP channels {nwp_channels} for mo_global data")
        # ds = ds.sel(variable=nwp_channels)

        # get directory of file
        regrid_coords = os.path.dirname(nwp.__file__)

        # regrid data
        ds = nwp.regrid_nwp_data(ds, f"{regrid_coords}/mo_global/india_coords.nc")

    # Save destination path
    log.info(f"Saving NWP data to {dest_nwp_path}")
    ds.to_zarr(dest_nwp_path, mode="a")


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
