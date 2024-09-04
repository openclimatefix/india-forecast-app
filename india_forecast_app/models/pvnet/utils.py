"""Useful functions for setting up PVNet model"""
import logging
import os
import random
from datetime import UTC, datetime, timedelta

import fsspec
import numpy as np
import xarray as xr
import yaml
from ocf_datapipes.batch import BatchKey

from .consts import (
    nwp_ecmwf_path,
    nwp_gfs_path,
    pv_metadata_path,
    pv_netcdf_path,
    satellite_path,
    wind_metadata_path,
    wind_netcdf_path,
)

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
        "wind": {"filename": wind_netcdf_path, "metadata_filename": wind_metadata_path},
        "pv": {"filename": pv_netcdf_path, "metadata_filename": pv_metadata_path},
        "nwp": {"ecmwf": nwp_ecmwf_path, "gfs": nwp_gfs_path},
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

    log.debug(config)

    with open(output_path, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def process_and_cache_nwp(source_nwp_path: str, dest_nwp_path: str):
    """Reads zarr file, renames t variable to t2m and saves zarr to new destination"""

    # Load dataset from source
    ds = xr.open_zarr(source_nwp_path)
    

    # This is important to avoid saving errors
    for v in list(ds.coords.keys()):
        if ds.coords[v].dtype == object:
            ds[v].encoding.clear()

    for v in list(ds.variables.keys()):
        if ds[v].dtype == object:
            ds[v].encoding.clear()

    is_gfs = "gfs" in source_nwp_path.lower()

    if not is_gfs: # this is for ECMWF NWP
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
    elif is_gfs:
        data_var = ds[list(ds.data_vars.keys())[0]]
        # # Use .to_dataset() to split the data variable based on 'variable' dim
        ds = data_var.to_dataset(dim='variable') 
        ds = ds.rename({"t2m": "t"})
    # Save destination path
    ds.to_zarr(dest_nwp_path, mode="a")

def download_satellite_data(satellite_source_file_path: str) -> None:
    """Download the sat data"""

    # download satellite data
    fs = fsspec.open(satellite_source_file_path).fs
    if fs.exists(satellite_source_file_path):
        fs.get(satellite_source_file_path, "sat_5_min.zarr.zip")
        os.system(f"unzip -qq sat_5_min.zarr.zip -d {satellite_path}")

def set_night_time_zeros(batch, preds, sun_elevation_limit=0.0):
    """
    Set all predictions to zero for night time values
    """
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

    # expand dimension from (1,197) to (1,197,7), 7 is due to the number plevels
    sun_elevation = np.repeat(sun_elevation[:, :, np.newaxis], 7, axis=2)
    # only take future time steps
    sun_elevation = sun_elevation[:, batch[t0_key] + 1 :, :]
    preds[sun_elevation < sun_elevation_limit] = 0

    return preds

# This section is to be deleted after generation data for ad sites is available
class FakeGenerationData:
    """Class to generate Fake data"""
    def __init__(self, start_utc, generation_power_kw):
        """ Initiate fake data """
        self.start_utc = start_utc
        self.generation_power_kw = generation_power_kw


def generate_fake_generation_data():
    """Generate fake 15 minutely generation data from delta minus 1 hour to now"""
    end_time = datetime.now(UTC).replace(second=0, microsecond=0, minute=0) + timedelta(minutes=15)
    start_time = end_time - timedelta(hours=1)
    
    generation_data = []
    current_time = start_time
    
    while current_time < end_time:
        # Simulate power generation between 0 and 200,000 kW
        power_kw = random.uniform(0, 200000)
        generation_data.append(FakeGenerationData(current_time, power_kw))
        current_time += timedelta(minutes=15)
    
    return generation_data
