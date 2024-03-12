"""Useful functions for setting up PVNet model"""
import logging

import fsspec
import xarray as xr
import yaml

from .consts import nwp_path, pv_metadata_path, pv_netcdf_path, wind_metadata_path, wind_netcdf_path

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
        "pv": {
            "filename": pv_netcdf_path,
            "metadata_filename": pv_metadata_path
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

    if "pv" in config["input_data"]:
        pv_config = config["input_data"]["pv"]
        assert "pv" in production_paths, "Missing production path: pv"
        pv_config["pv_files_groups"][0]["pv_filename"] = production_paths["pv"]['filename']
        pv_config["pv_files_groups"][0]["pv_metadata_filename"] = (
            production_paths)["pv"]['metadata_filename']

    log.debug(config)

    with open(output_path, 'w') as outfile:
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

    # Rename t variable to t2m
    variables = list(ds.variable.values)
    new_variables = []
    for var in variables:
        if 't' == var:
            new_variables.append('t2m')
            log.debug(f"Renamed t to t2m in NWP data {ds.variable.values}")
        elif 'clt' == var:
            new_variables.append('tcc')
            log.debug(f"Renamed clt to tcc in NWP data {ds.variable.values}")
        else:
            new_variables.append(var)
    ds.__setitem__('variable', new_variables)

    # Save destination path
    ds.to_zarr(dest_nwp_path, mode="a")


