"""Useful functions for setting up PVNet model"""

import fsspec
import yaml


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
        # "wind": os.environ["DB_URL"],
        "nwp": {"ecmwf": "nwp.zarr"}
    }

    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config.keys():
            if nwp_config[nwp_source]["nwp_zarr_path"] != "":
                assert "nwp" in production_paths, "Missing production path: nwp"
                assert nwp_source in production_paths["nwp"], f"Missing NWP path: {nwp_source}"
                nwp_config[nwp_source]["nwp_zarr_path"] = production_paths["nwp"][nwp_source]

    # We do not need to set wind/PV path right now. This currently done through datapipes
    # TODO - Move the wind/PV path to here?

    with open(output_path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
