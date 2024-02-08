"""
PVNet model class
"""

import datetime as dt
import logging
import os
import shutil
import tempfile

import fsspec
import numpy as np
import pandas as pd
import torch
from ocf_datapipes.batch import stack_np_examples_into_batch
from ocf_datapipes.training.pvnet import construct_sliced_data_pipeline as pv_base_pipeline
from ocf_datapipes.training.windnet import DictDatasetIterDataPipe
from ocf_datapipes.training.windnet import construct_sliced_data_pipeline as wind_base_pipeline
from ocf_datapipes.utils import Location
from ocf_datapipes.utils.utils import combine_to_single_dataset
from pvnet.data.utils import batch_to_tensor, copy_batch_to_device
from pvnet.models.base_model import BaseModel as PVNetBaseModel
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter import IterableWrapper

from .consts import (root_data_path, nwp_path
, wind_metadata_path, wind_netcdf_path, wind_path)
from .utils import populate_data_config_sources, worker_init_fn

# Global settings for running the model

# Model will use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WIND_MODEL_NAME = os.getenv("WIND_MODEL_NAME", default="openclimatefix/windnet_india")
WIND_MODEL_VERSION = os.getenv("WIND_MODEL_VERSION",
                               default="c6af802823edc5e87b22df680b41b0dcdb4869e1")

PV_MODEL_NAME = os.getenv("PV_MODEL_NAME", default="openclimatefix/pvnet_india")
PV_MODEL_VERSION = os.getenv("PV_MODEL_VERSION",
                             default="d194488203375e766253f0d2961010356de52eb9")

BATCH_SIZE = 10

log = logging.getLogger(__name__)


class PVNetModel:
    """
    Instantiates a PVNet model for inference
    """

    @property
    def name(self):
        """Model name"""

        return WIND_MODEL_NAME if self.asset_type == "wind" else PV_MODEL_NAME

    @property
    def version(self):
        """Model version"""

        return WIND_MODEL_VERSION if self.asset_type == "wind" else PV_MODEL_VERSION

    def __init__(self, asset_type: str, timestamp: dt.datetime):
        """Initializer for the model"""

        self.asset_type = asset_type
        self.t0 = timestamp

        # Setup the data, dataloader, and model
        self._prepare_data_sources()
        self.dataloader = self._create_dataloader()
        self.model = self._load_model()

    def predict(self, site_id: str, timestamp: dt.datetime):
        """Make a prediction for the model"""

        normed_preds = []
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                log.info(f"Predicting for batch: {i}")

                # Run batch through model
                device_batch = copy_batch_to_device(batch_to_tensor(batch), DEVICE)
                preds = self.model(device_batch).detach().cpu().numpy()

                # Store predictions
                normed_preds += [preds]

                # log max prediction
                log.info(f"Max prediction: {np.max(preds, axis=1)}")
                log.info(f"Completed batch: {i}")

        normed_preds = np.concatenate(normed_preds)
        n_times = normed_preds.shape[1]
        valid_times = pd.to_datetime([self.t0 + dt.timedelta(minutes=30 * (i + 1))
                                      for i in range(n_times)])

        return [{
            "start_utc": valid_times[i],
            "end_utc": valid_times[i] + dt.timedelta(minutes=30),
            "forecast_power_kw": v
        } for i, v in enumerate(normed_preds)]

    def _prepare_data_sources(self):
        """Pull and prepare data sources required for inference"""

        log.info("Preparing data sources")

        # Create root data directory if not exists
        try:
            os.mkdir(root_data_path)
        except FileExistsError:
            pass

        # Load remote zarr source
        nwp_source_file_path = os.environ["NWP_ZARR_PATH"]
        fs = fsspec.open(nwp_source_file_path).fs

        # Remove local zarr if already exists
        shutil.rmtree(nwp_path, ignore_errors=True)

        # Copy remote zarr locally
        fs.get(nwp_source_file_path, nwp_path, recursive=True)

        if self.asset_type == "wind":
            # Load remote netcdf and metadata sources
            wind_netcdf_source_file_path = os.environ["WIND_NETCDF_PATH"]
            wind_metadata_source_file_path = os.environ["WIND_METADATA_PATH"]

            # Clear local wind data if already exists
            shutil.rmtree(wind_path, ignore_errors=True)
            os.mkdir(wind_path)

            # Copy remote netcdf locally
            fs = fsspec.open(wind_netcdf_source_file_path).fs
            fs.get(wind_netcdf_source_file_path,
                   wind_netcdf_path,
                   recursive=True)

            # Copy remote metadata locally
            fs = fsspec.open(wind_metadata_source_file_path).fs
            fs.get(wind_metadata_source_file_path,
                   wind_metadata_path,
                   recursive=True)

    def _create_dataloader(self):
        """Setup dataloader with prepared data sources"""

        log.info("Creating dataloader")

        # Pull the data config from huggingface
        data_config_filename = PVNetBaseModel.get_data_config(
            self.name,
            revision=self.version,
        )

        # Populate the data config with production data paths
        temp_dir = tempfile.TemporaryDirectory()
        populated_data_config_filename = f"{temp_dir.name}/data_config.yaml"

        populate_data_config_sources(data_config_filename, populated_data_config_filename)

        # Location and time datapipes
        # TODO not sure what to use here for the location pipe - site uuid/location?
        location_pipe = IterableWrapper([Location(coordinate_system="lon_lat", x=72.6399, y=26.4499)])
        t0_datapipe = IterableWrapper([self.t0])
        # t0_datapipe = IterableWrapper([self.t0]).repeat(len(location_pipe))

        location_pipe = location_pipe.sharding_filter()
        t0_datapipe = t0_datapipe.sharding_filter()

        # Batch datapipe
        if self.asset_type == "wind":
            base_datapipe_dict = (
                wind_base_pipeline(
                    config_filename=populated_data_config_filename,
                    location_pipe=location_pipe,
                    t0_datapipe=t0_datapipe
                )
            )
            log.error(base_datapipe_dict)
            base_datapipe = DictDatasetIterDataPipe(
                {k: v for k, v in base_datapipe_dict.items() if k != "config"},
            ).map(combine_to_single_dataset)
        else:
            base_datapipe = (
                pv_base_pipeline(
                    config_filename=populated_data_config_filename,
                    location_pipe=location_pipe,
                    t0_datapipe=t0_datapipe,
                    production=True
                )
            )

        batch_datapipe = base_datapipe.batch(BATCH_SIZE).map(stack_np_examples_into_batch)

        n_workers = os.cpu_count() - 1

        # Set up dataloader for parallel loading
        dataloader_kwargs = dict(
            shuffle=False,
            batch_size=None,  # batched in datapipe step
            sampler=None,
            batch_sampler=None,
            num_workers=n_workers,
            collate_fn=None,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=worker_init_fn,
            prefetch_factor=None if n_workers == 0 else 2,
            persistent_workers=False,
        )

        return DataLoader(batch_datapipe, **dataloader_kwargs)

    def _load_model(self):
        """Load model"""

        log.info(f"Loading model: {self.name} - {self.version}")
        return PVNetBaseModel.from_pretrained(
            self.name,
            revision=self.version
        ).to(DEVICE)
