"""
PVNet model class
"""

import datetime as dt
import logging
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch
from ocf_datapipes.batch import stack_np_examples_into_batch
from ocf_datapipes.training.pvnet_site import construct_sliced_data_pipeline as pv_base_pipeline
from ocf_datapipes.training.windnet import DictDatasetIterDataPipe, split_dataset_dict_dp
from ocf_datapipes.training.windnet import construct_sliced_data_pipeline as wind_base_pipeline
from ocf_datapipes.utils import Location
from pvnet.data.utils import batch_to_tensor, copy_batch_to_device
from pvnet.models.base_model import BaseModel as PVNetBaseModel
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter import IterableWrapper

from .consts import (
    nwp_path,
    pv_metadata_path,
    pv_netcdf_path,
    pv_path,
    root_data_path,
    wind_metadata_path,
    wind_netcdf_path,
    wind_path,
)
from .utils import (
    populate_data_config_sources,
    process_and_cache_nwp,
    worker_init_fn,
)

# Global settings for running the model

# Model will use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WIND_MODEL_NAME = os.getenv("WIND_MODEL_NAME", default="openclimatefix/windnet_india")
WIND_MODEL_VERSION = os.getenv(
    "WIND_MODEL_VERSION", default="5fdfc7e6995f491106a94978376d5b0ee1d2b371"
)

PV_MODEL_NAME = os.getenv("PV_MODEL_NAME", default="openclimatefix/pvnet_india")
PV_MODEL_VERSION = os.getenv(
    "PV_MODEL_VERSION", default="3196e09b35879b207435f8b6fe4f6e21322dd7f7")

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

    def __init__(
        self, asset_type: str, timestamp: dt.datetime, generation_data: dict[str, pd.DataFrame]
    ):
        """Initializer for the model"""

        self.asset_type = asset_type
        self.t0 = timestamp
        log.info(f"Model initialised at t0={self.t0}")

        # Setup the data, dataloader, and model
        self.generation_data = generation_data
        self._prepare_data_sources()
        self.dataloader = self._create_dataloader()
        self.model = self._load_model()

    def predict(self, site_id: str, timestamp: dt.datetime):
        """Make a prediction for the model"""

        capacity_kw = self.generation_data["metadata"].iloc[0]["capacity_megawatts"] * 1000

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
        valid_times = pd.to_datetime(
            [self.t0 + dt.timedelta(minutes=15 * (i + 1)) for i in range(n_times)]
        )

        values_df = pd.DataFrame(
            [
                {
                    "start_utc": valid_times[i],
                    "end_utc": valid_times[i] + dt.timedelta(minutes=15),
                    "forecast_power_kw": int(v * capacity_kw),
                }
                for i, v in enumerate(normed_preds[0, :, 3])
            ]
        )  # index 3 is the 50th percentile)

        # smooth with a 1 hour rolling window
        values_df["forecast_power_kw"] = (
            values_df["forecast_power_kw"].rolling(4, min_periods=1).mean().astype(int)
        )

        # remove any negative values
        values_df["forecast_power_kw"] = values_df["forecast_power_kw"].clip(lower=0.0)

        return values_df.to_dict("records")

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

        # Remove local cached zarr if already exists
        shutil.rmtree(nwp_path, ignore_errors=True)

        # Process/cache remote zarr locally
        process_and_cache_nwp(nwp_source_file_path, nwp_path)

        if self.asset_type == "wind":
            # Clear local cached wind data if already exists
            shutil.rmtree(wind_path, ignore_errors=True)
            os.mkdir(wind_path)

            # Save generation data as netcdf file
            generation_da = self.generation_data["data"].to_xarray()
            generation_da.to_netcdf(wind_netcdf_path, engine="h5netcdf")

            # Save metadata as csv
            self.generation_data["metadata"].to_csv(wind_metadata_path, index=False)

        if self.asset_type == "pv":
            # Clear local cached wind data if already exists
            shutil.rmtree(pv_path, ignore_errors=True)
            os.mkdir(pv_path)

            # Save generation data as netcdf file
            generation_da = self.generation_data["data"].to_xarray()
            generation_da.to_netcdf(pv_netcdf_path, engine="h5netcdf")

            # Save metadata as csv
            self.generation_data["metadata"].to_csv(pv_metadata_path, index=False)

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
        gen_sites = self.generation_data["metadata"]
        location_pipe = IterableWrapper(
            [
                Location(coordinate_system="lon_lat", x=s.longitude, y=s.latitude)
                for s in gen_sites.itertuples()
            ]
        )
        t0_datapipe = IterableWrapper([self.t0 for _ in range(gen_sites.shape[0])])

        location_pipe = location_pipe.sharding_filter()
        t0_datapipe = t0_datapipe.sharding_filter()

        batch_size = 1

        # Batch datapipe
        if self.asset_type == "wind":
            base_datapipe_dict = wind_base_pipeline(
                config_filename=populated_data_config_filename,
                location_pipe=location_pipe,
                t0_datapipe=t0_datapipe,
                upsample_nwp=True,
            )

            base_datapipe = DictDatasetIterDataPipe(
                {k: v for k, v in base_datapipe_dict.items() if k != "config"},
            ).map(split_dataset_dict_dp)

            batch_datapipe = (
                base_datapipe.windnet_convert_to_numpy_batch()
                .batch(batch_size)
                .map(stack_np_examples_into_batch)
            )

        else:
            base_datapipe_dict = pv_base_pipeline(
                config_filename=populated_data_config_filename,
                location_pipe=location_pipe,
                t0_datapipe=t0_datapipe,
            )

            base_datapipe = DictDatasetIterDataPipe(
                {k: v for k, v in base_datapipe_dict.items() if k != "config"},
            ).map(split_dataset_dict_dp)

            batch_datapipe = (
                base_datapipe.pvnet_site_convert_to_numpy_batch()
                .batch(batch_size)
                .map(stack_np_examples_into_batch)
            )

            # batch_datapipe = base_datapipe.batch(batch_size).map(stack_np_examples_into_batch)

        n_workers = 0

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
            prefetch_factor=None,
            persistent_workers=False,
        )

        dataloader = DataLoader(batch_datapipe, **dataloader_kwargs)

        return dataloader

    def _load_model(self):
        """Load model"""

        log.info(f"Loading model: {self.name} - {self.version}")
        return PVNetBaseModel.from_pretrained(self.name, revision=self.version).to(DEVICE)
