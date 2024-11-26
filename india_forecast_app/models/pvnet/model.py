"""
PVNet model class
"""

import datetime as dt
import logging
import os
import shutil
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import torch
from ocf_datapipes.batch import batch_to_tensor, copy_batch_to_device, stack_np_examples_into_batch
from ocf_datapipes.training.pvnet_site import construct_sliced_data_pipeline as pv_base_pipeline
from ocf_datapipes.training.windnet import DictDatasetIterDataPipe, split_dataset_dict_dp
from ocf_datapipes.training.windnet import construct_sliced_data_pipeline as wind_base_pipeline
from ocf_datapipes.utils import Location
from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvsite_datamodel.sqlmodels import SiteAssetType
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter import IterableWrapper

from .consts import (
    nwp_ecmwf_path,
    nwp_gfs_path,
    nwp_mo_global_path,
    pv_metadata_path,
    pv_netcdf_path,
    pv_path,
    root_data_path,
    satellite_path,
    wind_metadata_path,
    wind_netcdf_path,
    wind_path,
)
from .utils import (
    download_satellite_data,
    populate_data_config_sources,
    process_and_cache_nwp,
    set_night_time_zeros,
    worker_init_fn,
)

# Global settings for running the model

# Model will use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log = logging.getLogger(__name__)


class PVNetModel:
    """
    Instantiates a PVNet model for inference
    """

    def __init__(
        self,
        asset_type: str,
        timestamp: dt.datetime,
        generation_data: dict[str, pd.DataFrame],
        hf_repo: str,
        hf_version: str,
        name: str,
        smooth_blocks: Optional[int] = 0,
    ):
        """Initializer for the model"""

        self.asset_type = asset_type
        self.id = hf_repo
        self.version = hf_version
        self.name = name
        self.site_uuid = None
        self.t0 = timestamp
        self.smooth_blocks = smooth_blocks
        log.info(f"Model initialised at t0={self.t0}")

        self.client = os.getenv("CLIENT_NAME", "ruvnl")
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")

        # Setup the data, dataloader, and model
        self.generation_data = generation_data
        self.dataloader = self._create_dataloader()
        self._prepare_data_sources()
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

                # filter out night time
                if self.asset_type == SiteAssetType.pv.name:
                    preds = set_night_time_zeros(batch, preds)

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

        # index of the 50th percentile, assumed number of p values odd and in order
        middle_plevel_index = normed_preds.shape[2] // 2

        values_df = pd.DataFrame(
            [
                {
                    "start_utc": valid_times[i],
                    "end_utc": valid_times[i] + dt.timedelta(minutes=15),
                    "forecast_power_kw": int(v * capacity_kw),
                }
                for i, v in enumerate(normed_preds[0, :, middle_plevel_index])
            ]
        )

        if self.asset_type == "wind":

            log.info("Feathering the forecast to the lastest value of generation")

            # Feather in the last generation, if it exists
            system_id = str(self.generation_data["data"].columns[0])
            generation_da = self.generation_data["data"].to_xarray()

            # Check if the generation exists, if so, take the value at t0 and
            # feather it in over the next 8 timesteps (2 hours)
            if self.t0 in generation_da.index.values:
                final_gen_points = 0
                final_gen_index = 0
                for gen_idx in range(len(generation_da.index.values) - 1, -1, -1):
                    current_gen = generation_da.isel(index=gen_idx)
                    current_gen = current_gen[system_id].values
                    if not np.isnan(current_gen) and current_gen > 0:
                        final_gen_points = current_gen * 1000.0
                        # Convert to KW back from MW
                        # Orig conversion is line 112 in app.py
                        break
                    final_gen_index += 1
                log.info(
                    f"The final generation values is {final_gen_points}"
                    f" at index {final_gen_index}"
                )

                # Feather in the difference between this value and the next forecasted values
                smooth_values = [
                    0.8,
                    0.7,
                    0.6,
                    0.5,
                    0.4,
                    0.3,
                    0.2,
                    0.1,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                log.debug(f"Previous values are {values_df['forecast_power_kw']}")
                for idx in range(8):
                    values_df["forecast_power_kw"][idx] -= (
                        values_df["forecast_power_kw"][idx] - final_gen_points
                    ) * smooth_values[final_gen_index + idx]
                log.debug(f"New values are {values_df['forecast_power_kw']}")

        if self.asset_type == "wind":
            # Smooth with a 1 hour rolling window
            # Only smooth the wind else we introduce too much of a lag in the solar
            # going up and down throughout the day
            values_df["forecast_power_kw"] = (
                values_df["forecast_power_kw"].rolling(4, min_periods=1).mean().astype(int)
            )

        if self.smooth_blocks:
            log.info(f"Smoothing the forecast with {self.smooth_blocks} blocks")
            values_df["forecast_power_kw"] = (
                values_df["forecast_power_kw"]
                .rolling(window=self.smooth_blocks, min_periods=1, center=True)
                .mean()
                .astype(int)
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
        use_satellite = os.getenv("USE_SATELLITE", "false").lower() == "true"
        satellite_source_file_path = os.getenv("SATELLITE_ZARR_PATH", None)

        # only load nwp that we need
        nwp_paths = []
        nwp_source_file_paths = []
        nwp_keys = self.config["input_data"]["nwp"].keys()
        if "ecmwf" in nwp_keys:
            nwp_ecmwf_source_file_path = os.environ["NWP_ECMWF_ZARR_PATH"]
            nwp_source_file_paths.append(nwp_ecmwf_source_file_path)
            nwp_paths.append(nwp_ecmwf_path)
        if "gfs" in nwp_keys:
            nwp_gfs_source_file_path = os.environ["NWP_GFS_ZARR_PATH"]
            nwp_source_file_paths.append(nwp_gfs_source_file_path)
            nwp_paths.append(nwp_gfs_path)
        if "mo_global" in nwp_keys:
            nwp_mo_global_source_file_path = os.environ["NWP_MO_GLOBAL_ZARR_PATH"]
            nwp_source_file_paths.append(nwp_mo_global_source_file_path)
            nwp_paths.append(nwp_mo_global_path)

        # Remove local cached zarr if already exists
        for nwp_source_file_path, nwp_path in zip(nwp_source_file_paths, nwp_paths, strict=False):
            # Process/cache remote zarr locally
            process_and_cache_nwp(nwp_source_file_path, nwp_path)
        if use_satellite and "satellite" in self.config["input_data"].keys():
            shutil.rmtree(satellite_path, ignore_errors=True)
            download_satellite_data(satellite_source_file_path)

        if self.asset_type == "wind":
            log.info("Preparing wind data sources")
            # Clear local cached wind data if already exists
            shutil.rmtree(wind_path, ignore_errors=True)
            os.mkdir(wind_path)

            # Save generation data as netcdf file
            generation_da = self.generation_data["data"].to_xarray()
            # Add the forecast timesteps to the generation, with 0 values
            forecast_timesteps = pd.date_range(
                start=self.t0 - pd.Timedelta("1H"), periods=197, freq="15min"
            )

            generation_da = generation_da.reindex(index=forecast_timesteps, fill_value=0.00001)

            # if generation_da is still empty make nans
            if len(generation_da) == 0:
                cols = [str(col) for col in self.generation_data["data"].columns]
                generation_df = pd.DataFrame(index=forecast_timesteps, columns=cols, data=0.0001)
                generation_da = generation_df.to_xarray()
            generation_da.to_netcdf(wind_netcdf_path, engine="h5netcdf")

            # Save metadata as csv
            self.generation_data["metadata"].to_csv(wind_metadata_path, index=False)

        if self.asset_type == "pv":
            log.info("Preparing PV data sources")
            # Clear local cached wind data if already exists
            shutil.rmtree(pv_path, ignore_errors=True)
            os.mkdir(pv_path)

            # Save generation data as netcdf file
            generation_da = self.generation_data["data"].to_xarray()
            # Add the forecast timesteps to the generation, with 0 values
            # TODO: Remove the hardcoding of delta time and the periods
            # Should be taken from config instead
            if self.client == "ruvnl":
                forecast_timesteps = pd.date_range(
                    start=self.t0 - pd.Timedelta("1H"), periods=197, freq="15min"
                )
            elif self.client == "ad":
                forecast_timesteps = pd.date_range(
                    start=self.t0 - pd.Timedelta("3H"), periods=46, freq="15min"
                )
            generation_da = generation_da.reindex(index=forecast_timesteps, fill_value=0.00001)

            # if generation_da is still empty make nans
            if len(generation_da) == 0:
                cols = [str(col) for col in self.generation_data["data"].columns]
                generation_df = pd.DataFrame(index=forecast_timesteps, columns=cols, data=0.0001)
                generation_da = generation_df.to_xarray()
            generation_da.to_netcdf(pv_netcdf_path, engine="h5netcdf")

            # Save metadata as csv
            self.generation_data["metadata"].to_csv(pv_metadata_path, index=False)

    def _create_dataloader(self):
        """Setup dataloader with prepared data sources"""

        log.info("Creating dataloader")

        # Pull the data config from huggingface

        data_config_filename = PVNetBaseModel.get_data_config(
            self.id, revision=self.version, token=self.hf_token
        )

        # Populate the data config with production data paths
        temp_dir = tempfile.TemporaryDirectory()
        populated_data_config_filename = f"{temp_dir.name}/data_config.yaml"

        self.config = populate_data_config_sources(
            data_config_filename, populated_data_config_filename
        )

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
                upsample_nwp=False,
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
        log.info(f"Loading model: {self.id} - {self.version} ({self.name})")

        return PVNetBaseModel.from_pretrained(
            model_id=self.id, revision=self.version, token=self.hf_token
        ).to(DEVICE)
