"""Functions for retrieving and preparing generation data for forecasting."""

import asyncio
import datetime as dt
import logging
import os

import numpy as np
import ocf.dp as dp
import pandas as pd
import pvlib
from pvsite_datamodel import LocationSQL
from pvsite_datamodel.read import get_pv_generation_by_sites
from pvsite_datamodel.sqlmodels import LocationAssetType
from sqlalchemy.orm import Session

from india_forecast_app.save.data_platform import (
    fetch_dp_location_map,
    get_dataplatform_client,
)
from india_forecast_app.save.utils import ensure_timezone_aware

log = logging.getLogger(__name__)


def energy_source_for_asset_type(asset_type: str) -> dp.EnergySource:
    """Map an asset type ("pv"/"wind") to a Data Platform energy source."""
    return dp.EnergySource.WIND if asset_type == "wind" else dp.EnergySource.SOLAR


async def fetch_generation_from_dp(
    client_location_name: str,
    start: dt.datetime,
    end: dt.datetime,
    asset_type: str,
    observer_name: str | None = None,
) -> list[tuple[dt.datetime, float]]:
    """Fetch generation (observation) data from the Data Platform."""
    if not client_location_name:
        return []

    observer_name = observer_name or os.getenv("OBSERVER_NAME", "india")
    energy_source = energy_source_for_asset_type(asset_type)

    start_utc = ensure_timezone_aware(start)
    end_utc = ensure_timezone_aware(end)

    async with get_dataplatform_client() as client:
        location_map = await fetch_dp_location_map(client)
        target_uuid = location_map.get(client_location_name)
        if not target_uuid:
            log.warning(
                f"Location {client_location_name!r} not found in the Data Platform"
            )
            return []

        request = dp.GetObservationsAsTimeseriesRequest(
            location_uuid=target_uuid,
            observer_name=observer_name,
            energy_source=energy_source,
            time_window=dp.TimeWindow(
                start_timestamp_utc=start_utc,
                end_timestamp_utc=end_utc,
            ),
        )
        log.info(
            f"Reading generation from Data Platform for {client_location_name!r} "
            f"(uuid={target_uuid}, observer={observer_name!r}, "
            f"energy_source={energy_source.name}) from {start_utc} to {end_utc}",
        )
        try:
            response = await client.get_observations_as_timeseries(request)
        except Exception as e:
            log.error(f"Failed to fetch observations for {client_location_name!r}: {e}")
            return []

    if not response.values:
        return []

    cap_w = response.values[0].effective_capacity_watts
    data = [
        (v.timestamp_utc, v.value_fraction * cap_w / 1000.0)
        for v in response.values
    ]
    log.info(
        f"Fetched {len(data)} generation value(s) from the Data Platform "
        f"for {client_location_name!r}",
    )
    return data


def get_generation_data(
    db_session: Session, sites: list[LocationSQL], timestamp: dt.datetime
) -> dict[str, pd.DataFrame]:
    """
    Gets generation data values for given sites

    Args:
            db_session: A SQLAlchemy session
            sites: A list of LocationSQL objects
            timestamp: The end time from which to retrieve data

    Returns:
            A Dict containing:
            - "data": Dataframe containing 15-minutely generation data
            - "metadata": Dataframe containing information about the site
    """

    site_uuids = [s.location_uuid for s in sites]
    # TODO change this from  hardcoded to site and config related variable
    client = os.getenv("CLIENT_NAME", "ruvnl")
    if client == "ruvnl":
        start = timestamp - dt.timedelta(hours=1)
    elif client == "ad":
        start = timestamp - dt.timedelta(hours=25)
    # pad by 1 second to ensure get_pv_generation_by_sites returns correct data
    end = timestamp + dt.timedelta(seconds=1)

    # get the ml id, this only works for one site right now
    system_id = sites[0].ml_id

    read_from_data_platform = os.getenv("READ_FROM_DATA_PLATFORM", "false").lower() == "true"
    if read_from_data_platform:
        asset_type = "pv" if sites[0].asset_type == LocationAssetType.pv else "wind"
        log.info(
            f"Reading generation from Data Platform for {sites[0].client_location_name!r} "
            f"from {start} to {end}",
        )
        dp_data = asyncio.run(
            fetch_generation_from_dp(
                client_location_name=sites[0].client_location_name,
                start=start,
                end=end,
                asset_type=asset_type,
            ),
        )
        formatted_data = [(t, p, system_id) for t, p in dp_data]
    else:
        log.info(f"Getting generation data for sites: {site_uuids}, from {start=} to {end=}")
        generation_data = get_pv_generation_by_sites(
            session=db_session, site_uuids=site_uuids, start_utc=start, end_utc=end
        )
        formatted_data = [
            (g.start_utc, g.generation_power_kw, system_id) for g in generation_data
        ]

    if len(formatted_data) == 0:
        log.warning("No generation found for the specified sites/period")
        generation_df = pd.DataFrame(columns=[str(system_id)])

    else:
        # Convert to dataframe
        generation_df = pd.DataFrame(
            formatted_data,
            columns=["time_utc", "power_kw", "ml_id"],
        ).pivot(index="time_utc", columns="ml_id", values="power_kw")

        if generation_df.index.tz is not None:
            generation_df.index = generation_df.index.tz_convert("UTC").tz_localize(None)

        log.info(generation_df)

        # Filter out any 0 values when the sun is up
        if sites[0].asset_type == LocationAssetType.pv:
            generation_df = filter_on_sun_elevation(generation_df, sites[0])

        # Ensure timestamps line up with 3min intervals
        generation_df.index = generation_df.index.round("3min")

        # Drop any duplicated timestamps
        generation_df = generation_df[~generation_df.index.duplicated()]

        # xarray (used later) expects columns with string names
        generation_df.columns = generation_df.columns.astype(str)

        # Handle any missing timestamps
        contiguous_dt_idx = pd.date_range(start=start, end=end, freq="3min")[:-1]
        generation_df = generation_df.reindex(contiguous_dt_idx, fill_value=None)

        # Interpolate NaNs
        generation_df = generation_df.interpolate(method="linear", limit_direction="both")

        # Down-sample from 3 min to 15 min intervals
        generation_df = generation_df.resample("15min").mean()

        # Add a final row for t0, and interpolate this row
        generation_df.loc[timestamp] = np.nan
        generation_df = generation_df.interpolate(method="quadratic", fill_value="extrapolate")

        # convert to watts,
        # as this is current what ocf_datapipes expects
        # This is because we normalize by the watts amount
        col = generation_df.columns[0]
        generation_df[col] = generation_df[col].astype(float) * 1e3

    # Site metadata dataframe
    sites_df = pd.DataFrame(
        [
            (system_id, s.latitude, s.longitude, s.capacity_kw / 1000.0, s.capacity_kw * 1000)
            for s in sites
        ],
        columns=["system_id", "latitude", "longitude", "capacity_megawatts", "capacity_watts"],
    )

    return {"data": generation_df, "metadata": sites_df}


def filter_on_sun_elevation(generation_df, site) -> pd.DataFrame:
    """Filter the data on sun elevation

    If the sun is up, the generation values should be above zero
    param:
        generation_df: A dataframe containing generation data,
            with a column "power_kw", and index of datetimes
        site: A LocationSQL object

    return: dataframe with generation data
    """
    # using pvlib, calculate the sun elevations
    solpos = pvlib.solarposition.get_solarposition(
        time=generation_df.index,
        longitude=site.longitude,
        latitude=site.latitude,
        method="nrel_numpy",
    )
    elevation = solpos["elevation"].values

    # find the values that are <=0 and elevation >5
    mask = (elevation > 5) & (generation_df[generation_df.columns[0]] <= 0)

    dropping_datetimes = generation_df.index[mask]
    if len(dropping_datetimes) > 0:
        log.warning(
            f"Will be dropping {len(dropping_datetimes)} rows "
            f"from generation data: {dropping_datetimes.values} "
            f"due to sun elevation > 5 degrees and generation <= 0.0 kW. "
            f"This is likely due error in the generation data)"
        )

    generation_df = generation_df[~mask]
    return generation_df
