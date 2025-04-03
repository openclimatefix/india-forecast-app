import datetime as dt
import os
import logging

import numpy as np
import pandas as pd
import pvlib
from pvsite_datamodel import SiteSQL
from pvsite_datamodel.read import get_pv_generation_by_sites
from pvsite_datamodel.sqlmodels import SiteAssetType
from sqlalchemy.orm import Session


log = logging.getLogger(__name__)


def get_generation_data(
    db_session: Session, sites: list[SiteSQL], timestamp: dt.datetime
) -> dict[str, pd.DataFrame]:
    """
    Gets generation data values for given sites

    Args:
            db_session: A SQLAlchemy session
            sites: A list of SiteSQL objects
            timestamp: The end time from which to retrieve data

    Returns:
            A Dict containing:
            - "data": Dataframe containing 15-minutely generation data
            - "metadata": Dataframe containing information about the site
    """

    site_uuids = [s.site_uuid for s in sites]
    # TODO change this from  hardcoded to site and config related variable
    client = os.getenv("CLIENT_NAME", "ruvnl")
    if client == "ruvnl":
        start = timestamp - dt.timedelta(hours=1)
    elif client == "ad":
        start = timestamp - dt.timedelta(hours=25)
    # pad by 1 second to ensure get_pv_generation_by_sites returns correct data
    end = timestamp + dt.timedelta(seconds=1)

    log.info(f"Getting generation data for sites: {site_uuids}, from {start=} to {end=}")
    generation_data = get_pv_generation_by_sites(
        session=db_session, site_uuids=site_uuids, start_utc=start, end_utc=end
    )
    # get the ml id, this only works for one site right now
    system_id = sites[0].ml_id

    if len(generation_data) == 0:
        log.warning("No generation found for the specified sites/period")
        generation_df = pd.DataFrame(columns=[str(system_id)])

    else:
        # Convert to dataframe
        generation_df = pd.DataFrame(
            [(g.start_utc, g.generation_power_kw, system_id) for g in generation_data],
            columns=["time_utc", "power_kw", "ml_id"],
        ).pivot(index="time_utc", columns="ml_id", values="power_kw")

        log.info(generation_df)

        # Filter out any 0 values when the sun is up
        if sites[0].asset_type == SiteAssetType.pv:
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
        generation_df: A dataframe containing generation data, columns of "time_utc" and "power_kw"
        site: A SiteSQL object

    return: dataframe with generation data
    """
    # using pvlib, calculate the sun elevations
    solpos = pvlib.solarposition.get_solarposition(
        time=generation_df["time_utc"],
        longitude=site.longitude,
        latitude=site.latitude,
        method="nrel_numpy",
    )
    elevation = solpos["elevation"].values

    # find the values that are <=0 and elevation >5
    mask = (elevation > 5) & (generation_df['power_kw'] <= 0)

    dropping_datetimes = generation_df.index[mask]
    if len(dropping_datetimes) > 0:
        log.warning(
            f"Will be dropping {len(dropping_datetimes)} rows "
            f"from generation data: {dropping_datetimes} "
            f"due to sun elevation > 5 degrees and generation <= 0.0 kW. "
            f"This is likely due error in the generation data)"
        )

    generation_df = generation_df[~mask]
    return generation_df
