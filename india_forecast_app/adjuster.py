""" Adjuster code, adjust forecast by last 7 days of ME"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from pvsite_datamodel.read import get_site_by_uuid
from pvsite_datamodel.sqlmodels import ForecastSQL, ForecastValueSQL, GenerationSQL, MLModelSQL
from sqlalchemy import INT, cast, text
from sqlalchemy.sql import func

log = logging.getLogger(__name__)

# Wad to get all the forecast for the last 7 days made, at this time.
# And find the ME for each forecast horizon

"""
Here is the SQL query that it based off:

select 
AVG(forecast_values.forecast_power_kw - generation.generation_power_kw)
-- generation.generation_power_kw,
-- forecast_values.forecast_power_kw,
-- forecast_values.start_utc,
,horizon_minutes
-- *
from forecast_values 
JOIN forecasts ON forecasts.forecast_uuid = forecast_values.forecast_uuid 
join generation on generation.start_utc = forecast_values.start_utc
WHERE forecast_values.start_utc >= '2024-11-05' 
AND forecasts.site_uuid = 'adaf6be8-4e30-4c98-ac27-964447e9c8e6'
AND generation.site_uuid = 'adaf6be8-4e30-4c98-ac27-964447e9c8e6'
and extract(hour from forecasts.created_utc) = 7
group by horizon_minutes
-- order by forecast_values.start_utc

I've left some sql comments in, so its easier to remove the group by, when debugging. 
The site_uuid is hardcoded, but this should be updated.
"""


def get_me_values(
    session,
    hour: int,
    site_uuid: str,
    start_datetime: Optional[datetime] = None,
    ml_model_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Get the ME values for the last 7 days for a given hour, for a given hour creation time

    Args:
    hour: the hour of when the forecast is created
    site_uuid: the site this is for
    start_datetime:: the start datetime to filter on. This defaults to 7 days before now.
    session: sqlalchemy session
    ml_model_name: the name of the model to filter on, this is optional.
    """

    if start_datetime is None:
        start_datetime = datetime.now() - timedelta(days=7)

    query = session.query(
        func.avg(ForecastValueSQL.forecast_power_kw - GenerationSQL.generation_power_kw),
        # create hour column
        (cast(ForecastValueSQL.horizon_minutes / 60, INT)).label("horizon_hour"),
    )

    # join
    query = query.join(ForecastSQL)

    # round Generation start_utc and join to forecast start_utc
    start_utc_minute_rounded = (
        cast(func.date_part("minute", GenerationSQL.start_utc), INT)
        / 30
        * text("interval '30 min'")
    )
    start_utc_hour = func.date_trunc("hour", GenerationSQL.start_utc)
    generation_start_utc = start_utc_hour + start_utc_minute_rounded
    query = query.filter(generation_start_utc == ForecastValueSQL.start_utc)

    # only include the last x days
    query = query.filter(ForecastValueSQL.start_utc >= start_datetime)
    query = query.filter(GenerationSQL.start_utc >= start_datetime)

    # filter on site
    query = query.filter(ForecastSQL.site_uuid == site_uuid)
    query = query.filter(GenerationSQL.site_uuid == site_uuid)

    # filter on created_utc
    query = query.filter((func.extract("hour", ForecastSQL.created_utc) == hour))

    if ml_model_name is not None:
        query = query.join(MLModelSQL)
        query = query.filter(MLModelSQL.name == ml_model_name)

    # group by forecast horizon
    query = query.group_by('horizon_hour')

    # order by forecast horizon
    query = query.order_by('horizon_hour')

    me = query.all()

    me_df = pd.DataFrame(me, columns=["me_kw", "horizon_hour"])
    me_df["horizon_minutes"] = me_df["horizon_hour"] * 60

    # drop the hour column
    me_df.drop(columns=["horizon_hour"], inplace=True)

    if len(me_df) == 0:
        return me_df

    # interpolate horizon_minutes up to 15 minutes blocks, its currently in 60 minute blocks
    # currently in 0, 60, 120,...
    # change to 0, 15, 30, 45, 60, 75, 90, 105, 120, ...
    me_df = me_df.set_index("horizon_minutes")
    me_df = me_df.reindex(range(0, max(me_df.index), 15)).interpolate(limit=3)

    # reset indiex
    me_df = me_df.reset_index()

    # log the maximum and minimum adjuster results
    log.info(f"ME results: max={me_df['me_kw'].max()}, min={me_df['me_kw'].min()}")

    return me_df


def adjust_forecast_with_adjuster(
    db_session, forecast_meta: dict, forecast_values_df: pd.DataFrame, ml_model_name: str
):
    """
    Adjust forecast values with ME values

    Args:
    db_session: sqlalchemy session
    forecast_meta: forecast metadata
    forecast_values_df: forecast values dataframe
    ml_model_name: the ml model name

    """
    # get the ME values
    me_values = get_me_values(
        db_session,
        forecast_meta["timestamp_utc"].hour,
        site_uuid=forecast_meta["site_uuid"],
        ml_model_name=ml_model_name,
    )
    log.debug(f"ME values: {me_values}")

    # smooth results out, 1 hour each side
    # me_values["me_kw"] = me_values["me_kw"].rolling(window=5, min_periods=1, center=True).mean()

    # clip me values by 10% of the capacity
    site = get_site_by_uuid(db_session, forecast_meta["site_uuid"])
    capacity = site.capacity_kw
    me_kw_limit = 0.1 * capacity
    n_values_above_limit = (me_values["me_kw"] > me_kw_limit).sum()
    n_values_below_limit = (me_values["me_kw"] > me_kw_limit).sum()
    me_values["me_kw"].clip(lower=-0.1 * capacity, upper=0.1 * capacity, inplace=True)
    log.debug(
        f"ME values clipped: There were {n_values_above_limit} values above the limit and "
        f"{n_values_below_limit} values below the limit."
    )

    # join forecast_values_df with me_values on horizon_minutes
    forecast_values_df_adjust = forecast_values_df.copy()
    forecast_values_df_adjust = forecast_values_df_adjust.merge(
        me_values, on="horizon_minutes", how="left"
    )

    # if me_kw is null, set to 0
    forecast_values_df_adjust["me_kw"].fillna(0, inplace=True)

    # adjust forecast_power_kw by ME values
    forecast_values_df_adjust["forecast_power_kw"] = (
        forecast_values_df_adjust["forecast_power_kw"] - forecast_values_df_adjust["me_kw"]
    )
    # drop me_kw column
    forecast_values_df_adjust.drop(columns=["me_kw"], inplace=True)

    # clip negative values to 0
    forecast_values_df_adjust["forecast_power_kw"].clip(lower=0, inplace=True)
    return forecast_values_df_adjust
