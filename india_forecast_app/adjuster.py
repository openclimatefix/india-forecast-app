from pvsite_datamodel.sqlmodels import ForecastValueSQL, GenerationSQL, ForecastSQL
from sqlalchemy.sql import func
from sqlalchemy import cast, INT, text

from datetime import datetime, timedelta
import pandas as pd
from typing import Optional

# Wad to get all the forecast for the last 7 days made, at this time.
# And find the ME for each forecast horizon

"""
Here is the SQL query that it based off

select 
AVG(generation.generation_power_kw - forecast_values.forecast_power_kw)
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
"""


def get_me_values(
    session, hour: int, site_uuid: str, start_datetime: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Get the ME values for the last 7 days for a given hour, for a given hour creation time

    args:
    hour: the hour of whent he forecast is created
    start_datetime:: the start datetime to filter on.
    session: sqlalchemy session

    """

    if start_datetime is None:
        start_datetime = datetime.now() - timedelta(days=7)

    query = session.query(
        func.avg(ForecastValueSQL.forecast_power_kw - GenerationSQL.generation_power_kw),
        ForecastValueSQL.horizon_minutes,
    )

    # join
    query = query.join(ForecastSQL)

    # round Generation start_utc and join to forecast start_utc
    start_utc_minute_rounded = (
        cast(func.date_part("minute", GenerationSQL.start_utc), INT)
        / 15
        * text("interval '15 min'")
    )
    start_utc_hour = func.date_trunc("hour", GenerationSQL.start_utc)
    generation_start_utc = start_utc_hour + start_utc_minute_rounded
    query = query.filter(generation_start_utc == ForecastValueSQL.start_utc)

    # only include the last 7 days
    query = query.filter(ForecastValueSQL.start_utc >= start_datetime)
    query = query.filter(GenerationSQL.start_utc >= start_datetime)

    # filter on site
    query = query.filter(ForecastSQL.site_uuid == site_uuid)
    query = query.filter(GenerationSQL.site_uuid == site_uuid)

    # filter on created_utc
    query = query.filter((func.extract("hour", ForecastSQL.created_utc) == hour))

    # group by forecast horizon
    query = query.group_by(ForecastValueSQL.horizon_minutes)

    # order by forecast horizon
    query = query.order_by(ForecastValueSQL.horizon_minutes)

    me = query.all()

    me_df = pd.DataFrame(me, columns=["me_kw", "horizon_minutes"])

    return me_df
