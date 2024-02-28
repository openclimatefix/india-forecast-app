"""
Main forecast app entrypoint
"""

import datetime as dt
import logging
import os
import sys

import click
import pandas as pd
from pvsite_datamodel import DatabaseConnection
from pvsite_datamodel.read import get_pv_generation_by_sites, get_sites_by_country
from pvsite_datamodel.sqlmodels import SiteAssetType, SiteSQL
from pvsite_datamodel.write import insert_forecast_values
from sqlalchemy.orm import Session

from india_forecast_app.models import DummyModel, PVNetModel

log = logging.getLogger(__name__)


def get_sites(db_session: Session) -> list[SiteSQL]:
    """
    Gets all available sites in India

    Args:
            db_session: A SQLAlchemy session

    Returns:
            A list of SiteSQL objects
    """
    
    sites = get_sites_by_country(db_session, country="india")
    return [sites[1]]


def get_generation_data(
        db_session: Session,
        sites: list[SiteSQL],
        timestamp: dt.datetime
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

    start = timestamp - dt.timedelta(hours=1)
    # pad by 1 second to ensure get_pv_generation_by_sites returns correct data
    end = timestamp + dt.timedelta(seconds=1)

    log.info(f'Getting generation data for sites: {site_uuids}, from {start=} to {end=}')
    generation_data = get_pv_generation_by_sites(
        session=db_session, site_uuids=site_uuids, start_utc=start, end_utc=end
    )

    # Convert to dataframe
    generation_df = (pd.DataFrame(
        [(g.start_utc, g.generation_power_kw, g.site.ml_id) for g in generation_data],
        columns=["time_utc", "power_kw", "ml_id"]
    ).pivot(index="time_utc", columns="ml_id", values="power_kw"))

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

    # Add a final row for t0, set to the mean of the previous values
    generation_df.loc[timestamp] = generation_df.mean()

    # Site metadata dataframe
    sites_df = pd.DataFrame(
        [(s.ml_id, s.latitude, s.longitude, s.capacity_kw/1000.0) for s in sites],
        columns=["system_id", "latitude", "longitude", "capacity_megawatts"]
    )

    return {
        "data": generation_df,
        "metadata": sites_df
    }


def get_model(asset_type: str, timestamp: dt.datetime, generation_data) -> PVNetModel:
    """
    Instantiates and returns the forecast model ready for running inference

    Args:
            asset_type: One or "pv" or "wind"
            timestamp: Datetime at which the forecast will be made
            generation_data: Latest historic generation data

    Returns:
            A forecasting model
    """

    # Only windnet is ready, so if asset_type is PV, continue using dummy model
    if asset_type == "wind":
        model_cls = PVNetModel
    else:
        model_cls = DummyModel

    model = model_cls(asset_type, timestamp, generation_data)
    return model


def run_model(model, site_id: str, timestamp: dt.datetime):
    """
    Runs inference on model for the given site & timestamp

    Args:
            model: A forecasting model
            site_id: A specific site ID
            timestamp: timestamp to run a forecast for

    Returns:
            A forecast or None if model inference fails
    """

    try:
        forecast = model.predict(site_id=site_id, timestamp=timestamp)
    except Exception:
        log.error(
            f"Error while running model.predict for site_id={site_id}. Skipping",
            exc_info=True,
        )
        return None

    return forecast


def save_forecast(db_session: Session, forecast, write_to_db: bool):
    """
    Saves a forecast for a given site & timestamp

    Args:
            db_session: A SQLAlchemy session
            forecast: a forecast dict containing forecast meta and predicted values
            write_to_db: If true, forecast values are written to db, otherwise to stdout

    Raises:
            IOError: An error if database save fails
    """

    forecast_meta = {
        "site_uuid": forecast["meta"]["site_id"],
        "timestamp_utc": forecast["meta"]["timestamp"],
        "forecast_version": forecast["meta"]["version"],
    }
    forecast_values_df = pd.DataFrame(forecast["values"])
    forecast_values_df["horizon_minutes"] = (
        (forecast_values_df["start_utc"] - forecast_meta["timestamp_utc"])
        / pd.Timedelta("60s")
    ).astype("int")

    if write_to_db:
        insert_forecast_values(db_session, forecast_meta, forecast_values_df)
    else:
        output = f'Forecast for site_id={forecast_meta["site_uuid"]},\
                   timestamp={forecast_meta["timestamp_utc"]},\
                   version={forecast_meta["forecast_version"]}:'
        log.info(output.replace('  ', ''))
        log.info(f'\n{forecast_values_df.to_string()}')


@click.command()
@click.option(
    "--date",
    "-d",
    "timestamp",
    type=click.DateTime(formats=["%Y-%m-%d-%H-%M"]),
    default=None,
    help='Date-time (UTC) at which we make the prediction. \
Format should be YYYY-MM-DD-HH-mm. Defaults to "now".',
)
@click.option(
    "--write-to-db",
    is_flag=True,
    default=False,
    help="Set this flag to actually write the results to the database.",
)
@click.option(
    "--log-level",
    default="info",
    help="Set the python logging log level",
    show_default=True,
)
def app(timestamp: dt.datetime | None, write_to_db: bool, log_level: str):
    """
    Main function for running forecasts for sites in India
    """
    logging.basicConfig(stream=sys.stdout, level=getattr(logging, log_level.upper()), force=True)

    if timestamp is None:
        # get the timestamp now rounded down the nearest 15 minutes
        timestamp = pd.Timestamp.now(tz=None).floor("15min")
        log.info(f'Timestamp omitted - will generate forecasts for "now" ({timestamp})')
    else:
        timestamp = pd.Timestamp(timestamp).floor("15min")
        
    # 0. Initialise DB connection
    url = os.environ["DB_URL"]

    db_conn = DatabaseConnection(url, echo=False)
    
    with db_conn.get_session() as session:

        # 1. Get sites
        log.info("Getting sites...")
        sites = get_sites(session)

        pv_sites = [site for site in sites if site.asset_type == SiteAssetType.wind]
        log.info(f"Found {len(pv_sites)} pv sites")
        wind_sites = [site for site in sites if site.asset_type == SiteAssetType.wind]
        log.info(f"Found {len(wind_sites)} wind sites")

        # 2. Load data/models
        models = {}
        for asset_type in ["pv", "wind"]:
            asset_sites = pv_sites if asset_type == "pv" else wind_sites
            if len(asset_sites) > 0:
                log.info(f"Reading latest historic {asset_type} generation data...")
                if asset_type == "wind":
                    generation_data = get_generation_data(session, asset_sites, timestamp)
                else:
                    generation_data = {"data": pd.DataFrame(), "metadata": pd.DataFrame()}
                log.info(f"{generation_data['data']=}")
                log.info(f"{generation_data['metadata']=}")
                log.info(f"Loading {asset_type} model...")
                models[asset_type] = get_model(asset_type, timestamp, generation_data)
                log.info(f"{asset_type} model loaded")

        for site in sites:
            # 3. Run model for each site
            site_id = site.site_uuid
            asset_type = site.asset_type.name
            log.info(f"Running {asset_type} model for site={site_id}...")
            forecast_values = run_model(
                model=models[asset_type],
                site_id=site_id,
                timestamp=timestamp
            )

            if forecast_values is None:
                log.info(f"No forecast values for site_id={site_id}")
            else:
                # 4. Write forecast to DB or stdout
                log.info(f"Writing forecast for site_id={site_id}")
                forecast = {
                    "meta": {
                        "site_id": site_id,
                        # TODO model version strings too long to store db field (max 32 chars)
                        # "version": models[asset_type].version,
                        "version": "0.0.0",
                        "timestamp": timestamp,
                    },
                    "values": forecast_values,
                }
                save_forecast(
                    session,
                    forecast=forecast,
                    write_to_db=write_to_db,
                )


if __name__ == "__main__":
    app()
