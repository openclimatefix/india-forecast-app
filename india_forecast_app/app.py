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
from pvsite_datamodel.read import get_sites_by_country, get_pv_generation_by_sites
from pvsite_datamodel.sqlmodels import SiteSQL, GenerationSQL, SiteAssetType
from pvsite_datamodel.write import insert_forecast_values
from sqlalchemy.orm import Session

from .models import DummyModel, PVNetModel

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
    return sites


def get_generation_data(db_session: Session, sites: list[SiteSQL], timestamp: dt.datetime) -> list[GenerationSQL]:
    """
        Gets generation data values for given sites

        Args:
                db_session: A SQLAlchemy session
                sites: A list of SiteSQL objects
                timestamp: The time from which to get generation data for

        Returns:
                A list of SiteSQL objects
    """

    site_uuids = [s.site_uuid for s in sites]
    start = timestamp - dt.timedelta(hours=1)
    end = timestamp

    generation_data = get_pv_generation_by_sites(
        session=db_session, site_uuids=site_uuids, start_utc=start, end_utc=end
    )

    # TODO resample data to 15 min intervals - ensure 5 values for wind
    # Expect that some values may be missing

    return generation_data


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
        log.info(
            f'site_id={forecast_meta["site_uuid"]}, \
            timestamp={forecast_meta["timestamp_utc"]}, \
            version={forecast_meta["forecast_version"]}, \
            forecast values={forecast_values_df.to_string()}'
        )


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
    logging.basicConfig(stream=sys.stdout, level=getattr(logging, log_level.upper()))

    if timestamp is None:
        # get the timestamp now rounded down the neartes 15 minutes
        timestamp = pd.Timestamp.now(tz="UTC").floor("15min")
        log.info('Timestamp omitted - will generate forecasts for "now"')
    else:
        timestamp = pd.Timestamp(timestamp).floor("15min")
        
    # 0. Initialise DB connection
    url = os.environ["DB_URL"]

    db_conn = DatabaseConnection(url, echo=False)
    
    with db_conn.get_session() as session:

        # 1. Get sites
        log.info("Getting sites...")
        sites = get_sites(session)
        log.info(sites)
        pv_sites = [site for site in sites if site.asset_type == SiteAssetType.wind]
        log.info(f"Found {len(pv_sites)} pv sites")
        wind_sites = [site for site in sites if site.asset_type == SiteAssetType.wind]
        log.info(f"Found {len(wind_sites)} wind sites")

        # 2. Load data/models
        if len(pv_sites) > 0:
            # TODO get gen data -> pass to get_model
            log.info("Loading PV model...")
            pv_model = get_model("pv", timestamp, generation_data=[])
            log.info("PV model loaded")

        if len(wind_sites) > 0:
            log.info("Reading latest historic wind generation data...")
            generation_data = get_generation_data(session, wind_sites, timestamp)
            log.info("Loading wind model...")
            wind_model = get_model("wind", timestamp, generation_data)
            log.info("Wind model loaded")

        for site in sites:
            # 3. Run model for each site
            site_id = site.site_uuid
            asset_type = site.asset_type.name
            log.info(f"Running {asset_type} model for site={site_id}...")
            model = wind_model if asset_type == "wind" else pv_model
            forecast_values = run_model(model=model, site_id=site_id, timestamp=timestamp)

            if forecast_values is None:
                log.info(f"No forecast values for site_id={site_id}")
            else:
                # 4. Write forecast to DB or stdout
                log.info(f"Writing forecast for site_id={site_id}")
                forecast = {
                    "meta": {
                        "site_id": site_id,
                        # TODO model version strings too long to store db field (max 32 chars)
                        # "version": model.version,
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
