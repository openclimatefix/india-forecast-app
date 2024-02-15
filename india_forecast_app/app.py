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
from pvsite_datamodel.read import get_sites_by_country
from pvsite_datamodel.sqlmodels import SiteSQL
from pvsite_datamodel.write import insert_forecast_values
from sqlalchemy.orm import Session

from .model import DummyModel

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


def get_model(asset_type: str):
    """
    Instantiates and returns the forecast model ready for running inference

    Args:
            asset_type: One or "pv" or "wind"

    Returns:
            A forecasting model
    """

    model = DummyModel(asset_type)
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
        # Ensure timestamp is UTC
        timestamp.replace(tzinfo=dt.UTC)
        
    # 0. Initialise DB connection
    url = os.environ["DB_URL"]

    db_conn = DatabaseConnection(url, echo=False)
    
    with db_conn.get_session() as session:

        # 1. Get sites
        log.info("Getting sites...")
        sites = get_sites(session)
        log.info(f"Found {len(sites)} sites")

        # 2. Load models
        log.info("Loading models...")
        pv_model = get_model("pv")
        log.info("Loaded PV model")
        wind_model = get_model("wind")
        log.info("Loaded wind model")

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
                        "version": model.version,
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
