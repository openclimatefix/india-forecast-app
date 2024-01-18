import datetime as dt
import logging
import os

import click
import pandas as pd
from pvsite_datamodel import DatabaseConnection
from pvsite_datamodel.read import get_sites_by_country
from pvsite_datamodel.write import insert_forecast_values

from .model import DummyModel

log = logging.getLogger(__name__)

url = os.environ["DB_URL"]
db_conn = DatabaseConnection(url, echo=False)


def _get_site_ids() -> list[str]:
    """
    Gets all avaiable site_ids in India

    Returns:
            A list of site_ids
    """

    with db_conn.get_session() as session:
        sites = get_sites_by_country(session, country="india")

        return [s.site_uuid for s in sites]


def _get_model():
    """
    Instantiates and returns the forecast model ready for running inference

    Returns:
            A forecasting model
    """

    model = DummyModel()
    return model


def _run_model(model, site_id: str, timestamp: dt.datetime):
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


def _save_forecast(forecast, write_to_db: bool):
    """
    Saves a forecast for a given site & timestamp

    Args:
            site_id: A specific site ID
            timestamp: timestamp to run a forecast for
            forecast: a forecast containing predicted generation
            values for the given site
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
        with db_conn.get_session() as session:
            insert_forecast_values(session, forecast_meta, forecast_values_df)

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
    help='Date-time (UTC) at which we make the prediction Format should be YYYY-MM-DD-HH-mm. Defaults to "now".',
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
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    if timestamp is None:
        timestamp = dt.datetime.now(tz=dt.UTC)
        log.info('Timestamp omitted - will generate forecasts for "now"')
    else:
        # Ensure timestamp is UTC
        timestamp.replace(tzinfo=dt.UTC)

    # 1. Get sites
    log.info("Getting sites")
    site_ids = _get_site_ids()

    # 2. Load model
    log.info("Loading model")
    model = _get_model()

    # 3. Run model for each site
    for site_id in site_ids:
        log.info(f"Running model for site={site_id}")
        forecast_values = _run_model(model=model, site_id=site_id, timestamp=timestamp)

        if forecast_values is not None:
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
            _save_forecast(
                forecast=forecast,
                write_to_db=write_to_db,
            )


if __name__ == "__main__":
    app()
