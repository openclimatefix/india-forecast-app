import datetime as dt
import logging

import click

from .model import DummyModel

log = logging.getLogger(__name__)


def _get_site_ids() -> list[str]:
    """
    Gets all avaiable site_ids in India

    Returns:
            A list of site_ids
    """

    return [
        "b0579f31-70d9-4682-962e-4e2b30fa1e85",
        "d0146492-90d2-41bf-9e44-153032492bad",
    ]


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


def _save_forecast(site_id: str, timestamp: dt.datetime, forecast, write_to_db: bool):
    """
    Saves a forecast for a given site & timestamp

    Args:
            site_id: A specific site ID
            timestamp: timestamp to run a forecast for
            forecast: a forecast containing predicted generation values for the given site
            write_to_db: If true, forecast values are written to db, otherwise to stdout

    Raises:
            IOError: An error if database save fails
    """

    if write_to_db:
        pass
    else:
        log.info(
            f"site_id={site_id}, timestamp={timestamp}, forecast values={forecast}"
        )


@click.command()
@click.option(
    "--date",
    "-d",
    "timestamp",
    type=click.DateTime(formats=["%Y-%m-%d-%H-%M"]),
    default=None,
    help='Date-time (UTC) at which we make the prediction. Defaults to "now".',
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
        timestamp = dt.datetime.utcnow()
        log.info('Timestamp omitted - will generate forecasts for "now"')

    # 1. Get sites
    log.info("Getting sites")
    site_ids = _get_site_ids()

    # 2. Load model
    log.info("Loading model")
    model = _get_model()

    # 3. Run model for each site
    log.info("Running model for each site")
    for site_id in site_ids:
        forecast = _run_model(model=model, site_id=site_id, timestamp=timestamp)

        if forecast is not None:
            # 4. Write forecast to DB or stdout
            log.info(f"Writing forecast for site_id={site_id}")
            _save_forecast(
                site_id=site_id,
                timestamp=timestamp,
                forecast=forecast,
                write_to_db=write_to_db,
            )


if __name__ == "__main__":
    app()
