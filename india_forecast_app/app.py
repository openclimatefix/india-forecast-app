"""
Main forecast app entrypoint
"""

import datetime as dt
import logging
import os
import sys
from typing import Optional

import click
import pandas as pd
import sentry_sdk
from pvsite_datamodel import DatabaseConnection
from pvsite_datamodel.read import get_sites_by_country
from pvsite_datamodel.sqlmodels import LocationAssetType, LocationSQL
from pvsite_datamodel.write import insert_forecast_values
from sqlalchemy.orm import Session

import india_forecast_app
from india_forecast_app.adjuster import adjust_forecast_with_adjuster
from india_forecast_app.data.generation import get_generation_data
from india_forecast_app.models import PVNetModel, get_all_models
from india_forecast_app.sentry import traces_sampler

log = logging.getLogger(__name__)
version = india_forecast_app.__version__


sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    environment=os.getenv("ENVIRONMENT", "local"),
    traces_sampler=traces_sampler,
)


def get_sites(db_session: Session) -> list[LocationSQL]:
    """
    Gets all available sites in India

    Args:
            db_session: A SQLAlchemy session

    Returns:
            A list of LocationSQL objects
    """

    client = os.getenv("CLIENT_NAME", "ruvnl")
    log.info(f"Getting sites for client: {client}")

    sites = get_sites_by_country(db_session, country="india", client_name=client)

    log.info(f"Found {len(sites)} sites for {client} in India")
    return sites


def get_model(
    asset_type: str,
    timestamp: dt.datetime,
    generation_data,
    hf_repo: str,
    hf_version: str,
    name: str,
) -> PVNetModel:
    """
    Instantiates and returns the forecast model ready for running inference

    Args:
            asset_type: One or "pv" or "wind"
            timestamp: Datetime at which the forecast will be made
            generation_data: Latest historic generation data
            hf_repo: ID of the ML model used for the forecast
            hf_version: Version of the ML model used for the forecast
            name: Name of the ML model used for the forecast

    Returns:
            A forecasting model
    """

    # Only Windnet and PVnet is now used
    model_cls = PVNetModel

    model = model_cls(
        asset_type, timestamp, generation_data, hf_repo=hf_repo, hf_version=hf_version, name=name
    )
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


def save_forecast(
    db_session: Session,
    forecast,
    write_to_db: bool,
    ml_model_name: Optional[str] = None,
    ml_model_version: Optional[str] = None,
    use_adjuster: bool = True,
    adjuster_average_minutes: Optional[int] = 60,
):
    """
    Saves a forecast for a given site & timestamp

    Args:
            db_session: A SQLAlchemy session
            forecast: a forecast dict containing forecast meta and predicted values
            write_to_db: If true, forecast values are written to db, otherwise to stdout
            ml_model_name: Name of the ML model used for the forecast
            ml_model_version: Version of the ML model used for the forecast
            use_adjuster: Make new model, adjusted by last 7 days of ME values
            adjuster_average_minutes: The number of minutes that results are average over
                when calculating adjuster values

    Raises:
            IOError: An error if database save fails
    """

    forecast_meta = {
        "location_uuid": forecast["meta"]["site_id"],
        "timestamp_utc": forecast["meta"]["timestamp"],
        "forecast_version": forecast["meta"]["version"],
    }
    forecast_values_df = pd.DataFrame(forecast["values"])
    forecast_values_df["horizon_minutes"] = (
        (forecast_values_df["start_utc"] - forecast_meta["timestamp_utc"]) / pd.Timedelta("60s")
    ).astype("int")

    if write_to_db:
        insert_forecast_values(
            db_session,
            forecast_meta,
            forecast_values_df,
            ml_model_name=ml_model_name,
            ml_model_version=ml_model_version,
        )

    if use_adjuster:
        log.info(f"Adjusting forecast for site_id={forecast_meta['location_uuid']}...")

        forecast_values_df_adjust = adjust_forecast_with_adjuster(
            db_session,
            forecast_meta,
            forecast_values_df,
            ml_model_name=ml_model_name,
            average_minutes=adjuster_average_minutes,
        )

        log.info(forecast_values_df_adjust)

        if write_to_db:
            insert_forecast_values(
                db_session,
                forecast_meta,
                forecast_values_df_adjust,
                ml_model_name=f"{ml_model_name}_adjust",
                ml_model_version=ml_model_version,
            )

    output = f'Forecast for site_id={forecast_meta["location_uuid"]},\
               timestamp={forecast_meta["timestamp_utc"]},\
               version={forecast_meta["forecast_version"]}:'
    log.info(output.replace("  ", ""))
    log.info(f"\n{forecast_values_df.to_string()}\n")


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
    Main click function for running forecasts for sites in India
    """

    app_run(timestamp=timestamp, write_to_db=write_to_db, log_level=log_level)


def app_run(timestamp: dt.datetime | None, write_to_db: bool = False, log_level: str = "info"):
    """
    Main function for running forecasts for sites in India
    """
    logging.basicConfig(stream=sys.stdout, level=getattr(logging, log_level.upper()))

    log.info(f"Running India forecast app:{version}")

    if timestamp is None:
        # get the timestamp now rounded down the nearest 15 minutes
        # TODO better to have explicity UTC time here?
        timestamp = pd.Timestamp.now(tz="UTC").replace(tzinfo=None).floor("15min")
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

        pv_sites = [site for site in sites if site.asset_type == LocationAssetType.pv]
        log.info(f"Found {len(pv_sites)} pv sites")
        wind_sites = [site for site in sites if site.asset_type == LocationAssetType.wind]
        log.info(f"Found {len(wind_sites)} wind sites")

        # 2. Load data/models
        all_model_configs = get_all_models(client_abbreviation=os.getenv("CLIENT_NAME", "ruvnl"))
        successful_runs = 0
        runs = 0
        for model_config in all_model_configs.models:

            asset_sites = pv_sites if model_config.asset_type == "pv" else wind_sites
            asset_type = model_config.asset_type

            for site in asset_sites:
                runs += 1

                log.info(f"Reading latest historic {asset_type} generation data...")
                generation_data = get_generation_data(session, [site], timestamp)

                if asset_type == "wind":
                    # change from W to MW
                    generation_data["data"] = generation_data["data"] / 1e6

                log.debug(f"{generation_data['data']=}")
                log.debug(f"{generation_data['metadata']=}")

                log.info(f"Loading {asset_type} model {model_config.name}...")
                ml_model = get_model(
                    asset_type,
                    timestamp,
                    generation_data,
                    hf_repo=model_config.id,
                    hf_version=model_config.version,
                    name=model_config.name,
                )
                ml_model.location_uuid = site.location_uuid

                log.info(f"{asset_type} model loaded")

                # 3. Run model for each site
                site_id = ml_model.location_uuid
                asset_type = ml_model.asset_type
                log.info(f"Running {asset_type} model for site={site_id}...")
                forecast_values = run_model(model=ml_model, site_id=site_id, timestamp=timestamp)

                if forecast_values is None:
                    log.info(f"No forecast values for site_id={site_id}")
                else:
                    # 4. Write forecast to DB or stdout
                    log.info(f"Writing forecast for site_id={site_id}")
                    forecast = {
                        "meta": {
                            "site_id": site_id,
                            "version": version,
                            "timestamp": timestamp,
                        },
                        "values": forecast_values,
                    }
                    save_forecast(
                        session,
                        forecast=forecast,
                        write_to_db=write_to_db,
                        ml_model_name=ml_model.name,
                        ml_model_version=version,
                        adjuster_average_minutes=model_config.adjuster_average_minutes,
                    )
                    successful_runs += 1

        log.info(
            f"Completed forecasts for {successful_runs} runs for "
            f"{runs} model runs. This was for {len(sites)} sites"
        )
        if successful_runs == runs:
            log.info("All forecasts completed successfully")
        elif 0 < successful_runs < runs:
            raise Exception("Some forecasts failed")
        else:
            raise Exception("All forecasts failed")

        log.info("Forecast finished")


if __name__ == "__main__":
    app()
