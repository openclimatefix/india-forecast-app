"""Orchestrates saving forecasts to the database and/or Data Platform."""

from __future__ import annotations

import asyncio
import logging
import os

import pandas as pd
from sqlalchemy.orm import Session  # noqa: TC002

from india_forecast_app.save.data_platform import save_to_dataplatform
from india_forecast_app.save.database import write_forecast_to_db

log = logging.getLogger(__name__)


def save_forecast(
    db_session: Session,
    forecast: dict,
    write_to_db: bool = False,
    ml_model_name: str | None = None,
    ml_model_version: str | None = None,
    use_adjuster_database: bool = True,
    adjuster_average_minutes: int | None = 60,
    location_map: dict[str, str] | None = None,
) -> None:
    """Save a forecast for a given site & timestamp.

    Args:
        db_session: A SQLAlchemy session
        forecast: A forecast dict containing forecast meta and predicted values
        write_to_db: If true, forecast values are written to the DB, otherwise to stdout
        ml_model_name: Name of the ML model used for the forecast
        ml_model_version: Version of the ML model used for the forecast
        use_adjuster_database: Make a new model adjusted by last 7 days of ME values.
            Also controls whether an adjusted forecast is sent to the Data Platform.
        adjuster_average_minutes: Minutes to average over when calculating adjuster values
        location_map: Optional pre-fetched mapping of DP location name to UUID.
            When provided, avoids a list_locations gRPC call per site.

    Raises:
        IOError: An error if the database save fails
    """
    log.info(f"Saving forecast for location_id={forecast['meta']['site_id']}...")

    forecast_meta = {
        "location_uuid": forecast["meta"]["site_id"],
        "timestamp_utc": forecast["meta"]["timestamp"],
        "forecast_version": forecast["meta"]["version"],
        "client_location_name": forecast["meta"].get("client_location_name"),
        "capacity_kw": forecast["meta"].get("capacity_kw"),
        "latitude": forecast["meta"].get("latitude"),
        "longitude": forecast["meta"].get("longitude"),
        "location_type": forecast["meta"].get("location_type"),
    }

    # Only the fields ForecastSQL accepts — extra keys cause a TypeError via **forecast_meta
    db_forecast_meta = {
        "location_uuid": forecast_meta["location_uuid"],
        "timestamp_utc": forecast_meta["timestamp_utc"],
        "forecast_version": forecast_meta["forecast_version"],
    }

    forecast_values_df = pd.DataFrame(forecast["values"])
    forecast_values_df["horizon_minutes"] = (
        (forecast_values_df["start_utc"] - forecast_meta["timestamp_utc"]) / pd.Timedelta("60s")
    ).astype("int")

    # Persist base forecast to DB
    write_forecast_to_db(
        db_session,
        db_forecast_meta,
        forecast_values_df,
        write_to_db=bool(write_to_db),
        ml_model_name=ml_model_name,
        ml_model_version=ml_model_version,
    )

    # Persist adjuster forecast to DB
    if use_adjuster_database and ml_model_name is not None:
        from india_forecast_app.save.database import adjust_and_save_forecast
        adjust_and_save_forecast(
            db_session,
            db_forecast_meta,
            forecast_values_df,
            ml_model_name=ml_model_name,
            ml_model_version=ml_model_version,
            adjuster_average_minutes=adjuster_average_minutes,
            write_to_db=bool(write_to_db),
        )

    output = (
        f"Forecast for location_id={forecast_meta['location_uuid']},"
        f"timestamp={forecast_meta['timestamp_utc']},"
        f"version={forecast_meta['forecast_version']}:"
    )
    log.info(output)
    log.info(f"\n{forecast_values_df.to_string()}\n")

    # Optionally push to the Data Platform
    if os.getenv("SAVE_TO_DATA_PLATFORM", "false").lower() == "true":
        log.info("Saving to Data Platform...")
        asyncio.run(
            save_to_dataplatform(
                forecast_df=forecast_values_df,
                forecast_meta=forecast_meta,
                ml_model_name=ml_model_name,
                location_map=location_map,
                use_adjuster=ml_model_name is not None,
            ),
        )
