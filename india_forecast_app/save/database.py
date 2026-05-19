"""Database operations for persisting forecasts."""

from __future__ import annotations

import logging
import traceback

import pandas as pd  # noqa: TC002
from pvsite_datamodel.write import insert_forecast_values
from sqlalchemy.orm import Session  # noqa: TC002

from india_forecast_app.adjuster import adjust_forecast_with_adjuster

log = logging.getLogger(__name__)


def write_forecast_to_db(
    db_session: Session,
    forecast_meta: dict,
    forecast_values_df: pd.DataFrame,
    *,
    write_to_db: bool,
    ml_model_name: str | None,
    ml_model_version: str | None,
) -> None:
    """Write a forecast dataframe to DB when enabled."""
    if not write_to_db:
        return

    insert_forecast_values(
        db_session,
        forecast_meta,
        forecast_values_df,
        ml_model_name=ml_model_name,
        ml_model_version=ml_model_version,
    )


def adjust_and_save_forecast(
    db_session: Session,
    forecast_meta: dict,
    forecast_values_df: pd.DataFrame,
    ml_model_name: str,
    ml_model_version: str | None,
    adjuster_average_minutes: int | None,
    write_to_db: bool,
) -> None:
    """Adjust forecast using the adjuster and save to DB."""
    log.info(f"Adjusting forecast for location_id={forecast_meta['location_uuid']}...")
    try:
        forecast_values_df_adjust = adjust_forecast_with_adjuster(
            db_session,
            forecast_meta,
            forecast_values_df,
            ml_model_name=ml_model_name,
            average_minutes=adjuster_average_minutes,
        )
        log.info(f"Adjusted forecast shape: {forecast_values_df_adjust.shape}")

        write_forecast_to_db(
            db_session,
            forecast_meta,
            forecast_values_df_adjust,
            write_to_db=write_to_db,
            ml_model_name=f"{ml_model_name}_adjust",
            ml_model_version=ml_model_version,
        )
    except Exception as e:
        log.error(f"Failed to adjust/save forecast for {ml_model_name}: {e}")
        log.error(traceback.format_exc())
