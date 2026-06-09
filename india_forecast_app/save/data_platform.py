"""Data Platform operations: location management, forecaster lifecycle, and forecast saving."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import traceback
from collections.abc import AsyncIterator  # noqa: TC003
from datetime import UTC, datetime, timedelta
from importlib.metadata import version

import ocf.dp as dp
import pandas as pd
from betterproto.lib.google.protobuf import Struct, Value
from grpclib.client import Channel

from india_forecast_app.save.utils import (
    add_or_convert_to_utc,
    ensure_timezone_aware,
    limit_adjuster,
)

log = logging.getLogger(__name__)

# -- Version --
# we need to keep this static so that the adjust and api works,
# even if we change version
# we will put the app version in the metadata
dp_forecaster_version = "1.4.0"

# Type alias for the Data Platform client stub
DataPlatformClient = dp.DataPlatformDataServiceStub


async def fetch_dp_location_map(client: DataPlatformClient) -> dict[str, str]:
    """Fetch all locations (SITE, NATION, STATE, etc.) from the Data Platform.

    Returns a name → UUID map. Pre-fetching avoids separate list_locations calls
    for every forecast save.
    """
    resp = await client.list_locations(dp.ListLocationsRequest())
    return {loc.location_name: loc.location_uuid for loc in resp.locations}


async def build_dp_location_map() -> dict[str, str]:
    """Async wrapper: open a channel, fetch the location map, close."""
    async with get_dataplatform_client() as client:
        return await fetch_dp_location_map(client)


@contextlib.asynccontextmanager
async def get_dataplatform_client() -> AsyncIterator[DataPlatformClient]:
    """Async context manager that opens a gRPC channel and yields a ready-to-use client.

    Usage::

        async with get_dataplatform_client() as client:
            await save_forecast_to_dataplatform(..., client=client)

    The channel is always closed on exit, even if an exception is raised.
    Host and port are read from the ``DATA_PLATFORM_HOST``/
    ``DATA_PLATFORM_PORT`` environment variables
    (defaulting to ``localhost:50051``).
    """
    channel = Channel(
        host=os.getenv("DATA_PLATFORM_HOST", "localhost"),
        port=int(os.getenv("DATA_PLATFORM_PORT", "50051")),
    )
    try:
        yield dp.DataPlatformDataServiceStub(channel)
    finally:
        channel.close()


async def save_to_dataplatform(
    forecast_df: pd.DataFrame,
    forecast_meta: dict,
    ml_model_name: str | None,
    location_map: dict[str, str] | None = None,
    use_adjuster: bool = True,
) -> None:
    """Save Forecast to Dataplatform."""
    client_location_name = forecast_meta.get("client_location_name")
    model_tag = ml_model_name if ml_model_name else "default-model"
    init_time_utc = forecast_meta["timestamp_utc"]
    capacity_kw = forecast_meta.get("capacity_kw")

    log.info(
        "Starting DP save | "
        f"location={client_location_name!r}  model={model_tag!r}  "
        f"init_time={init_time_utc}  capacity_kw={capacity_kw}  "
        f"df_rows={len(forecast_df)}  "
        f"location_map_size={len(location_map) if location_map else None}",
    )

    try:
        async with get_dataplatform_client() as client:
            await save_forecast_to_dataplatform(
                forecast_df=forecast_df,
                client_location_name=client_location_name,
                model_tag=model_tag,
                init_time_utc=init_time_utc,
                client=client,
                capacity_kw=capacity_kw,
                latitude=forecast_meta.get("latitude"),
                longitude=forecast_meta.get("longitude"),
                location_type=forecast_meta.get("location_type", dp.LocationType.SITE),
                location_map=location_map,
                use_adjuster=use_adjuster,
            )
        log.info(f"Save complete for location={client_location_name!r}")
    except Exception as e:
        log.error(f"Failed to save forecast to Data Platform: {e}")
        log.error(traceback.format_exc())


async def make_forecaster_adjuster(
    client: DataPlatformClient,
    location_uuid: str,
    init_time_utc: datetime,
    forecast_values: list[dp.CreateForecastRequestForecastValue],
    model_tag: str,
    forecaster: dp.Forecaster,
) -> dp.CreateForecastRequest:
    """Build an adjusted forecast request using week-average deltas from the Data Platform.

    Fetches week-average forecast vs. observation deltas for each horizon via
    ``GetWeekAverageDeltasRequest``, applies ``limit_adjuster`` to cap the correction,
    and returns a ``CreateForecastRequest`` tagged with the ``{model_tag}_adjust} forecaster.

    Args:
        client: An active DataPlatform gRPC client.
        location_uuid: DP location UUID string.
        init_time_utc: Forecast initialisation time (timezone-aware).
        forecast_values: Base forecast values to adjust.
        model_tag: Model name used to look up/create the adjuster forecaster.
        forecaster: The base forecaster object (used to fetch deltas).

    Returns:
        A ``CreateForecastRequest`` for the adjusted forecast.
    """
    deltas_request = dp.GetWeekAverageDeltasRequest(
        location_uuid=location_uuid,
        energy_source=dp.EnergySource.SOLAR,
        pivot_timestamp_utc=init_time_utc.replace(tzinfo=UTC),
        forecaster=forecaster,
        observer_name=os.getenv("OBSERVER_NAME", "india"),
    )
    deltas_response = await client.get_week_average_deltas(deltas_request)
    deltas = deltas_response.deltas

    # Build a horizon -> delta_fraction lookup for O(1) access
    delta_by_horizon: dict[int, float] = {d.horizon_mins: d.delta_fraction for d in deltas}

    # Get location capacity for capping
    location = await client.get_location(
        dp.GetLocationRequest(
            location_uuid=location_uuid,
            energy_source=dp.EnergySource.SOLAR,
            include_geometry=False,
        ),
    )
    capacity_mw = location.effective_capacity_watts / 1_000_000.0

    adjusted_values: list[dp.CreateForecastRequestForecastValue] = []
    for fv in forecast_values:
        raw_delta = delta_by_horizon.get(fv.horizon_mins, 0.0)
        capped_delta = limit_adjuster(
            delta_fraction=raw_delta,
            value_fraction=fv.p50_fraction,
            capacity_mw=capacity_mw,
        )

        new_p50 = max(0.0, min(1.0, fv.p50_fraction - capped_delta))
        new_other_stats: dict[str, float] = {
            key: max(0.0, min(1.0, val - capped_delta))
            for key, val in fv.other_statistics_fractions.items()
        }

        adjusted_values.append(
            dp.CreateForecastRequestForecastValue(
                horizon_mins=fv.horizon_mins,
                p50_fraction=new_p50,
                other_statistics_fractions=new_other_stats,
            ),
        )

    adjuster_forecaster = await create_forecaster_if_not_exists(
        client=client,
        model_tag=model_tag + "_adjust",
    )

    return dp.CreateForecastRequest(
        forecaster=adjuster_forecaster,
        location_uuid=location_uuid,
        energy_source=dp.EnergySource.SOLAR,
        init_time_utc=init_time_utc.replace(tzinfo=UTC),
        values=adjusted_values,
    )


async def resolve_target_uuid(
    client: DataPlatformClient,
    client_location_name: str,
    location_map: dict[str, str] | None = None,
) -> str | None:
    """Look up the DP location UUID by name.

    If a pre-fetched *location_map* (name → UUID) is supplied it is used directly,
    avoiding an extra list_locations gRPC call.  When None, the map is fetched
    on-demand as before.

    Returns the UUID string if found, or None if the location does not exist yet.
    Raises on unexpected gRPC errors.
    """
    if location_map is None:
        resp = await client.list_locations(dp.ListLocationsRequest())
        location_map = {loc.location_name: loc.location_uuid for loc in resp.locations}

    if client_location_name in location_map:
        target_uuid = location_map[client_location_name]
        log.info(f"Mapped client location '{client_location_name}' to DP UUID {target_uuid}")
        return target_uuid

    log.warning(f"DP location '{client_location_name}' not found — will create it.")
    return None


async def get_location_capacity(
    client: DataPlatformClient,
    target_uuid_str: str,
) -> int:
    """Fetch effective capacity (watts) for an existing DP location."""
    location = await client.get_location(
        dp.GetLocationRequest(
            location_uuid=target_uuid_str,
            energy_source=dp.EnergySource.SOLAR,
            include_geometry=False,
        ),
    )
    return location.effective_capacity_watts


async def create_new_location(
    client: DataPlatformClient,
    client_location_name: str,
    capacity_kw: float,
    latitude: float | None,
    longitude: float | None,
    init_time_utc: datetime,
    location_type: dp.LocationType = dp.LocationType.SITE,
) -> str:
    """Create a new location in the Data Platform and return its UUID."""
    log.warning(
        f"Location {client_location_name} (type={location_type.name}) not found. "
        "Attempting to create it...",
    )

    lon, lat = longitude or 0.0, latitude or 0.0
    wkt = f"POINT ({lon} {lat})"
    capacity_watts = int(capacity_kw * 1000)

    try:
        create_req = dp.CreateLocationRequest(
            location_name=client_location_name,
            energy_source=dp.EnergySource.SOLAR,
            geometry_wkt=wkt,
            effective_capacity_watts=capacity_watts,
            location_type=location_type,
            valid_from_utc=init_time_utc - timedelta(days=7),
        )
        create_resp = await client.create_location(create_req)
        log.info(f"Created new location {create_resp.location_uuid} for '{client_location_name}'")
        return create_resp.location_uuid
    except Exception as create_error:
        log.error(f"Failed to create location: {create_error}")
        raise


async def create_forecaster_if_not_exists(
    client: DataPlatformClient,
    model_tag: str,
) -> dp.Forecaster:
    """Create the current forecaster if it does not exist."""
    forecaster_name = model_tag.replace("-", "_").lower()

    list_forecasters_request = dp.ListForecastersRequest(
        forecaster_names_filter=[forecaster_name],
    )
    try:
        list_forecasters_response = await client.list_forecasters(list_forecasters_request)
        existing_forecasters = list_forecasters_response.forecasters
    except Exception as e:
        if "NOT_FOUND" in str(e) or "No forecasters found" in str(e):
            existing_forecasters = []
        else:
            raise

    if len(existing_forecasters) > 0:
        filtered_forecasters = [
            f
            for f in existing_forecasters
            if f.forecaster_version == dp_forecaster_version
        ]
        if len(filtered_forecasters) == 1:
            return filtered_forecasters[0]
        else:
            update_forecaster_request = dp.UpdateForecasterRequest(
                name=forecaster_name,
                new_version=dp_forecaster_version,
            )
            update_forecaster_response = await client.update_forecaster(update_forecaster_request)
            return update_forecaster_response.forecaster
    else:
        create_forecaster_request = dp.CreateForecasterRequest(
            name=forecaster_name,
            version=dp_forecaster_version,
        )
        create_forecaster_response = await client.create_forecaster(create_forecaster_request)
        return create_forecaster_response.forecaster


def prepare_forecast_values(
    forecast_df: pd.DataFrame,
    init_time_utc: datetime,
    capacity_watts: int,
) -> list[dp.CreateForecastRequestForecastValue]:
    """Convert a forecast DataFrame to a list of DP forecast value objects."""
    init_ts = add_or_convert_to_utc(init_time_utc)

    forecast_values: list[dp.CreateForecastRequestForecastValue] = []

    # Pre-parse probabilistic values if they exist
    prob_values_parsed: dict = {}
    if "probabilistic_values" in forecast_df.columns:
        for idx, prob_str in forecast_df["probabilistic_values"].items():
            if pd.notna(prob_str):
                # Handle both dict and stringified JSON
                if isinstance(prob_str, dict):
                    prob_values_parsed[idx] = prob_str
                else:
                    prob_values_parsed[idx] = json.loads(prob_str)

    for row in forecast_df.itertuples():
        # Calculate horizon
        if hasattr(row, "horizon_minutes") and pd.notna(row.horizon_minutes):
            horizon_mins = int(row.horizon_minutes)
        else:
            start_ts = add_or_convert_to_utc(row.start_utc)
            horizon_mins = int((start_ts - init_ts).total_seconds() / 60)

        # Convert and clamp power fraction
        p50_fraction = max(0.0, min(1.0, (row.forecast_power_kw * 1000) / capacity_watts))

        # Process probabilistic values
        other_stats: dict[str, float] = {}
        if row.Index in prob_values_parsed:
            for key, val_kw in prob_values_parsed[row.Index].items():
                frac = max(0.0, min(1.0, (val_kw * 1000) / capacity_watts))
                other_stats[key] = frac

        forecast_values.append(
            dp.CreateForecastRequestForecastValue(
                horizon_mins=horizon_mins,
                p50_fraction=p50_fraction,
                other_statistics_fractions=other_stats,
            ),
        )

    return forecast_values


async def save_forecast_to_dataplatform(
    forecast_df: pd.DataFrame,
    client_location_name: str | None,
    model_tag: str,
    init_time_utc: datetime,
    client: DataPlatformClient,
    capacity_kw: float | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    location_type: dp.LocationType = dp.LocationType.SITE,
    location_map: dict[str, str] | None = None,
    use_adjuster: bool = True,
) -> None:
    """Save forecast to the Data Platform."""
    app_version = version("india-forecast-app")
    if forecast_df.empty:
        log.warning("forecast dataframe is empty")
        return

    if not client_location_name:
        log.error("client_location_name is None/empty — cannot save")
        raise ValueError("client_location_name is required to save to the Data Platform")

    init_time_utc = ensure_timezone_aware(init_time_utc)
    log.info(
        f"location={client_location_name!r}  "
        f"model={model_tag!r}  init_time={init_time_utc}  rows={len(forecast_df)}",
    )

    log.info(f"resolving UUID and forecaster for {client_location_name!r}")
    target_uuid_str, forecaster = await asyncio.gather(
        resolve_target_uuid(client, client_location_name, location_map),
        create_forecaster_if_not_exists(client=client, model_tag=model_tag),
    )
    log.info(
        f"uuid={target_uuid_str}  "
        f"forecaster={forecaster.forecaster_name!r} v{forecaster.forecaster_version}",
    )

    if target_uuid_str is None:
        log.warning(
            f"location {client_location_name!r} not in DP — creating it  "
            f"(capacity_kw={capacity_kw}, lat={latitude}, lon={longitude})",
        )
        target_uuid_str = await create_new_location(
            client,
            client_location_name,
            capacity_kw or 0.0,
            latitude,
            longitude,
            init_time_utc,
            location_type=location_type,
        )
        log.info(f"created location uuid={target_uuid_str}")
    else:
        log.info(f"location already exists uuid={target_uuid_str}")

    capacity_watts = await get_location_capacity(client=client, target_uuid_str=target_uuid_str)
    log.info(
        f"capacity_watts={capacity_watts:,}  ({capacity_watts / 1000:,.1f} kW)",
    )

    if capacity_watts == 0:
        log.error(
            f"location {target_uuid_str} has 0 W capacity — "
            "no forecast values can be expressed as fractions; skipping save",
        )
        return

    forecast_values = prepare_forecast_values(forecast_df, init_time_utc, capacity_watts)
    log.info(f"prepared {len(forecast_values)} forecast value(s)")

    if forecast_values:
        sample = forecast_values[0]
        log.info(
            f"sample[0]: horizon_mins={sample.horizon_mins}  "
            f"p50_fraction={sample.p50_fraction:.6f}  "
            f"other_stats={dict(sample.other_statistics_fractions)}",
        )
        p50s = [fv.p50_fraction for fv in forecast_values]
        log.info(
            f"p50 range: min={min(p50s):.6f}  max={max(p50s):.6f}  "
            f"mean={sum(p50s)/len(p50s):.6f}",
        )
    else:
        log.warning("no forecast values after preparation")
        return

    base_request = dp.CreateForecastRequest(
        forecaster=forecaster,
        location_uuid=target_uuid_str,
        energy_source=dp.EnergySource.SOLAR,
        init_time_utc=init_time_utc,
        values=forecast_values,
        metadata=Struct(fields={"app_version": Value(string_value=app_version)}),
    )
    log.info(
        f"submitting forecast  "
        f"forecaster={forecaster.forecaster_name!r}  "
        f"location={target_uuid_str}  values={len(forecast_values)}",
    )

    await client.create_forecast(base_request)
    log.info(f"Forecast submitted for {client_location_name!r}")

    if use_adjuster:
        log.info(f"Building adjuster forecast for {client_location_name!r}")
        try:
            adjusted_request = await make_forecaster_adjuster(
                client=client,
                location_uuid=target_uuid_str,
                init_time_utc=init_time_utc,
                forecast_values=forecast_values,
                model_tag=model_tag,
                forecaster=forecaster,
            )
            await client.create_forecast(adjusted_request)
            log.info(f"Adjusted forecast submitted for {client_location_name!r}")
        except Exception:
            log.error(
                f"Failed to save adjusted forecast to Data Platform for {client_location_name!r}\n"
                + traceback.format_exc(),
            )
