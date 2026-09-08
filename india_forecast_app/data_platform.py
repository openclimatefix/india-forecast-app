"""Loads the locations to make forecasts for from the Data Platform."""

from __future__ import annotations

import logging
import uuid

import betterproto
import ocf.dp as dp
from betterproto.lib.google.protobuf import Struct
from pvsite_datamodel.sqlmodels import LocationAssetType, LocationSQL

from india_forecast_app.models.pydantic_models import Model
from india_forecast_app.save.data_platform import (
    DataPlatformClient,
    get_dataplatform_client,
    to_dp_location_type,
)
from india_forecast_app.save.utils import energy_source_for_asset_type

log = logging.getLogger(__name__)


async def get_sites_from_data_platform(model_config: Model) -> list[LocationSQL]:
    """Gets the locations to make forecasts for, as set by the model config.

    Args:
            model_config: The model configuration to load locations for

    Returns:
            A list of LocationSQL objects
    """
    async with get_dataplatform_client() as client:
        locations = await list_dp_locations(client, model_config)

    return [
        dp_location_to_site(location, ml_id=ml_id_for_location(location)) for location in locations
    ]


async def list_dp_locations(
    client: DataPlatformClient,
    model_config: Model,
) -> list[dp.ListLocationsResponseLocationSummary]:
    """Lists the Data Platform locations that this model forecasts.

    A Data Platform location can hold several energy sources, each with its own capacity, and
    list_locations returns one entry per energy source. Filtering by the model's energy source
    therefore picks the right half of a shared location, such as the wind half of the ruvnl state.

    Args:
            client: An active Data Platform client
            model_config: The model configuration to load locations for

    Returns:
            A list of Data Platform locations, sorted by name for a deterministic order
    """
    response = await client.list_locations(
        dp.ListLocationsRequest(
            location_type_filter=to_dp_location_type(model_config.location_type),
            energy_source_filter=energy_source_for_asset_type(model_config.asset_type),
        ),
    )
    locations = [
        location
        for location in response.locations
        if location_name_matches(location.location_name, model_config)
    ]
    log.info(
        f"Found {len(locations)} {model_config.location_type} locations in the Data Platform "
        f"for asset_type={model_config.asset_type}",
    )
    return sorted(locations, key=lambda location: location.location_name)


def location_name_matches(location_name: str, model_config: Model) -> bool:
    """Checks a Data Platform location name against the filters in the model config.

    Args:
            location_name: The name of the Data Platform location
            model_config: The model configuration to check against

    Returns:
            True if the location should be loaded
    """
    if model_config.dp_location_name is not None:
        return location_name == model_config.dp_location_name

    return location_name.startswith(model_config.client)


def ml_id_for_location(location: dp.ListLocationsResponseLocationSummary) -> int:
    """Works out the ml id for a Data Platform location.

    The ml id picks which row of the model's inputs and outputs a location gets, so it has to
    match what the model was trained with, not where the location sits in the list.

    Args:
            location: A Data Platform location

    Returns:
            The ml id for the location
    """
    if location.location_type == dp.LocationType.NATION:
        return 0

    for key in ("region_id", "ml_id"):
        ml_id = metadata_int(location.metadata, key)
        if ml_id is not None:
            return ml_id

    # Adani and RUVNL have no id in the Data Platform yet, so they need one adding.
    log.warning(
        f"Location {location.location_name} has no region_id or ml_id in its Data Platform "
        "metadata, falling back to ml_id 1",
    )
    return 1


def metadata_int(metadata: Struct, key: str) -> int | None:
    """Reads a whole number out of a Data Platform location's metadata.

    A metadata value is a protobuf Value, so the same key can arrive as a number or a string
    depending on how it was written.

    Args:
            metadata: The metadata of a Data Platform location
            key: The metadata key to read

    Returns:
            The value as an int, or None if it is missing or is not a whole number
    """
    value = metadata.fields.get(key)
    if value is None:
        return None

    _, set_value = betterproto.which_one_of(value, "kind")

    try:
        return int(set_value)
    except (TypeError, ValueError):
        log.warning(f"Data Platform metadata {key}={set_value!r} is not a whole number")
        return None


def dp_location_to_site(
    location: dp.ListLocationsResponseLocationSummary,
    ml_id: int,
) -> LocationSQL:
    """Makes a site from a Data Platform location.

    Args:
            location: A Data Platform location
            ml_id: The ml id to give the site

    Returns:
            A LocationSQL object
    """
    return LocationSQL(
        location_uuid=uuid.UUID(location.location_uuid),
        client_location_name=location.location_name,
        asset_type=(
            LocationAssetType.wind
            if location.energy_source == dp.EnergySource.WIND
            else LocationAssetType.pv
        ),
        capacity_kw=location.effective_capacity_watts / 1000,
        latitude=location.latlng.latitude,
        longitude=location.latlng.longitude,
        ml_id=ml_id,
    )
