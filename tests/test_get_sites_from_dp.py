"""
Tests for loading the sites to forecast from the Data Platform.

Tests:
 1. test_loads_the_pinned_location_for_the_asset_type                  - pinned location, one asset
 2. test_loads_the_client_sites_sorted_by_name                         - client sites, sorted
 3. test_matches_on_client_prefix                                      - matched on client prefix
 4. test_matches_on_dp_location_name                                   - exact dp_location_name
 5. test_is_zero_for_the_national_location                             - national location is 0
 6. test_comes_from_region_id_before_ml_id                             - region_id beats ml_id
 7. test_falls_back_to_ml_id_metadata                                  - ml_id without region_id
 8. test_reads_a_metadata_value_written_as_a_string                    - string metadata is read
 9. test_metadata_of_zero_is_used                                      - 0 is a real id
10. test_defaults_to_one_without_metadata                              - no metadata means 1
11. test_defaults_to_one_when_the_metadata_is_not_a_number             - unparsable id means 1
12. test_get_sites_needs_a_model_config_to_load_from_the_data_platform - needs a model config
"""

import asyncio
import contextlib
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import ocf.dp as dp
import pytest
from betterproto.lib.google.protobuf import Struct, Value

from india_forecast_app.app import get_sites
from india_forecast_app.data_platform import (
    get_sites_from_data_platform,
    location_name_matches,
    ml_id_for_location,
)
from india_forecast_app.models.pydantic_models import Model


def _metadata(values: dict) -> Struct:
    """Make a Data Platform metadata struct."""
    return Struct(
        fields={
            key: Value(string_value=value) if isinstance(value, str) else Value(number_value=value)
            for key, value in values.items()
        },
    )


def _make_location(
    location_name: str,
    location_type: dp.LocationType = dp.LocationType.STATE,
    energy_source: dp.EnergySource = dp.EnergySource.SOLAR,
    effective_capacity_watts: int = 5_000_000,
    location_uuid: str | None = None,
    metadata: dict | None = None,
) -> dp.ListLocationsResponseLocationSummary:
    """Make a Data Platform location, using the real message type."""
    return dp.ListLocationsResponseLocationSummary(
        location_uuid=location_uuid or str(uuid.uuid4()),
        location_name=location_name,
        location_type=location_type,
        energy_source=energy_source,
        effective_capacity_watts=effective_capacity_watts,
        latlng=dp.LatLng(latitude=27.0, longitude=73.0),
        metadata=_metadata(metadata or {}),
    )


def _patch_client(locations: list[dp.ListLocationsResponseLocationSummary]):
    """Patch get_dataplatform_client with one honouring the type and energy source filters."""
    client = MagicMock()

    async def list_locations(request: dp.ListLocationsRequest) -> MagicMock:
        return MagicMock(
            locations=[
                location
                for location in locations
                if location.location_type == request.location_type_filter
                and location.energy_source == request.energy_source_filter
            ],
        )

    client.list_locations = AsyncMock(side_effect=list_locations)

    @contextlib.asynccontextmanager
    async def _fake_client():
        yield client

    return patch("india_forecast_app.data_platform.get_dataplatform_client", _fake_client)


def _model_config(**kwargs) -> Model:
    """Make a model config, so these tests do not depend on all_models.yaml."""
    return Model(name="test_model", id="openclimatefix-models/test", version="abc123", **kwargs)


class TestGetSitesFromDataPlatform:
    """Tests for get_sites_from_data_platform."""

    def test_loads_the_pinned_location_for_the_asset_type(self):
        """RUVNL solar and wind share one location, split by energy source."""
        ruvnl_uuid = str(uuid.uuid4())
        locations = [
            _make_location(
                "ruvnl",
                energy_source=dp.EnergySource.SOLAR,
                effective_capacity_watts=5_000_000,
                location_uuid=ruvnl_uuid,
            ),
            _make_location(
                "ruvnl",
                energy_source=dp.EnergySource.WIND,
                effective_capacity_watts=3_000_000,
                location_uuid=ruvnl_uuid,
            ),
            _make_location("nl_drenthe", metadata={"region_id": 1}),
        ]
        model_config = _model_config(
            client="ruvnl",
            asset_type="wind",
            location_type="state",
            dp_location_name="ruvnl",
        )

        with _patch_client(locations):
            sites = asyncio.run(get_sites_from_data_platform(model_config))

        assert len(sites) == 1
        assert sites[0].client_location_name == "ruvnl"
        assert sites[0].asset_type.name == "wind"
        assert sites[0].capacity_kw == 3000
        assert str(sites[0].location_uuid) == ruvnl_uuid
        assert sites[0].latitude == 27.0
        assert sites[0].longitude == 73.0

    def test_loads_the_client_sites_sorted_by_name(self):
        """Without a dp_location_name, the client's sites are loaded, sorted by name."""
        locations = [
            _make_location(
                "ad_site_2",
                location_type=dp.LocationType.SITE,
                energy_source=dp.EnergySource.WIND,
                metadata={"ml_id": 2},
            ),
            _make_location(
                "ad_site_1",
                location_type=dp.LocationType.SITE,
                energy_source=dp.EnergySource.WIND,
                metadata={"ml_id": 1},
            ),
            _make_location(
                "ruvnl",
                energy_source=dp.EnergySource.WIND,
            ),
        ]
        model_config = _model_config(client="ad", asset_type="wind", location_type="site")

        with _patch_client(locations):
            sites = asyncio.run(get_sites_from_data_platform(model_config))

        assert [site.client_location_name for site in sites] == ["ad_site_1", "ad_site_2"]
        assert [site.ml_id for site in sites] == [1, 2]


class TestLocationNameMatches:
    """Tests for location_name_matches."""

    def test_matches_on_client_prefix(self):
        """Without a dp_location_name, locations are matched on the client prefix."""
        model_config = _model_config(client="ad")

        assert location_name_matches("ad_site_1", model_config)
        assert not location_name_matches("ruvnl", model_config)

    def test_matches_on_dp_location_name(self):
        """With a dp_location_name, only that exact location is matched."""
        model_config = _model_config(client="ruvnl", dp_location_name="ruvnl")

        assert location_name_matches("ruvnl", model_config)
        assert not location_name_matches("ruvnl_jaisalmer", model_config)


class TestMlIdForLocation:
    """Tests for ml_id_for_location."""

    def test_is_zero_for_the_national_location(self):
        """The national location keeps ml_id 0, whatever its metadata says."""
        location = _make_location("india", dp.LocationType.NATION, metadata={"region_id": 9})

        assert ml_id_for_location(location) == 0

    def test_comes_from_region_id_before_ml_id(self):
        """region_id wins when a location carries both."""
        location = _make_location("ruvnl", metadata={"region_id": 3, "ml_id": 7})

        assert ml_id_for_location(location) == 3

    def test_falls_back_to_ml_id_metadata(self):
        """A location without a region_id uses its ml_id metadata."""
        location = _make_location("ad_site_1", dp.LocationType.SITE, metadata={"ml_id": 7})

        assert ml_id_for_location(location) == 7

    def test_reads_a_metadata_value_written_as_a_string(self):
        """Metadata values can arrive as strings rather than numbers."""
        location = _make_location("ad_site_1", dp.LocationType.SITE, metadata={"ml_id": "7"})

        assert ml_id_for_location(location) == 7

    def test_metadata_of_zero_is_used(self):
        """A metadata id of 0 is a real value, not a missing one."""
        location = _make_location("ad_site_1", dp.LocationType.SITE, metadata={"ml_id": 0})

        assert ml_id_for_location(location) == 0

    def test_defaults_to_one_without_metadata(self):
        """Locations that carry no id at all fall back to 1."""
        location = _make_location("ruvnl")

        assert ml_id_for_location(location) == 1

    def test_defaults_to_one_when_the_metadata_is_not_a_number(self):
        """A metadata value that is not a whole number is ignored rather than raising."""
        location = _make_location(
            "ad_site_1", dp.LocationType.SITE, metadata={"ml_id": "not a number"}
        )

        assert ml_id_for_location(location) == 1


def test_get_sites_needs_a_model_config_to_load_from_the_data_platform(monkeypatch):
    """The model config determines which locations to load, so it cannot be None"""
    monkeypatch.setenv("LOAD_SITES_FROM_DATA_PLATFORM", "true")

    with pytest.raises(ValueError, match="LOAD_SITES_FROM_DATA_PLATFORM"):
        get_sites(db_session=None)
