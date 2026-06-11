"""
Tests for india_forecast_app.save.data_platform.

Covers:
  - save/data_platform.py : prepare_forecast_values, resolve_target_uuid, fetch_dp_location_map
"""

from __future__ import annotations

import asyncio
import datetime as dt
from datetime import UTC
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from india_forecast_app.save.data_platform import (
    fetch_dp_location_map,
    prepare_forecast_values,
    resolve_target_uuid,
)
from tests._utils import _make_forecast_values_df


class TestPrepareForecastValues:
    """Tests for prepare_forecast_values (pure function, no gRPC)."""

    def test_returns_correct_number_of_values(self):
        """[14] Test that the correct number of forecast values is returned."""
        df = _make_forecast_values_df(n=4)
        init_time = dt.datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        capacity_watts = 10_000_000  # 10 MW

        result = prepare_forecast_values(df, init_time, capacity_watts)
        assert len(result) == 4

    def test_p50_fraction_clamped_between_0_and_1(self):
        """[15] Test that p50 fraction values are clamped to the 0-1 range."""
        # forecast_power_kw much larger than capacity → fraction should be 1.0
        init_time = dt.datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        starts = [init_time + dt.timedelta(minutes=15 * i) for i in range(3)]
        ends = [s + dt.timedelta(minutes=15) for s in starts]
        df = pd.DataFrame(
            {
                "start_utc": starts,
                "end_utc": ends,
                "forecast_power_kw": [0.0, 5000.0, 999_999.0],  # last far exceeds capacity
                "horizon_minutes": [0, 15, 30],
            }
        )
        capacity_watts = 1_000_000  # 1 MW = 1000 kW
        result = prepare_forecast_values(df, init_time, capacity_watts)

        assert result[0].p50_fraction == pytest.approx(0.0)
        assert 0.0 <= result[1].p50_fraction <= 1.0
        assert result[2].p50_fraction == pytest.approx(1.0)  # clamped to 1

    def test_horizon_mins_computed_from_start_utc_when_no_column(self):
        """[16] Test that horizon mins are computed from start_utc if missing."""
        init_time = dt.datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        starts = [init_time + dt.timedelta(minutes=15 * i) for i in range(3)]
        ends = [s + dt.timedelta(minutes=15) for s in starts]
        df = pd.DataFrame(
            {
                "start_utc": starts,
                "end_utc": ends,
                "forecast_power_kw": [100.0, 200.0, 300.0],
                # NOTE: no horizon_minutes column
            }
        )
        capacity_watts = 1_000_000
        result = prepare_forecast_values(df, init_time, capacity_watts)
        assert [fv.horizon_mins for fv in result] == [0, 15, 30]

    def test_horizon_from_existing_column_used(self):
        """[17] Test that the existing horizon_minutes column is used."""
        init_time = dt.datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        starts = [init_time + dt.timedelta(minutes=15 * i) for i in range(3)]
        ends = [s + dt.timedelta(minutes=15) for s in starts]
        df = pd.DataFrame(
            {
                "start_utc": starts,
                "end_utc": ends,
                "forecast_power_kw": [100.0, 200.0, 300.0],
                "horizon_minutes": [0, 15, 30],
            }
        )
        capacity_watts = 1_000_000
        result = prepare_forecast_values(df, init_time, capacity_watts)
        assert [fv.horizon_mins for fv in result] == [0, 15, 30]

    def test_probabilistic_values_parsed_when_dict(self):
        """[18] Test parsing probabilistic values when they are a dictionary."""
        init_time = dt.datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        starts = [init_time + dt.timedelta(minutes=15 * i) for i in range(2)]
        ends = [s + dt.timedelta(minutes=15) for s in starts]
        df = pd.DataFrame(
            {
                "start_utc": starts,
                "end_utc": ends,
                "forecast_power_kw": [500.0, 1000.0],
                "horizon_minutes": [0, 15],
                "probabilistic_values": [
                    {"p10": 400.0, "p90": 600.0},
                    {"p10": 800.0, "p90": 1200.0},
                ],
            }
        )
        capacity_watts = 2_000_000  # 2 MW = 2000 kW
        result = prepare_forecast_values(df, init_time, capacity_watts)
        assert "p10" in result[0].other_statistics_fractions
        assert "p90" in result[0].other_statistics_fractions
        assert 0.0 <= result[0].other_statistics_fractions["p10"] <= 1.0

    def test_empty_dataframe_returns_empty_list(self):
        """[19] Test that an empty DataFrame returns an empty list."""
        df = pd.DataFrame(columns=["start_utc", "end_utc", "forecast_power_kw", "horizon_minutes"])
        init_time = dt.datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        result = prepare_forecast_values(df, init_time, 1_000_000)
        assert result == []


class TestResolveTargetUuid:
    """Tests for resolve_target_uuid (uses mocked gRPC client)."""

    def test_returns_uuid_from_prefetched_map(self):
        """[20] Test returning UUID from prefetched map directly."""
        async def _run():
            mock_client = MagicMock()
            location_map = {"my_location": "abc-123"}
            result = await resolve_target_uuid(mock_client, "my_location", location_map)
            assert result == "abc-123"
            mock_client.list_locations.assert_not_called()

        asyncio.run(_run())

    def test_returns_none_when_location_not_in_map(self):
        """[21] Test returning None when location is not in map."""
        async def _run():
            mock_client = MagicMock()
            location_map = {"other_location": "xyz-456"}
            result = await resolve_target_uuid(mock_client, "my_location", location_map)
            assert result is None
            mock_client.list_locations.assert_not_called()

        asyncio.run(_run())

    def test_fetches_map_when_none_provided(self):
        """[22] Test fetching the location map if not provided."""
        async def _run():
            mock_loc = MagicMock()
            mock_loc.location_name = "my_location"
            mock_loc.location_uuid = "uuid-999"

            mock_client = AsyncMock()
            mock_client.list_locations.return_value = MagicMock(locations=[mock_loc])

            result = await resolve_target_uuid(mock_client, "my_location", location_map=None)
            assert result == "uuid-999"
            mock_client.list_locations.assert_called_once()

        asyncio.run(_run())

    def test_returns_none_when_not_found_in_fetched_map(self):
        """[23] Test returning None when location is not found in fetched map."""
        async def _run():
            mock_client = AsyncMock()
            mock_client.list_locations.return_value = MagicMock(locations=[])

            result = await resolve_target_uuid(mock_client, "missing_location", location_map=None)
            assert result is None

        asyncio.run(_run())


class TestFetchDpLocationMap:
    """Tests for fetch_dp_location_map."""

    def test_builds_name_to_uuid_map(self):
        """[24] Test building location name to UUID map from response."""
        async def _run():
            locs = []
            for name, uid in [("loc_a", "uuid-a"), ("loc_b", "uuid-b")]:
                m = MagicMock()
                m.location_name = name
                m.location_uuid = uid
                locs.append(m)

            mock_client = AsyncMock()
            mock_client.list_locations.return_value = MagicMock(locations=locs)

            result = await fetch_dp_location_map(mock_client)
            assert result == {"loc_a": "uuid-a", "loc_b": "uuid-b"}

        asyncio.run(_run())

    def test_empty_locations_returns_empty_map(self):
        """[25] Test returning empty map when no locations exist."""
        async def _run():
            mock_client = AsyncMock()
            mock_client.list_locations.return_value = MagicMock(locations=[])

            result = await fetch_dp_location_map(mock_client)
            assert result == {}

        asyncio.run(_run())
