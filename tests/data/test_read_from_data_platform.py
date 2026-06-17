"""Tests for reading generation data from the Data Platform.

Tests cover (all in data/generation.py):
1. energy_source_for_asset_type: "pv" maps to the SOLAR energy source
2. energy_source_for_asset_type: "wind" maps to the WIND energy source
3. fetch_generation_from_dp: value_fraction is converted to power_kw using capacity
4. fetch_generation_from_dp: request carries the mapped energy source and tz-aware window
5. fetch_generation_from_dp: unknown location returns empty without a gRPC call
6. get_generation_data: READ_FROM_DATA_PLATFORM branch sources generation from the DP
"""

import asyncio
import contextlib
import datetime as dt
from unittest.mock import AsyncMock, MagicMock, patch

import ocf.dp as dp
from pvsite_datamodel.sqlmodels import LocationAssetType

from india_forecast_app.data.generation import (
    energy_source_for_asset_type,
    fetch_generation_from_dp,
    get_generation_data,
)


def _patch_client(mock_client):
    """Patch get_dataplatform_client to yield the given mock client."""

    @contextlib.asynccontextmanager
    async def _fake_client():
        yield mock_client

    return patch(
        "india_forecast_app.data.generation.get_dataplatform_client",
        _fake_client,
    )


class TestEnergySourceForAssetType:
    """Tests for energy_source_for_asset_type."""

    def test_pv_maps_to_solar(self):
        """[1] Test that "pv" maps to the SOLAR energy source."""
        assert energy_source_for_asset_type("pv") == dp.EnergySource.SOLAR

    def test_wind_maps_to_wind(self):
        """[2] Test that "wind" maps to the WIND energy source."""
        assert energy_source_for_asset_type("wind") == dp.EnergySource.WIND


class TestFetchGenerationFromDp:
    """Tests for fetch_generation_from_dp (uses a mocked gRPC client)."""

    def test_converts_fraction_to_power_kw(self):
        """[3] Test value_fraction * effective_capacity_watts / 1000 == power_kw."""
        ts = dt.datetime(2024, 1, 1, 12, 0, tzinfo=dt.UTC)
        value = dp.GetObservationsAsTimeseriesResponseValue(
            timestamp_utc=ts,
            value_fraction=0.5,
            effective_capacity_watts=2_000_000,
        )
        mock_client = MagicMock()
        mock_client.get_observations_as_timeseries = AsyncMock(
            return_value=dp.GetObservationsAsTimeseriesResponse(
                location_uuid="uuid-1", values=[value],
            ),
        )

        async def _run():
            with _patch_client(mock_client), patch(
                "india_forecast_app.data.generation.fetch_dp_location_map",
                AsyncMock(return_value={"my_location": "uuid-1"}),
            ):
                return await fetch_generation_from_dp(
                    client_location_name="my_location",
                    start=dt.datetime(2024, 1, 1, 11, 0),
                    end=dt.datetime(2024, 1, 1, 12, 0),
                    asset_type="wind",
                )

        result = asyncio.run(_run())
        # 0.5 * 2_000_000 W = 1_000_000 W = 1000 kW
        assert result == [(ts, 1000.0)]

    def test_request_uses_correct_energy_source_and_window(self):
        """[4] Test the DP request carries the mapped energy source and a tz-aware window."""
        mock_client = MagicMock()
        mock_client.get_observations_as_timeseries = AsyncMock(
            return_value=dp.GetObservationsAsTimeseriesResponse(
                location_uuid="uuid-1", values=[],
            ),
        )

        async def _run():
            with _patch_client(mock_client), patch(
                "india_forecast_app.data.generation.fetch_dp_location_map",
                AsyncMock(return_value={"my_location": "uuid-1"}),
            ):
                return await fetch_generation_from_dp(
                    client_location_name="my_location",
                    start=dt.datetime(2024, 1, 1, 11, 0),
                    end=dt.datetime(2024, 1, 1, 12, 0),
                    asset_type="pv",
                    observer_name="india",
                )

        asyncio.run(_run())

        request = mock_client.get_observations_as_timeseries.call_args.args[0]
        assert request.location_uuid == "uuid-1"
        assert request.observer_name == "india"
        assert request.energy_source == dp.EnergySource.SOLAR
        assert request.time_window.start_timestamp_utc.tzinfo is not None

    def test_returns_empty_when_location_unknown(self):
        """[5] Test location not in the map -> no gRPC observation call, empty result."""
        mock_client = MagicMock()
        mock_client.get_observations_as_timeseries = AsyncMock()

        async def _run():
            with _patch_client(mock_client), patch(
                "india_forecast_app.data.generation.fetch_dp_location_map",
                AsyncMock(return_value={}),
            ):
                return await fetch_generation_from_dp(
                    client_location_name="missing",
                    start=dt.datetime(2024, 1, 1, 11, 0),
                    end=dt.datetime(2024, 1, 1, 12, 0),
                    asset_type="wind",
                )

        result = asyncio.run(_run())
        assert result == []
        mock_client.get_observations_as_timeseries.assert_not_called()


class TestGetGenerationDataReadFromDp:
    """Tests for the READ_FROM_DATA_PLATFORM branch of get_generation_data."""

    def test_reads_from_dp_when_env_set(
        self, db_session, sites, init_timestamp, client_ruvnl, monkeypatch,
    ):
        """[6] Test that with the env flag set, generation is sourced from the DP, not the DB."""
        monkeypatch.setenv("READ_FROM_DATA_PLATFORM", "true")

        wind_site = [s for s in sites if s.asset_type == LocationAssetType.wind][0]

        # Three 3-min readings ending at t0, tz-aware UTC as the DP returns them.
        dp_values = [
            (
                (init_timestamp - dt.timedelta(minutes=m)).tz_localize("UTC"),
                5000.0,
            )
            for m in (6, 3, 0)
        ]

        with patch(
            "india_forecast_app.data.generation.fetch_generation_from_dp",
            AsyncMock(return_value=dp_values),
        ) as mock_fetch:
            gen_data = get_generation_data(db_session, [wind_site], timestamp=init_timestamp)

        gen_df = gen_data["data"]

        # The DP fetch was used with the correct location name and asset type.
        kwargs = mock_fetch.call_args.kwargs
        assert kwargs["client_location_name"] == wind_site.client_location_name.lower()
        assert kwargs["asset_type"] == "wind"

        # Index is tz-naive after normalisation and covers the expected window.
        assert gen_df.index.tz is None
        assert gen_df.index[-1] == init_timestamp
        assert not gen_df["0"].isnull().any()
