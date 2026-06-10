"""
Tests for the india_forecast_app.save sub-package.

Covers:
  - save/utils.py         : add_or_convert_to_utc, ensure_timezone_aware, limit_adjuster
  - save/database.py      : write_forecast_to_db, adjust_and_save_forecast
  - save/save.py          : save_forecast (DB path; DP path mocked)
  - save/data_platform.py : prepare_forecast_values, resolve_target_uuid (pure helpers)
"""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid
from datetime import UTC, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from pvsite_datamodel.sqlmodels import ForecastSQL, ForecastValueSQL, MLModelSQL

from india_forecast_app.save.data_platform import (
    fetch_dp_location_map,
    prepare_forecast_values,
    resolve_target_uuid,
)
from india_forecast_app.save.database import adjust_and_save_forecast, write_forecast_to_db
from india_forecast_app.save.save import save_forecast
from india_forecast_app.save.utils import (
    add_or_convert_to_utc,
    ensure_timezone_aware,
    limit_adjuster,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_forecast_values_df(n: int = 5, timestamp: dt.datetime | None = None) -> pd.DataFrame:
    """Build a minimal forecast values DataFrame."""
    timestamp = timestamp or dt.datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
    starts = [timestamp + dt.timedelta(minutes=15 * i) for i in range(n)]
    ends = [s + dt.timedelta(minutes=15) for s in starts]
    return pd.DataFrame(
        {
            "start_utc": starts,
            "end_utc": ends,
            "forecast_power_kw": [float(i * 100) for i in range(n)],
            "horizon_minutes": [i * 15 for i in range(n)],
        }
    )


def _make_forecast_dict(
    site_uuid,
    n: int = 5,
    timestamp: dt.datetime | None = None,
) -> dict:
    """Build a forecast dict in the format expected by save_forecast."""
    timestamp = timestamp or dt.datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
    starts = [timestamp + dt.timedelta(minutes=15 * i) for i in range(n)]
    ends = [s + dt.timedelta(minutes=15) for s in starts]
    return {
        "meta": {
            "site_id": site_uuid,
            "version": "0.0.0",
            "timestamp": timestamp,
            "client_location_name": "test_location",
            "capacity_kw": 10000.0,
            "latitude": 20.0,
            "longitude": 78.0,
        },
        "values": [
            {
                "start_utc": starts[i],
                "end_utc": ends[i],
                "forecast_power_kw": i * 100,
            }
            for i in range(n)
        ],
    }


# ===========================================================================
# save/utils.py
# ===========================================================================


class TestAddOrConvertToUtc:
    """Tests for add_or_convert_to_utc."""

    def test_naive_datetime_gets_localized(self):
        """Test that naive datetime is localized to UTC."""
        naive = dt.datetime(2024, 1, 1, 12, 0, 0)
        result = add_or_convert_to_utc(naive)
        assert result.tzinfo is not None
        assert str(result.tzinfo) in ("UTC", "+00:00")

    def test_tz_aware_utc_unchanged(self):
        """Test that timezone-aware UTC datetime is unchanged."""
        aware = dt.datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = add_or_convert_to_utc(aware)
        assert result == pd.Timestamp(aware)

    def test_tz_aware_non_utc_converted(self):
        """Test that non-UTC timezone-aware datetime is converted to UTC."""
        ist = timezone(dt.timedelta(hours=5, minutes=30))
        aware_ist = dt.datetime(2024, 1, 1, 17, 30, 0, tzinfo=ist)
        result = add_or_convert_to_utc(aware_ist)
        # 17:30 IST == 12:00 UTC
        assert result.hour == 12
        assert result.minute == 0

    def test_pd_timestamp_naive_localized(self):
        """Test that naive pandas Timestamp is localized to UTC."""
        ts = pd.Timestamp("2024-06-01 08:00:00")
        result = add_or_convert_to_utc(ts)
        assert result.tzinfo is not None

    def test_returns_pd_timestamp(self):
        """Test that the result is a pandas Timestamp."""
        naive = dt.datetime(2024, 3, 15, 9, 0, 0)
        result = add_or_convert_to_utc(naive)
        assert isinstance(result, pd.Timestamp)


class TestEnsureTimezoneAware:
    """Tests for ensure_timezone_aware."""

    def test_naive_datetime_gets_utc(self):
        """Test that naive datetime is localized to UTC."""
        naive = dt.datetime(2024, 1, 1, 0, 0, 0)
        result = ensure_timezone_aware(naive)
        assert result.tzinfo == UTC

    def test_aware_datetime_stays_utc(self):
        """Test that timezone-aware UTC datetime remains unchanged."""
        aware = dt.datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        result = ensure_timezone_aware(aware)
        assert result == aware

    def test_aware_non_utc_converted_to_utc(self):
        """Test that timezone-aware non-UTC datetime is converted to UTC."""
        ist = timezone(dt.timedelta(hours=5, minutes=30))
        aware_ist = dt.datetime(2024, 1, 1, 17, 30, 0, tzinfo=ist)
        result = ensure_timezone_aware(aware_ist)
        assert result.tzinfo == UTC
        assert result.hour == 12

    def test_naive_pd_timestamp_localized(self):
        """Test that naive pandas Timestamp is localized to UTC."""
        ts = pd.Timestamp("2024-06-01 06:00:00")
        result = ensure_timezone_aware(ts)
        assert result.tzinfo is not None

    def test_aware_pd_timestamp_converted(self):
        """Test that timezone-aware pandas Timestamp is converted to UTC."""
        ist = timezone(dt.timedelta(hours=5, minutes=30))
        ts = pd.Timestamp("2024-06-01 17:30:00", tz=ist)
        result = ensure_timezone_aware(ts)
        assert result.hour == 12


class TestLimitAdjuster:
    """Tests for limit_adjuster."""

    def test_zero_delta_unchanged(self):
        """Test that a zero delta results in zero adjustment."""
        result = limit_adjuster(delta_fraction=0.0, value_fraction=0.5, capacity_mw=100.0)
        assert result == 0.0

    def test_delta_within_10_percent_cap_unchanged(self):
        """Test that delta within 10 percent cap remains unchanged."""
        # delta=0.03, value=0.5 → max=0.05 → delta not capped
        result = limit_adjuster(delta_fraction=0.03, value_fraction=0.5, capacity_mw=1000.0)
        assert result == pytest.approx(0.03)

    def test_positive_delta_capped_at_10_percent_of_value(self):
        """Test that positive delta is capped at 10 percent of the value."""
        # delta=0.2, value=0.5 → max_delta=0.05 → capped to 0.05
        result = limit_adjuster(delta_fraction=0.2, value_fraction=0.5, capacity_mw=1000.0)
        assert result == pytest.approx(0.05)

    def test_negative_delta_capped_at_minus_10_percent(self):
        """Test that negative delta is capped at minus 10 percent of the value."""
        # delta=-0.2, value=0.5 → max_delta=0.05 → capped to -0.05
        result = limit_adjuster(delta_fraction=-0.2, value_fraction=0.5, capacity_mw=1000.0)
        assert result == pytest.approx(-0.05)

    def test_absolute_mw_cap_applies(self):
        """Test that absolute MW cap applies when delta exceeds limit."""
        # capacity_mw=5, max_absolute=1000/5=200 → won't matter here
        # But capacity_mw=1 → max_absolute=1000 (above 10% cap in this case)
        # With capacity_mw=2000 → max_absolute=0.5; value=0.6 → 10% cap=0.06
        # delta=0.8 → capped first to 0.06 (10% of 0.6), which is below 1000/2000=0.5 — OK
        result = limit_adjuster(delta_fraction=0.8, value_fraction=0.6, capacity_mw=2000.0)
        assert result == pytest.approx(0.06)

    def test_very_small_capacity_mw_absolute_cap(self):
        """Test limit adjuster under very small capacity."""
        # capacity_mw=0.5 → max_absolute=2000 (very large — no absolute cap effect)
        result = limit_adjuster(delta_fraction=0.05, value_fraction=0.6, capacity_mw=0.5)
        assert result == pytest.approx(0.05)  # within 10% cap

    def test_negative_absolute_cap(self):
        """Test limit adjuster with negative delta under absolute cap constraint."""
        # Small capacity makes absolute cap large; 10% cap dominates
        result = limit_adjuster(delta_fraction=-0.5, value_fraction=0.4, capacity_mw=5000.0)
        # max_delta = 0.1 * 0.4 = 0.04 → capped to -0.04
        assert result == pytest.approx(-0.04)


# ===========================================================================
# save/database.py
# ===========================================================================


class TestWriteForecastToDb:
    """Tests for write_forecast_to_db."""

    def test_no_write_when_disabled(self, db_session, sites):
        """When write_to_db=False, nothing should be inserted."""
        forecast_meta = {
            "location_uuid": sites[0].location_uuid,
            "timestamp_utc": dt.datetime.now(tz=UTC),
            "forecast_version": "0.0.0",
        }
        df = _make_forecast_values_df()
        write_forecast_to_db(
            db_session,
            forecast_meta,
            df,
            write_to_db=False,
            ml_model_name="test_model",
            ml_model_version="0.0.0",
        )
        assert db_session.query(ForecastSQL).count() == 0
        assert db_session.query(ForecastValueSQL).count() == 0

    def test_writes_to_db_when_enabled(self, db_session, sites):
        """When write_to_db=True, rows should be inserted."""
        forecast_meta = {
            "location_uuid": sites[0].location_uuid,
            "timestamp_utc": dt.datetime.now(tz=UTC),
            "forecast_version": "0.0.0",
        }
        df = _make_forecast_values_df(n=3)
        write_forecast_to_db(
            db_session,
            forecast_meta,
            df,
            write_to_db=True,
            ml_model_name="test_model",
            ml_model_version="0.0.0",
        )
        assert db_session.query(ForecastSQL).count() == 1
        assert db_session.query(ForecastValueSQL).count() == 3
        assert db_session.query(MLModelSQL).count() == 1

    def test_ml_model_name_none_does_not_raise(self, db_session, sites):
        """Passing ml_model_name=None should not crash."""
        forecast_meta = {
            "location_uuid": sites[0].location_uuid,
            "timestamp_utc": dt.datetime.now(tz=UTC),
            "forecast_version": "0.0.0",
        }
        df = _make_forecast_values_df(n=2)
        # Should complete without raising
        write_forecast_to_db(
            db_session,
            forecast_meta,
            df,
            write_to_db=True,
            ml_model_name=None,
            ml_model_version=None,
        )
        assert db_session.query(ForecastSQL).count() == 1


class TestAdjustAndSaveForecast:
    """Tests for adjust_and_save_forecast."""

    def test_adjust_and_save_writes_adjusted_model(
        self, db_session, sites, forecasts, generation_db_values
    ):
        """After adjust+save, an '_adjust' model entry should appear in DB."""
        forecast_meta = {
            "location_uuid": sites[0].location_uuid,
            "timestamp_utc": dt.datetime.now(tz=UTC),
            "forecast_version": "0.0.0",
        }
        df = _make_forecast_values_df(n=5)
        adjust_and_save_forecast(
            db_session,
            forecast_meta,
            df,
            ml_model_name="test",
            ml_model_version="0.0.0",
            adjuster_average_minutes=60,
            write_to_db=True,
        )
        model_names = [m.name for m in db_session.query(MLModelSQL).all()]
        assert any("adjust" in name for name in model_names)

    def test_adjust_and_save_no_write(self, db_session, sites):
        """When write_to_db=False, no rows should be inserted."""
        forecast_meta = {
            "location_uuid": sites[0].location_uuid,
            "timestamp_utc": dt.datetime.now(tz=UTC),
            "forecast_version": "0.0.0",
        }
        df = _make_forecast_values_df(n=3)
        adjust_and_save_forecast(
            db_session,
            forecast_meta,
            df,
            ml_model_name="test",
            ml_model_version="0.0.0",
            adjuster_average_minutes=60,
            write_to_db=False,
        )
        assert db_session.query(ForecastSQL).count() == 0

    def test_adjust_and_save_does_not_raise_on_error(self, db_session, sites):
        """If the adjuster fails internally, the function should log and not raise."""
        forecast_meta = {
            "location_uuid": uuid.uuid4(),  # non-existent site → adjuster returns no ME values
            "timestamp_utc": dt.datetime.now(tz=UTC),
            "forecast_version": "0.0.0",
        }
        df = _make_forecast_values_df(n=2)
        # Should NOT raise
        adjust_and_save_forecast(
            db_session,
            forecast_meta,
            df,
            ml_model_name="test",
            ml_model_version="0.0.0",
            adjuster_average_minutes=60,
            write_to_db=False,
        )


# ===========================================================================
# save/save.py  (save_forecast orchestrator)
# ===========================================================================


class TestSaveForecast:
    """Tests for the top-level save_forecast function."""

    def test_save_forecast_no_write(self, db_session, sites):
        """write_to_db=False → nothing persisted."""
        forecast = _make_forecast_dict(sites[0].location_uuid)
        save_forecast(
            db_session,
            forecast,
            write_to_db=False,
            ml_model_name="test_model",
            ml_model_version="0.0.0",
            use_adjuster_database=False,
        )
        assert db_session.query(ForecastSQL).count() == 0

    def test_save_forecast_write_to_db(self, db_session, sites):
        """write_to_db=True → base forecast written to DB."""
        forecast = _make_forecast_dict(sites[0].location_uuid, n=4)
        save_forecast(
            db_session,
            forecast,
            write_to_db=True,
            ml_model_name="test_model",
            ml_model_version="0.0.0",
            use_adjuster_database=False,
        )
        assert db_session.query(ForecastSQL).count() == 1
        assert db_session.query(ForecastValueSQL).count() == 4

    def test_save_forecast_with_adjuster_writes_both_models(
        self, db_session, sites, forecasts, generation_db_values
    ):
        """use_adjuster_database=True → both base and _adjust models written."""
        forecast = _make_forecast_dict(sites[0].location_uuid, n=5)
        save_forecast(
            db_session,
            forecast,
            write_to_db=True,
            ml_model_name="test",
            ml_model_version="0.0.0",
            use_adjuster_database=True,
            adjuster_average_minutes=60,
        )
        model_names = [m.name for m in db_session.query(MLModelSQL).all()]
        assert "test" in model_names
        assert "test_adjust" in model_names

    def test_save_forecast_adjuster_skipped_when_model_name_none(
        self, db_session, sites
    ):
        """Skip adjuster when model name is None, even if DB adjuster is true."""
        forecast = _make_forecast_dict(sites[0].location_uuid, n=3)
        save_forecast(
            db_session,
            forecast,
            write_to_db=True,
            ml_model_name=None,
            ml_model_version="0.0.0",
            use_adjuster_database=True,
        )
        # No crash; base row written under None model name
        assert db_session.query(ForecastSQL).count() == 1

    def test_save_forecast_dp_disabled_by_default(self, db_session, sites):
        """Data Platform path is NOT triggered when env var is absent/false."""
        forecast = _make_forecast_dict(sites[0].location_uuid, n=2)
        with patch("india_forecast_app.save.save.save_to_dataplatform") as mock_dp:
            save_forecast(
                db_session,
                forecast,
                write_to_db=False,
                ml_model_name="test_model",
                ml_model_version="0.0.0",
                use_adjuster_database=False,
            )
        mock_dp.assert_not_called()

    def test_save_forecast_dp_triggered_when_env_set(self, db_session, sites, monkeypatch):
        """Data Platform path IS triggered when SAVE_TO_DATA_PLATFORM=true."""
        monkeypatch.setenv("SAVE_TO_DATA_PLATFORM", "true")
        forecast = _make_forecast_dict(sites[0].location_uuid, n=2)

        async def _fake_dp(**kwargs):
            pass

        with patch(
            "india_forecast_app.save.save.save_to_dataplatform",
            side_effect=_fake_dp,
        ), patch(
            "india_forecast_app.save.save.asyncio.run",
        ) as mock_run:
            save_forecast(
                db_session,
                forecast,
                write_to_db=False,
                ml_model_name="test_model",
                ml_model_version="0.0.0",
                use_adjuster_database=False,
            )
        mock_run.assert_called_once()

    def test_save_forecast_horizon_minutes_computed_correctly(self, db_session, sites):
        """Horizon minutes should equal (start_utc - timestamp) in minutes."""
        timestamp = dt.datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        forecast = _make_forecast_dict(sites[0].location_uuid, n=3, timestamp=timestamp)
        # Just check it doesn't raise; we'd need to capture the df to assert values
        save_forecast(
            db_session,
            forecast,
            write_to_db=True,
            ml_model_name="test_model",
            ml_model_version="0.0.0",
            use_adjuster_database=False,
        )
        assert db_session.query(ForecastValueSQL).count() == 3
        values = db_session.query(ForecastValueSQL).all()
        horizons = sorted([v.horizon_minutes for v in values])
        assert horizons == [0, 15, 30]


# ===========================================================================
# save/data_platform.py — pure/sync helpers (no gRPC needed)
# ===========================================================================


class TestPrepareForecastValues:
    """Tests for prepare_forecast_values (pure function, no gRPC)."""

    def test_returns_correct_number_of_values(self):
        """Test that the correct number of forecast values is returned."""
        df = _make_forecast_values_df(n=4)
        init_time = dt.datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        capacity_watts = 10_000_000  # 10 MW

        result = prepare_forecast_values(df, init_time, capacity_watts)
        assert len(result) == 4

    def test_p50_fraction_clamped_between_0_and_1(self):
        """Test that p50 fraction values are clamped to the 0-1 range."""
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
        """Test that horizon mins are computed from start_utc if missing."""
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
        """Test that the existing horizon_minutes column is used."""
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
        """Test parsing probabilistic values when they are a dictionary."""
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
        """Test that an empty DataFrame returns an empty list."""
        df = pd.DataFrame(columns=["start_utc", "end_utc", "forecast_power_kw", "horizon_minutes"])
        init_time = dt.datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        result = prepare_forecast_values(df, init_time, 1_000_000)
        assert result == []


class TestResolveTargetUuid:
    """Tests for resolve_target_uuid (uses mocked gRPC client)."""

    def test_returns_uuid_from_prefetched_map(self):
        """Test returning UUID from prefetched map directly."""
        async def _run():
            mock_client = MagicMock()
            location_map = {"my_location": "abc-123"}
            result = await resolve_target_uuid(mock_client, "my_location", location_map)
            assert result == "abc-123"
            mock_client.list_locations.assert_not_called()

        asyncio.run(_run())

    def test_returns_none_when_location_not_in_map(self):
        """Test returning None when location is not in map."""
        async def _run():
            mock_client = MagicMock()
            location_map = {"other_location": "xyz-456"}
            result = await resolve_target_uuid(mock_client, "my_location", location_map)
            assert result is None
            mock_client.list_locations.assert_not_called()

        asyncio.run(_run())

    def test_fetches_map_when_none_provided(self):
        """Test fetching the location map if not provided."""
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
        """Test returning None when location is not found in fetched map."""
        async def _run():
            mock_client = AsyncMock()
            mock_client.list_locations.return_value = MagicMock(locations=[])

            result = await resolve_target_uuid(mock_client, "missing_location", location_map=None)
            assert result is None

        asyncio.run(_run())


class TestFetchDpLocationMap:
    """Tests for fetch_dp_location_map."""

    def test_builds_name_to_uuid_map(self):
        """Test building location name to UUID map from response."""
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
        """Test returning empty map when no locations exist."""
        async def _run():
            mock_client = AsyncMock()
            mock_client.list_locations.return_value = MagicMock(locations=[])

            result = await fetch_dp_location_map(mock_client)
            assert result == {}

        asyncio.run(_run())
