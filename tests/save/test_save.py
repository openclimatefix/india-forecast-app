"""
Tests for india_forecast_app.save.save (the save_forecast orchestrator).

Tests cover:
1. write_to_db=False: nothing is persisted to the database
2. write_to_db=True: base forecast is written to the database
3. use_adjuster_database=True: both base and _adjust models are written
4. ml_model_name=None: adjuster is skipped even when DB adjuster is True
5. SAVE_TO_DATA_PLATFORM absent/false: Data Platform path is not triggered
6. SAVE_TO_DATA_PLATFORM=true: Data Platform path is triggered via asyncio.run
7. horizon_minutes: computed correctly as (start_utc - timestamp) in minutes
"""

from __future__ import annotations

import datetime as dt
from datetime import UTC
from unittest.mock import patch

from pvsite_datamodel.sqlmodels import ForecastSQL, ForecastValueSQL, MLModelSQL

from india_forecast_app.save.save import save_forecast
from tests._utils import _make_forecast_dict


class TestSaveForecast:
    """Tests for the top-level save_forecast function."""

    def test_save_forecast_no_write(self, db_session, sites):
        """[1] write_to_db=False → nothing persisted."""
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
        """[2] write_to_db=True → base forecast written to DB."""
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
        """[3] use_adjuster_database=True → both base and _adjust models written."""
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
        """[4] Skip adjuster when model name is None, even if DB adjuster is true."""
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
        """[5] Data Platform path is NOT triggered when env var is absent/false."""
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
        """[6] Data Platform path IS triggered when SAVE_TO_DATA_PLATFORM=true."""
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
        """[7] Horizon minutes should equal (start_utc - timestamp) in minutes."""
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
