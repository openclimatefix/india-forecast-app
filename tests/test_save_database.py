"""
Tests for india_forecast_app.save.database.

Covers:
  - save/database.py      : write_forecast_to_db, adjust_and_save_forecast
"""

from __future__ import annotations

import datetime as dt
import uuid
from datetime import UTC

import pytest
from pvsite_datamodel.sqlmodels import ForecastSQL, ForecastValueSQL, MLModelSQL

from india_forecast_app.save.database import adjust_and_save_forecast, write_forecast_to_db

from ._utils import _make_forecast_values_df


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

    def test_adjust_and_save_raises_on_error(self, db_session, sites):
        """If the adjuster fails internally (e.g. non-existent site), the function should raise."""
        forecast_meta = {
            "location_uuid": uuid.uuid4(),  # non-existent site → adjuster raises KeyError
            "timestamp_utc": dt.datetime.now(tz=UTC),
            "forecast_version": "0.0.0",
        }
        df = _make_forecast_values_df(n=2)
        # Should raise KeyError
        with pytest.raises(KeyError):
            adjust_and_save_forecast(
                db_session,
                forecast_meta,
                df,
                ml_model_name="test",
                ml_model_version="0.0.0",
                adjuster_average_minutes=60,
                write_to_db=False,
            )
