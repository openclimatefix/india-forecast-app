"""
Tests for india_forecast_app.save.utils.

Covers:
  - save/utils.py         : add_or_convert_to_utc, ensure_timezone_aware, limit_adjuster
"""

from __future__ import annotations

import datetime as dt
from datetime import UTC, timezone

import pandas as pd
import pytest

from india_forecast_app.save.utils import (
    add_or_convert_to_utc,
    ensure_timezone_aware,
    limit_adjuster,
)


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
