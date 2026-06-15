"""Shared utility helpers used across the save subpackage."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd


def add_or_convert_to_utc(timestamp: object) -> pd.Timestamp:
    """Ensure a timestamp is a timezone-aware UTC pd.Timestamp."""
    ts = pd.Timestamp(timestamp)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    return ts


def ensure_timezone_aware(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware and always in UTC."""
    if isinstance(dt, pd.Timestamp):
        return dt.tz_localize("UTC") if dt.tz is None else dt.tz_convert("UTC")
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def limit_adjuster(delta_fraction: float, value_fraction: float, capacity_mw: float) -> float:
    """Limit the adjuster to 10% of forecast and max 1000 MW."""
    # limit adjusted fractions to 10% of fv.p50_fraction
    max_delta = 0.1 * value_fraction
    if delta_fraction > max_delta:
        delta_fraction = max_delta
    elif delta_fraction < -max_delta:
        delta_fraction = -max_delta

    # limit adjust to 1000 MW
    max_delta_absolute = 1000.0 / capacity_mw
    if delta_fraction > max_delta_absolute:
        delta_fraction = max_delta_absolute
    elif delta_fraction < -max_delta_absolute:
        delta_fraction = -max_delta_absolute

    return delta_fraction
