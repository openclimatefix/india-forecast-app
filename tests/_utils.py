"""Testing utils."""

import datetime as dt
from datetime import UTC

import pandas as pd
from click.testing import CliRunner


def run_click_script(func, args: list[str], catch_exceptions: bool = False):
    """Util to test click scripts while showing the stdout."""

    runner = CliRunner()

    # We catch the exception here no matter what, but we'll reraise later if need be.
    result = runner.invoke(func, args, catch_exceptions=True)

    # Without this the output to stdout/stderr is grabbed by click's test runner.
    # print(result.output)

    # In case of an exception, raise it so that the test fails with the exception.
    if result.exception and not catch_exceptions:
        raise result.exception

    return result


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


