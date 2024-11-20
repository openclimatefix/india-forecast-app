""" Test for adjuster.py """
from datetime import datetime

import pandas as pd

from india_forecast_app.adjuster import adjust_forecast_with_adjuster, get_me_values


def test_get_me_values_no_values(db_session, sites):
    """Check no ME results are found with no forecast or generation values"""

    me_df = get_me_values(db_session, 10, site_uuid=sites[0].site_uuid, ml_model_name="test")

    assert len(me_df) == 0


def test_get_me_values(db_session, sites, generation_db_values, forecasts):
    """Check ME results are found"""

    hour = pd.Timestamp(datetime.now()).hour
    me_df = get_me_values(db_session, hour, site_uuid=sites[0].site_uuid, ml_model_name="test")

    assert len(me_df) != 0
    assert len(me_df) == 96
    assert me_df["me_kw"].sum() != 0
    assert me_df["horizon_minutes"][0] == 0
    assert me_df["horizon_minutes"][1] == 15


def test_get_me_values_no_generation(db_session, sites, forecasts):
    """Check no ME results are found with no generation values"""

    hour = pd.Timestamp(datetime.now()).hour
    me_df = get_me_values(db_session, hour, site_uuid=sites[0].site_uuid, ml_model_name="test")

    assert len(me_df) == 0


def test_get_me_values_no_forecasts(db_session, sites, generation_db_values):
    """Check no ME results are found with no generation values"""

    hour = pd.Timestamp(datetime.now()).hour
    me_df = get_me_values(db_session, hour, site_uuid=sites[0].site_uuid, ml_model_name="test")

    assert len(me_df) == 0


def test_adjust_forecast_with_adjuster(db_session, sites, generation_db_values, forecasts):
    """Check forecast gets adjuster"""
    forecast_meta = {"timestamp_utc": datetime.now(), "site_uuid": sites[0].site_uuid}
    forecast_values_df = pd.DataFrame(
        {
            "forecast_power_kw": [1, 2, 3, 4, 5],
            "horizon_minutes": [15, 30, 45, 60, 1200],
        }
    )

    forecast_values_df = adjust_forecast_with_adjuster(
        db_session, forecast_meta, forecast_values_df, ml_model_name="test"
    )

    assert len(forecast_values_df) == 5
    assert forecast_values_df["forecast_power_kw"][0:4].sum() == 10
    assert forecast_values_df["forecast_power_kw"][4] != 5
    # note the way the tests are setup, only the horizon_minutes=1200 has some ME values


def test_adjust_forecast_with_adjuster_no_values(db_session, sites):
    """Check forecast doesnt adjuster, no me values"""
    forecast_meta = {"timestamp_utc": datetime.now(), "site_uuid": sites[0].site_uuid}
    forecast_values_df = pd.DataFrame(
        {
            "forecast_power_kw": [1, 2, 3, 4, 5],
            "horizon_minutes": [15, 30, 45, 60, 1200],
        }
    )

    forecast_values_df = adjust_forecast_with_adjuster(
        db_session, forecast_meta, forecast_values_df, ml_model_name="test"
    )

    assert len(forecast_values_df) == 5
    assert forecast_values_df["forecast_power_kw"].sum() == 15
