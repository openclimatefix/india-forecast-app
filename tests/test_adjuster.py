from india_forecast_app.adjuster import get_me_values, adjust_forecast_with_adjuster
import pandas as pd

from datetime import datetime


def test_get_me_values_no_values(db_session, sites):

    me_df = get_me_values(db_session, 10, site_uuid=sites[0].site_uuid, ml_model_name="test")

    assert len(me_df) == 0


def test_get_me_values(db_session, sites, generation_db_values, forecasts):

    hour = pd.Timestamp(datetime.now()).hour
    me_df = get_me_values(db_session, hour, site_uuid=sites[0].site_uuid, ml_model_name="test")

    assert len(me_df) != 0


def test_get_me_values_no_generation(db_session, sites, forecasts):

    hour = pd.Timestamp(datetime.now()).hour
    me_df = get_me_values(db_session, hour, site_uuid=sites[0].site_uuid, ml_model_name="test")

    assert len(me_df) == 0


def test_get_me_values_no_forecasts(db_session, sites, generation_db_values):

    hour = pd.Timestamp(datetime.now()).hour
    me_df = get_me_values(db_session, hour, site_uuid=sites[0].site_uuid, ml_model_name="test")

    assert len(me_df) == 0


def test_adjust_forecast_with_adjuster(db_session, sites, generation_db_values, forecasts):
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
