"""
Tests for functions in app.py
"""

import datetime as dt
import multiprocessing as mp
import os
import uuid

import pytest
from pvsite_datamodel.sqlmodels import ForecastSQL, ForecastValueSQL, MLModelSQL, SiteAssetType

from india_forecast_app.app import (
    app,
    app_run,
    get_generation_data,
    get_model,
    get_sites,
    run_model,
    save_forecast,
)
from india_forecast_app.models.dummy import DummyModel
from india_forecast_app.models.pvnet.model import PVNetModel
from india_forecast_app.models.pydantic_models import get_all_models

from ._utils import run_click_script

mp.set_start_method("spawn", force=True)


def test_get_sites(db_session, sites):
    """Test for correct site ids"""

    sites = get_sites(db_session)
    sites = sorted(sites, key=lambda s: s.client_site_id)

    assert len(sites) == 2
    for site in sites:
        assert isinstance(site.site_uuid, uuid.UUID)
        assert sites[0].asset_type.name == "pv"
        assert sites[1].asset_type.name == "wind"


def test_get_generation_data(db_session, sites, generation_db_values, init_timestamp):
    """Test for correct generation data"""

    # Test only checks for wind data as solar data not ready yet
    gen_sites = [s for s in sites if s.asset_type == SiteAssetType.wind]  # 1 site
    gen_data = get_generation_data(db_session, gen_sites, timestamp=init_timestamp)
    gen_df, gen_meta = gen_data["data"], gen_data["metadata"]

    # Check for 5 (non-null) generation values
    assert len(gen_df) == 5
    assert not gen_df["0"].isnull().any()  # 0 is the ml_id/system_id of the wind site

    # Check first and last timestamps are correct
    assert gen_df.index[0] == init_timestamp - dt.timedelta(hours=1)
    assert gen_df.index[-1] == init_timestamp

    # Check for expected metadata
    assert len(gen_meta) == 1


@pytest.mark.parametrize("asset_type", ["pv", "wind"])
def test_get_model(
    db_session, asset_type, sites, nwp_data, nwp_gfs_data, generation_db_values, init_timestamp
):
    """Test for getting valid model"""

    all_models = get_all_models()
    ml_model = [model for model in all_models.models if model.asset_type == asset_type][0]
    gen_sites = [
        s
        for s in sites
        if s.asset_type.name == asset_type and s.client_site_name == "test_site_ruvnl"
    ]
    gen_data = get_generation_data(db_session, gen_sites, timestamp=init_timestamp)
    model = get_model(
        asset_type,
        timestamp=init_timestamp,
        generation_data=gen_data,
        hf_version=ml_model.version,
        hf_repo=ml_model.id,
        name="test",
    )

    assert hasattr(model, "version")
    assert isinstance(model.version, str)
    assert hasattr(model, "predict")


@pytest.mark.parametrize("asset_type", ["pv", "wind"])
def test_run_model(
    db_session, asset_type, sites, nwp_data, nwp_gfs_data, generation_db_values, init_timestamp
):
    """Test for running PV and wind models"""

    all_models = get_all_models()
    ml_model = [model for model in all_models.models if model.asset_type == asset_type][0]
    gen_sites = [
        s
        for s in sites
        if s.asset_type.name == asset_type and s.client_site_name == "test_site_ruvnl"
    ]
    gen_data = get_generation_data(db_session, sites=gen_sites, timestamp=init_timestamp)
    model_cls = PVNetModel if asset_type == "wind" else DummyModel
    model = model_cls(
        asset_type,
        timestamp=init_timestamp,
        generation_data=gen_data,
        hf_version=ml_model.version,
        hf_repo=ml_model.id,
        name="test",
    )
    forecast = run_model(model=model, site_id=str(uuid.uuid4()), timestamp=init_timestamp)

    assert isinstance(forecast, list)
    assert len(forecast) == 192  # value for every 15mins over 2 days
    assert all([isinstance(value["start_utc"], dt.datetime) for value in forecast])
    assert all([isinstance(value["end_utc"], dt.datetime) for value in forecast])
    assert all([isinstance(value["forecast_power_kw"], int) for value in forecast])


def test_save_forecast(db_session, sites, forecast_values):
    """Test for saving forecast"""

    site = sites[0]

    forecast = {
        "meta": {
            "site_id": site.site_uuid,
            "version": "0.0.0",
            "timestamp": dt.datetime.now(tz=dt.UTC),
        },
        "values": forecast_values,
    }

    save_forecast(
        db_session, forecast, write_to_db=True, ml_model_name="test", ml_model_version="0.0.0"
    )

    assert db_session.query(ForecastSQL).count() == 2
    assert db_session.query(ForecastValueSQL).count() == 10 * 2
    assert db_session.query(MLModelSQL).count() == 2


@pytest.mark.parametrize("write_to_db", [True, False])
def test_app(
    write_to_db, db_session, sites, nwp_data, nwp_gfs_data, nwp_mo_global_data, generation_db_values
):
    """Test for running app from command line"""

    init_n_forecasts = db_session.query(ForecastSQL).count()
    init_n_forecast_values = db_session.query(ForecastValueSQL).count()

    args = ["--date", dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d-%H-%M")]
    if write_to_db:
        args.append("--write-to-db")

    result = run_click_script(app, args)
    assert result.exit_code == 0

    if write_to_db:
        assert db_session.query(ForecastSQL).count() == init_n_forecasts + 5 * 2
        assert db_session.query(ForecastValueSQL).count() == init_n_forecast_values + (5 * 2 * 192)
        assert db_session.query(MLModelSQL).count() == 5 * 2
    else:
        assert db_session.query(ForecastSQL).count() == init_n_forecasts
        assert db_session.query(ForecastValueSQL).count() == init_n_forecast_values


def test_app_no_pv_data(
    db_session, sites, nwp_data, nwp_gfs_data, nwp_mo_global_data, generation_db_values_only_wind
):
    """Test for running app from command line"""

    init_n_forecasts = db_session.query(ForecastSQL).count()
    init_n_forecast_values = db_session.query(ForecastValueSQL).count()

    args = ["--date", dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d-%H-%M")]
    args.append("--write-to-db")

    result = run_click_script(app, args)
    assert result.exit_code == 0

    assert db_session.query(ForecastSQL).count() == init_n_forecasts + 2 * 5
    assert db_session.query(ForecastValueSQL).count() == init_n_forecast_values + (2 * 5 * 192)


@pytest.mark.requires_hf_token
def test_app_client_ad(
    db_session,
    sites,
    nwp_data,
    nwp_gfs_data,
    satellite_data,
    use_satellite,
    generation_db_values,
    client_ad,
):
    """Test for running app from command line"""

    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    # Skip the test if the token is not available
    if hf_token is None:
        pytest.skip("Hugging Face token not set in environment variables, skipping test.")

    app_run(timestamp=None)
