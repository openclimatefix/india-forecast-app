"""
Tests for functions in app.py
"""
import datetime as dt
import uuid

import pytest
from pvsite_datamodel.read import get_all_sites
from pvsite_datamodel.sqlmodels import ForecastSQL, ForecastValueSQL, SiteSQL

from india_forecast_app.app import app, get_model, get_site_ids, run_model, save_forecast
from india_forecast_app.model import DummyModel

from ._utils import run_click_script


def test_get_site_ids(db_session):
    """Test for correct site ids"""

    site_ids = get_site_ids(db_session)
    
    assert len(site_ids) == 3
    for site_id in site_ids:
        assert isinstance(site_id, uuid.UUID)


def test_get_model():
    """Test for getting valid model"""

    model = get_model()
    
    assert hasattr(model, 'version')
    assert isinstance(model.version, str)
    assert hasattr(model, 'predict')


def test_run_model(db_session):
    """Test for running model"""

    site = db_session.query(SiteSQL).first()
    model = DummyModel()
    
    forecast = run_model(
        model=model, 
        site_id=site.site_uuid, 
        timestamp=dt.datetime.now(tz=dt.UTC)
    )

    assert isinstance(forecast, list)
    assert len(forecast) == 192 # value for every 15mins over 2 days
    assert all([isinstance(value["start_utc"], dt.datetime) for value in forecast])
    assert all([isinstance(value["end_utc"], dt.datetime) for value in forecast])
    assert all([isinstance(value["forecast_power_kw"], int) for value in forecast])


def test_save_forecast(db_session, forecast_values):
    """Test for saving forecast"""

    site = get_all_sites(db_session)[0]

    forecast = {
        "meta": {
            "site_id": site.site_uuid,
            "version": "0.0.0",
            "timestamp": dt.datetime.now(tz=dt.UTC)
        },
        "values": forecast_values,
    }

    save_forecast(db_session, forecast, write_to_db=True)

    assert db_session.query(ForecastSQL).count() == 1
    assert db_session.query(ForecastValueSQL).count() == 10


@pytest.mark.parametrize("write_to_db", [True, False])
def test_app(write_to_db, db_session):
    """Test for running app from command line"""

    init_n_forecasts = db_session.query(ForecastSQL).count()
    init_n_forecast_values = db_session.query(ForecastValueSQL).count()

    args = ["--date", dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d-%H-%M")]
    if write_to_db:
        args.append('--write-to-db')

    result = run_click_script(app, args)
    assert result.exit_code == 0

    if write_to_db:
        assert db_session.query(ForecastSQL).count() == init_n_forecasts + 3
        assert db_session.query(ForecastValueSQL).count() == init_n_forecast_values + (3 * 192)
    else:
        assert db_session.query(ForecastSQL).count() == init_n_forecasts
        assert db_session.query(ForecastValueSQL).count() == init_n_forecast_values
