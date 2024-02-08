"""
Tests for functions in app.py
"""
import datetime as dt
import uuid

import pytest
from pvsite_datamodel.read import get_all_sites
from pvsite_datamodel.sqlmodels import ForecastSQL, ForecastValueSQL

from india_forecast_app.app import app, get_model, get_sites, run_model, save_forecast
from india_forecast_app.models.dummy import DummyModel
from india_forecast_app.models.pvnet.model import PVNetModel

from ._utils import run_click_script


def test_get_sites(db_session):
    """Test for correct site ids"""

    sites = get_sites(db_session)
    sites = sorted(sites, key=lambda s: s.client_site_id)
    
    assert len(sites) == 2
    for site in sites:
        assert isinstance(site.site_uuid, uuid.UUID)
        assert sites[0].asset_type.name == "pv"
        assert sites[1].asset_type.name == "wind"


@pytest.mark.parametrize("asset_type", ["pv", "wind"])
def test_get_model(asset_type, nwp_data, caplog):
    """Test for getting valid model"""

    caplog.set_level('INFO')
    model = get_model(asset_type, timestamp=dt.datetime.now(tz=dt.UTC))
    
    assert hasattr(model, 'version')
    assert isinstance(model.version, str)
    assert hasattr(model, 'predict')


@pytest.mark.skip(reason="Temporarily disabled while integrating Windnet")
@pytest.mark.parametrize("asset_type", ["pv", "wind"])
def test_run_model(db_session, asset_type, nwp_data):
    """Test for running PV and wind models"""

    model = PVNetModel if asset_type == "wind" else DummyModel

    forecast = run_model(
        model=model(asset_type, timestamp=dt.datetime.now(tz=dt)),
        site_id=str(uuid.uuid4()),
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


@pytest.mark.skip(reason="Temporarily disabled while integrating Windnet")
@pytest.mark.parametrize("write_to_db", [True, False])
def test_app(write_to_db, db_session, nwp_data):
    """Test for running app from command line"""

    init_n_forecasts = db_session.query(ForecastSQL).count()
    init_n_forecast_values = db_session.query(ForecastValueSQL).count()

    args = ["--date", dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d-%H-%M")]
    if write_to_db:
        args.append('--write-to-db')

    result = run_click_script(app, args)
    assert result.exit_code == 0

    if write_to_db:
        assert db_session.query(ForecastSQL).count() == init_n_forecasts + 2
        assert db_session.query(ForecastValueSQL).count() == init_n_forecast_values + (2 * 192)
    else:
        assert db_session.query(ForecastSQL).count() == init_n_forecasts
        assert db_session.query(ForecastValueSQL).count() == init_n_forecast_values
