"""
Tests for functions in app.py
"""

import datetime as dt
import uuid

from pvsite_datamodel.sqlmodels import SiteSQL

from india_forecast_app.app import get_model, get_site_ids, run_model
from india_forecast_app.model import DummyModel


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
    
    # TODO better assertions against forecast
    assert isinstance(forecast, list)

def test_save_forecast(db_session):
    """Test for saving forecast"""
    
    # TODO test for successful and unsuccessful saving of forecast
    pass


def test_app():
    """Test for running app from command line"""
    
    # TODO test click app
    pass
