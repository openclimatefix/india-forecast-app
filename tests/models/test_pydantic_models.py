""" Test for getting all ml models"""
from india_forecast_app.models.pydantic_models import get_all_models


def test_get_all_models():
    """Test for getting all models"""
    models = get_all_models()
    assert len(models.models) == 4


def test_get_all_models_client():
    """Test for getting all models for a specific client"""
    models = get_all_models(client_abbreviation="dummy")
    assert len(models.models) == 1
