""" A pydantic model for the ML models"""

import fsspec
from typing import List, Optional
from pydantic import BaseModel, Field
from pyaml_env import parse_config


class Model(BaseModel):
    name: str = Field(..., title="Model Name", description="The name of the model")
    type: Optional[str] = Field("pvnet", title="Model Type", description="The type of model")
    id: str = Field(
        ..., title="Model ID", description="The ID of the model, this what repo to load from in HF "
    )
    version: str = Field(
        ...,
        title="Model Version",
        description="The version of the model, this is what git version to load from in HF",
    )
    client: str = Field(
        "ruvnl", title="Client Abbreviation", description="The name of the client that the model is for"
    )
    asset_type: str = Field(
        "pv", title="Asset Type", description="The type of asset the model is for (pv or wind)"
    )


class Models(BaseModel):
    models: List[Model] = Field(
        ..., title="Models", description="A list of models to use for the forecast"
    )


def get_all_models(client_abbreviation: Optional[str] = None):
    """
    Returns all the models for a given client
    """

    # load models from yaml file
    import os
    filename = os.path.dirname(os.path.abspath(__file__)) + '/all_models.yaml'

    with fsspec.open(filename, mode="r") as stream:
        models = parse_config(data=stream)
        models = Models(**models)

    if client_abbreviation:
        models.models = [model for model in models.models if model.client == client_abbreviation]

    return models

