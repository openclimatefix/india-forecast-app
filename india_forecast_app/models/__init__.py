"""Available models for India forecast"""

from .dummy import DummyModel
from .pvnet.model import PVNetModel
from .pydantic_models import get_all_models

__all__ = ['DummyModel', 'PVNetModel']