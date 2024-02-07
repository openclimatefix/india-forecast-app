"""Available models for India forecast"""

from .dummy import DummyModel
from .pvnet.model import PVNetModel

__all__ = ['DummyModel', 'PVNetModel']