"""Attention backend selection for the Wan inference frontend."""

from .dense import DenseBackend
from .sparse import MatrixSparseBackend, SparseConfig

__all__ = ["DenseBackend", "MatrixSparseBackend", "SparseConfig"]
