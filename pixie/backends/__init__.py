"""Pluggable compute backends (local + cloud) with runtime cost reporting."""
from .base import Availability, ComputeBackend, CostEstimate, Operation
from .local_backend import LocalBackend
from .registry import BackendRegistry, build_registry

__all__ = [
    "Availability",
    "ComputeBackend",
    "CostEstimate",
    "Operation",
    "LocalBackend",
    "BackendRegistry",
    "build_registry",
]
