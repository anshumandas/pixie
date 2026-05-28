"""Builds the list of available backends and the per-operation cost table."""
from __future__ import annotations

from typing import Dict, List, Optional

from ..config import AppConfig
from .base import ComputeBackend, Operation
from .cloud_backends import ALL_CLOUD_BACKENDS
from .local_backend import LocalBackend


class BackendRegistry:
    """Holds every backend and answers 'who can do X, and what does it cost?'."""

    def __init__(self, backends: List[ComputeBackend], default_id: str = "local"):
        self._backends = backends
        self._by_id = {b.id: b for b in backends}
        self.default_id = default_id if default_id in self._by_id else backends[0].id

    @property
    def backends(self) -> List[ComputeBackend]:
        return list(self._backends)

    def get(self, backend_id: str) -> Optional[ComputeBackend]:
        return self._by_id.get(backend_id)

    def options_for(self, op: Operation, only_available: bool = False) -> List[ComputeBackend]:
        """Backends that can perform ``op`` (optionally only ready ones)."""
        out = [b for b in self._backends if b.supports(op)]
        if only_available:
            out = [b for b in out if b.availability().ready]
        return out

    def cost_table(self, op: Operation) -> List[Dict[str, str]]:
        """Rows for the picker UI: id, label, cost, privacy, status."""
        rows: List[Dict[str, str]] = []
        for b in self.options_for(op):
            avail = b.availability()
            rows.append(
                {
                    "id": b.id,
                    "name": b.display_name,
                    "cost": b.estimate_cost(op).display(),
                    "privacy": "uploads image" if b.sends_data_offsite else "on-device",
                    "ready": "yes" if avail.ready else f"no — {avail.reason}",
                }
            )
        return rows

    def best_default_for(self, op: Operation) -> ComputeBackend:
        """Prefer local if it can do it; else the first available cloud option;
        else the first that supports it (so the UI can show how to enable it)."""
        local = self.get("local")
        if local is not None and local.supports(op) and local.availability().ready:
            return local
        avail = self.options_for(op, only_available=True)
        if avail:
            return avail[0]
        any_support = self.options_for(op)
        return any_support[0] if any_support else local  # type: ignore[return-value]


def build_registry(config: AppConfig) -> BackendRegistry:
    """Construct the registry from the auto-detected config."""
    caps = config.capabilities
    local = LocalBackend(
        generative_local=(caps.generative_fill == "local"),
        neural_local=(caps.stylize_neural == "local"),
    )
    backends: List[ComputeBackend] = [local]
    backends.extend(cls() for cls in ALL_CLOUD_BACKENDS)
    return BackendRegistry(backends, default_id=config.default_backend_id)
