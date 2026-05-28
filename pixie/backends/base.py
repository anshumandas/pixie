"""Compute-backend abstraction.

Every heavy / optional operation is requested through a ``ComputeBackend``.
The same operation can run locally or on a remote GPU; backends report their
availability and a per-operation **cost estimate** so the UI can show the price
next to each option at the moment the user chooses.
"""
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


class Operation(str, enum.Enum):
    """Operations that may be routed to a backend."""
    SEGMENT = "segment"             # click-to-select (usually local)
    ERASE = "erase"                 # remove object + fill (LaMa / cv2.inpaint)
    GENERATIVE_FILL = "generative_fill"   # SD inpaint — invent new content
    STYLIZE_NEURAL = "stylize_neural"     # SD img2img / neural style transfer

    @property
    def label(self) -> str:
        return {
            Operation.SEGMENT: "Select object",
            Operation.ERASE: "Erase object",
            Operation.GENERATIVE_FILL: "Generative fill",
            Operation.STYLIZE_NEURAL: "Neural stylize",
        }[self]


@dataclass(frozen=True)
class CostEstimate:
    """An approximate price for one operation on a backend."""
    is_free: bool
    amount_usd: Optional[float] = None   # per-operation, when known
    unit: str = "operation"
    note: str = ""

    def display(self) -> str:
        if self.is_free:
            base = "Free"
        elif self.amount_usd is not None:
            base = f"~${self.amount_usd:.3f}/{self.unit}".rstrip("0").rstrip(".")
            # tidy: ~$0.005/operation -> ~$0.005/op
            base = base.replace("/operation", "/op")
        else:
            base = "Variable"
        return f"{base} — {self.note}" if self.note else base


@dataclass(frozen=True)
class Availability:
    ready: bool
    reason: str = ""   # why not, if ready is False (e.g. "set REPLICATE_API_TOKEN")


class ComputeBackend(ABC):
    """Base class for a place where operations can run."""

    #: stable identifier used in config/UI
    id: str = "base"
    #: human-facing name
    display_name: str = "Backend"
    #: 'local' | 'hosted-api' | 'self-hosted'
    kind: str = "local"
    #: whether images leave the user's device when this backend is used
    sends_data_offsite: bool = False

    @abstractmethod
    def availability(self) -> Availability:
        """Whether this backend is usable right now (keys present, etc.)."""

    @abstractmethod
    def supports(self, op: Operation) -> bool:
        """Whether this backend can perform ``op`` at all."""

    @abstractmethod
    def estimate_cost(self, op: Operation) -> CostEstimate:
        """Approximate price for one ``op`` — shown to the user at choose-time."""

    def run(self, op: Operation, **payload):
        """Execute ``op``. Subclasses override per supported operation."""
        raise NotImplementedError(
            f"{self.display_name} cannot run {op.label} yet."
        )

    # Convenience for the picker UI ------------------------------------------
    def option_line(self, op: Operation) -> str:
        avail = self.availability()
        cost = self.estimate_cost(op).display() if self.supports(op) else "n/a"
        privacy = " · uploads image" if self.sends_data_offsite else " · on-device"
        status = "" if avail.ready else f"  (unavailable: {avail.reason})"
        return f"{self.display_name} — {cost}{privacy}{status}"
