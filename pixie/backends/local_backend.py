"""The local compute backend — runs on the user's own machine, always free."""
from __future__ import annotations

from .base import Availability, ComputeBackend, CostEstimate, Operation


class LocalBackend(ComputeBackend):
    id = "local"
    display_name = "This computer (local)"
    kind = "local"
    sends_data_offsite = False

    def __init__(self, generative_local: bool = False, neural_local: bool = False):
        # Whether the machine can run the GPU-hungry ops locally. Light ops
        # (segment / erase / basic stylize) always run locally via fallbacks.
        self._generative_local = generative_local
        self._neural_local = neural_local

    def availability(self) -> Availability:
        return Availability(ready=True)

    def supports(self, op: Operation) -> bool:
        if op in (Operation.SEGMENT, Operation.ERASE):
            return True  # SAM/LaMa if present, else OpenCV fallback
        if op is Operation.GENERATIVE_FILL:
            return self._generative_local
        if op is Operation.STYLIZE_NEURAL:
            return self._neural_local
        return False

    def estimate_cost(self, op: Operation) -> CostEstimate:
        note = "uses your CPU/GPU"
        return CostEstimate(is_free=True, note=note)

    # Execution for the local backend is performed directly by the editor
    # engine (it owns the loaded models); see app/engine.py. ``run`` is here
    # only to satisfy the interface.
    def run(self, op: Operation, **payload):  # pragma: no cover - see engine
        raise RuntimeError("Local operations are dispatched by the editor engine.")
