"""Auto-configuration derived from the detected hardware.

``auto_configure(profile)`` turns a :class:`~app.system_check.SystemProfile`
into an :class:`AppConfig`: which selection model to load, which downscale
budget to use, and which heavy operations can run locally vs. need a cloud
backend. This runs on every launch, so the configuration tracks the machine.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .system_check import SystemProfile, detect_system

log = logging.getLogger(__name__)

# Approximate VRAM needed to run Stable Diffusion 1.5 inpaint/img2img locally.
_SD_MIN_VRAM_GB = 4.0
# VRAM above which we can comfortably load the larger, sharper SAM checkpoints.
_SAM_LARGE_MIN_VRAM_GB = 8.0


@dataclass
class Capabilities:
    """What the app can actually do right now, and how.

    Each value is one of: 'local' (runs on this machine), 'cloud' (needs a
    cloud backend), or 'fallback' (works via a built-in OpenCV method that
    needs no weights). The UI uses this to enable/label features honestly.
    """
    selection: str = "fallback"          # SAM if available, else GrabCut
    background_removal: str = "fallback"  # uses the selection mask
    erase_object: str = "fallback"        # LaMa if available, else cv2.inpaint
    generative_fill: str = "cloud"        # SD inpaint: local only with a GPU
    stylize_basic: str = "fallback"       # OpenCV ink/shadow — always available
    stylize_neural: str = "cloud"         # SD img2img / neural style
    text_commands: str = "fallback"       # rule-based now; LLM is roadmap

    def as_dict(self) -> Dict[str, str]:
        return {
            "Click-to-select": self.selection,
            "Background removal": self.background_removal,
            "Erase object": self.erase_object,
            "Generative fill": self.generative_fill,
            "Ink / shadow style": self.stylize_basic,
            "Neural / SD stylize": self.stylize_neural,
            "Text commands": self.text_commands,
        }


@dataclass
class AppConfig:
    profile: SystemProfile
    device: str = "cpu"

    # Selection model the auto-configurator would prefer (subject to weights
    # actually being present on disk — resolved later by the segmenter).
    sam_variant: str = "mobile_sam"

    # Largest dimension (px) fed to neural models. Bigger images are downscaled
    # for the model pass and edits re-applied at full resolution on export.
    max_process_dim: int = 1024

    capabilities: Capabilities = field(default_factory=Capabilities)
    default_backend_id: str = "local"
    recommended_cloud_backends: List[str] = field(default_factory=list)

    # Where downloaded model weights live (created lazily).
    weights_dir: Path = field(default_factory=lambda: Path.home() / ".pixie" / "weights")

    def describe(self) -> str:
        caps = ", ".join(f"{k}={v}" for k, v in self.capabilities.as_dict().items())
        return (
            f"device={self.device} · sam={self.sam_variant} · "
            f"max_process_dim={self.max_process_dim}\n  capabilities: {caps}"
        )


def auto_configure(profile: SystemProfile | None = None) -> AppConfig:
    """Derive an :class:`AppConfig` from the machine (detecting it if needed)."""
    if profile is None:
        profile = detect_system()

    gpu = profile.gpu
    device = profile.device
    has_cuda = gpu.backend == "cuda"
    has_mps = gpu.backend == "mps"
    vram = gpu.vram_gb

    # --- pick a SAM tier and a downscale budget -------------------------------
    if has_cuda and vram >= _SAM_LARGE_MIN_VRAM_GB:
        sam_variant, max_dim = "sam2_large", 4096
    elif has_cuda and vram >= _SD_MIN_VRAM_GB:
        sam_variant, max_dim = "sam2_small", 2048
    elif has_mps:
        # Apple Silicon shares RAM; be generous if there's plenty.
        sam_variant = "sam2_small" if profile.ram_gb >= 16 else "mobile_sam"
        max_dim = 1536
    else:  # CPU / integrated GPU
        sam_variant, max_dim = "mobile_sam", 1024
        if profile.ram_gb and profile.ram_gb < 8:
            max_dim = 768  # keep memory in check on light machines

    # --- decide which heavy ops can run locally -------------------------------
    can_generative_local = (has_cuda and vram >= _SD_MIN_VRAM_GB) or (
        has_mps and profile.ram_gb >= 16
    )

    caps = Capabilities(
        selection="fallback",  # upgraded to 'local' by the segmenter if SAM weights load
        background_removal="fallback",
        erase_object="fallback",  # upgraded to 'local' if LaMa loads
        generative_fill="local" if can_generative_local else "cloud",
        stylize_basic="fallback",
        stylize_neural="local" if can_generative_local else "cloud",
        text_commands="fallback",
    )

    # --- cloud recommendations when local GPU is weak/absent ------------------
    recommended: List[str] = []
    if not can_generative_local:
        recommended = ["hf_zerogpu", "modal", "replicate", "fal"]

    cfg = AppConfig(
        profile=profile,
        device=device,
        sam_variant=sam_variant,
        max_process_dim=max_dim,
        capabilities=caps,
        default_backend_id="local",
        recommended_cloud_backends=recommended,
    )
    cfg.weights_dir.mkdir(parents=True, exist_ok=True)
    log.info("Auto-config:\n  %s", cfg.describe())
    return cfg


if __name__ == "__main__":  # ``python -m app.config``
    logging.basicConfig(level=logging.INFO)
    c = auto_configure()
    print(c.profile.summary())
    print(c.describe())
