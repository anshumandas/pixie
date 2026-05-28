"""Runtime hardware detection.

This module is intentionally dependency-light: it works even if torch / psutil
are not installed, degrading to whatever it can detect. It is re-run on every
launch so that adding a GPU, drivers, or libraries later is picked up
automatically with no code changes.
"""
from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List

log = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Detected accelerator. ``backend`` is one of 'cuda', 'mps', 'none'."""
    backend: str = "none"
    name: str = ""
    vram_gb: float = 0.0

    @property
    def present(self) -> bool:
        return self.backend in ("cuda", "mps")


@dataclass
class SystemProfile:
    os_name: str = ""
    os_version: str = ""
    arch: str = ""
    cpu_cores: int = 1
    ram_gb: float = 0.0
    gpu: GPUInfo = field(default_factory=GPUInfo)
    torch_available: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def device(self) -> str:
        """The torch-style device string we should target."""
        return self.gpu.backend if self.gpu.present else "cpu"

    def summary(self) -> str:
        if self.gpu.present:
            vram = f", {self.gpu.vram_gb:.1f} GB VRAM" if self.gpu.vram_gb else ""
            gpu = f"{self.gpu.name} [{self.gpu.backend}{vram}]"
        else:
            gpu = "no dedicated GPU (integrated / CPU only)"
        return (
            f"{self.os_name} {self.os_version} ({self.arch}) · "
            f"{self.cpu_cores} CPU cores · {self.ram_gb:.1f} GB RAM · {gpu}"
        )


def _detect_ram_gb() -> float:
    # Preferred: psutil (cross-platform, accurate).
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        pass
    # Unix fallback.
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024 ** 3), 1)
    except (ValueError, AttributeError, OSError):
        pass
    # Windows fallback via ctypes.
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
        return round(stat.ullTotalPhys / (1024 ** 3), 1)
    except Exception:
        return 0.0


def _detect_gpu_via_torch() -> GPUInfo | None:
    try:
        import torch  # type: ignore
    except Exception:
        return None

    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return GPUInfo(
                backend="cuda",
                name=props.name,
                vram_gb=round(props.total_memory / (1024 ** 3), 1),
            )
    except Exception as exc:  # pragma: no cover - hardware specific
        log.debug("CUDA probe failed: %s", exc)

    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            # MPS uses unified memory; report 0 (shared with system RAM).
            return GPUInfo(backend="mps", name="Apple Silicon GPU (Metal/MPS)", vram_gb=0.0)
    except Exception as exc:  # pragma: no cover - hardware specific
        log.debug("MPS probe failed: %s", exc)

    return GPUInfo(backend="none")


def _detect_gpu_via_nvidia_smi() -> GPUInfo | None:
    """Fallback so we can still spot an NVIDIA GPU even if torch isn't installed."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        line = out.stdout.strip().splitlines()[0]
        name, mem = [p.strip() for p in line.split(",")[:2]]
        vram = float(re.findall(r"[\d.]+", mem)[0]) / 1024.0  # MiB -> GiB
        return GPUInfo(backend="cuda", name=name, vram_gb=round(vram, 1))
    except Exception as exc:  # pragma: no cover - hardware specific
        log.debug("nvidia-smi probe failed: %s", exc)
        return None


def detect_system() -> SystemProfile:
    """Inspect the current machine and return a :class:`SystemProfile`."""
    notes: List[str] = []

    torch_available = False
    try:
        import torch  # type: ignore  # noqa: F401

        torch_available = True
    except Exception:
        notes.append("PyTorch not installed — neural models disabled until it is.")

    gpu = _detect_gpu_via_torch()
    if gpu is None or not gpu.present:
        smi = _detect_gpu_via_nvidia_smi()
        if smi is not None and smi.present:
            gpu = smi
            if not torch_available:
                notes.append(
                    "NVIDIA GPU detected via driver, but PyTorch (CUDA build) is not "
                    "installed — install it to use the GPU."
                )
    if gpu is None:
        gpu = GPUInfo(backend="none")

    profile = SystemProfile(
        os_name=platform.system() or "Unknown",
        os_version=platform.release(),
        arch=platform.machine(),
        cpu_cores=os.cpu_count() or 1,
        ram_gb=_detect_ram_gb(),
        gpu=gpu,
        torch_available=torch_available,
        notes=notes,
    )
    log.info("System profile: %s", profile.summary())
    return profile


if __name__ == "__main__":  # quick manual check: ``python -m app.system_check``
    logging.basicConfig(level=logging.INFO)
    p = detect_system()
    print(p.summary())
    for n in p.notes:
        print("  note:", n)
