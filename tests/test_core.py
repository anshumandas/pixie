"""Headless unit tests for the core pipeline (no Qt, no torch required).

Run with:  pytest -q
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pixie.backends import Operation, build_registry  # noqa: E402
from pixie.config import auto_configure  # noqa: E402
from pixie.engine import Editor  # noqa: E402
from pixie.nlp import RuleBasedParser  # noqa: E402
from pixie.ops import adjust_brightness_contrast, composite_into, remove_background  # noqa: E402
from pixie.system_check import detect_system  # noqa: E402


def _demo_image(h=120, w=160) -> np.ndarray:
    """A structured RGB image so GrabCut/inpaint have something to chew on."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (30, 30, 30)
    # a bright rectangle "object" in the middle
    img[40:80, 60:100] = (220, 80, 60)
    return img


# --------------------------------------------------------------- system check
def test_detect_and_autoconfig():
    profile = detect_system()
    assert profile.cpu_cores >= 1
    assert profile.device in ("cpu", "cuda", "mps")

    cfg = auto_configure(profile)
    assert cfg.max_process_dim >= 256
    # On a CPU-only box, generative fill must route to cloud.
    if profile.device == "cpu":
        assert cfg.sam_variant == "mobile_sam"
        assert cfg.capabilities.generative_fill == "cloud"


# --------------------------------------------------------------- adjustments
def test_brightness_increases_mean():
    img = _demo_image()
    brighter = adjust_brightness_contrast(img, brightness=0.5, contrast=0.0)
    assert brighter.mean() > img.mean()
    darker = adjust_brightness_contrast(img, brightness=-0.5, contrast=0.0)
    assert darker.mean() < img.mean()


def test_adjust_only_inside_mask():
    img = _demo_image()
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[40:80, 60:100] = 1
    out = adjust_brightness_contrast(img, brightness=0.8, contrast=0.0, mask=mask, feather=0)
    # outside the mask is unchanged
    assert np.array_equal(out[0:10, 0:10], img[0:10, 0:10])
    # inside the mask is brighter
    assert out[40:80, 60:100].mean() > img[40:80, 60:100].mean()


# --------------------------------------------------------------- compositing
def test_remove_background_alpha():
    img = _demo_image()
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[40:80, 60:100] = 1
    rgba = remove_background(img, mask, feather=0)
    assert rgba.shape[2] == 4
    assert rgba[0, 0, 3] == 0           # outside subject -> transparent
    assert rgba[60, 80, 3] == 255       # inside subject -> opaque


def test_composite_into_blends():
    base = np.zeros((20, 20, 3), np.uint8)
    edited = np.full((20, 20, 3), 200, np.uint8)
    mask = np.zeros((20, 20), np.uint8)
    mask[5:15, 5:15] = 1
    out = composite_into(base, edited, mask, feather=0)
    assert out[10, 10].mean() > 100     # inside changed
    assert out[0, 0].mean() == 0        # outside unchanged


# --------------------------------------------------------------- backends/cost
def test_backend_registry_costs():
    cfg = auto_configure()
    reg = build_registry(cfg)
    local = reg.get("local")
    assert local is not None
    assert local.supports(Operation.SEGMENT)
    assert local.supports(Operation.ERASE)

    rows = reg.cost_table(Operation.GENERATIVE_FILL)
    ids = {r["id"] for r in rows}
    assert {"replicate", "fal", "modal", "hf_zerogpu"}.issubset(ids)
    # every row exposes a human-readable cost string
    for r in rows:
        assert r["cost"]
    # the paid options show a dollar figure; free ones say "Free"
    replicate = next(r for r in rows if r["id"] == "replicate")
    assert "$" in replicate["cost"]
    zerogpu = next(r for r in rows if r["id"] == "hf_zerogpu")
    assert "Free" in zerogpu["cost"]


# --------------------------------------------------------------- command parser
def test_rule_based_parser():
    p = RuleBasedParser()
    cmds = p.parse("remove the background and make it darker")
    actions = [c.action for c in cmds]
    assert "remove_background" in actions
    assert "adjust" in actions
    adj = next(c for c in cmds if c.action == "adjust")
    assert adj.args["brightness"] < 0

    ink = p.parse("convert this to ink")
    assert any(c.action == "stylize" and c.args.get("style") == "Ink" for c in ink)


# --------------------------------------------------------------- engine e2e
def test_engine_end_to_end_fallbacks():
    editor = Editor(config=auto_configure())
    editor.load_image_array(_demo_image())
    assert editor.has_image

    # click-to-select (GrabCut fallback) yields a non-empty selection
    editor.select_at(80, 60)
    assert editor.has_selection

    # brightness/contrast commit + undo
    before = editor.current.copy()
    editor.adjust(0.4, 0.2, commit=True)
    assert not np.array_equal(editor.current, before)
    editor.undo()
    assert np.array_equal(editor.current, before)

    # stylise whole image (ink) returns same spatial size
    styled = editor.stylize("Ink", whole_image=True)
    assert styled.shape[:2] == _demo_image().shape[:2]

    # re-select then erase via OpenCV inpaint fallback
    editor.select_at(80, 60)
    erased = editor.erase_selection()
    assert erased.shape[:2] == _demo_image().shape[:2]

    # background removal produces RGBA
    editor.select_at(80, 60)
    rgba = editor.remove_background()
    assert rgba.shape[2] == 4

    # export round-trips to disk
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.png")
        editor.export(out)
        assert os.path.exists(out) and os.path.getsize(out) > 0
