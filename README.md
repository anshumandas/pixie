# Pixie

A local-first desktop image editor. Click an object to select it, then **remove
the background**, **erase the object**, **restyle it** (ink / shadow / pencil /
oil), or **adjust brightness & contrast** of just that object.

It **auto-detects your hardware on every launch** and configures itself:

- Runs **immediately on any machine** — even CPU-only / integrated graphics —
  using built-in OpenCV methods, with **no model downloads required**.
- **Upgrades itself automatically** when better tools become available: drop in
  SAM weights for sharper selection, install LaMa for cleaner object removal,
  add a GPU for Stable Diffusion, or plug in a cloud backend for generative fill.

This is the **Phase 1–3 starter** (load → click-select → background removal,
object removal, brightness/contrast), with the seams for cloud backends and a
natural-language command bar already in place.

---

## Quick start

```bash
cd pixie
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python run.py
```

On launch it prints what it detected, e.g.:

```
Detected: Windows 11 (AMD64) · 8 CPU cores · 16.0 GB RAM · no dedicated GPU (integrated / CPU only)
device=cpu · sam=mobile_sam · max_process_dim=1024
  capabilities: Click-to-select=fallback, Erase object=fallback, Generative fill=cloud, ...
```

Then: **Open…** an image, **click** an object (a blue overlay shows the
selection), and use the buttons/sliders on the right.

---

## What runs where

| Feature | On a fresh install (no downloads) | After optional upgrade |
|---|---|---|
| Click-to-select | OpenCV **GrabCut** | **SAM / MobileSAM** (sharper) |
| Background removal | selection mask → transparent PNG | same, better masks with SAM |
| Erase object | **cv2.inpaint** | **LaMa** (cleaner fill) |
| Ink / shadow / pencil / oil | **OpenCV filters** | — (already great) |
| Brightness / contrast | NumPy on the masked region | — |
| Generative fill | needs GPU or cloud | local SD (GPU) or cloud backend |
| Text command bar | rule-based parser | local LLM via Ollama |

Capabilities are shown live in the **Detected setup** panel.

---

## Models (optional — for sharper selection / cleaner erase)

The app looks for SAM weights in `~/.pixie/weights/`. Install the
libraries (see `requirements.txt`) and drop in one checkpoint:

- **MobileSAM** (recommended for CPU): `mobile_sam.pt`
- SAM ViT-B: `sam_vit_b_01ec64.pth`
- SAM ViT-H: `sam_vit_h_4b8939.pth`

For object removal, `pip install simple-lama-inpainting` (LaMa downloads its own
weights on first use). No config needed — the app picks these up on next launch.

---

## Cloud GPU backends (optional)

Heavy operations (generative fill, neural/SD stylise) can run on a cloud GPU.
Open **"Choose backend (shows cost)…"** to see every option with its live cost
and a privacy note. Enable one by installing its package and setting its
credential:

| Backend | Cost (approx) | Status | Setup |
|---|---|---|---|
| This computer (local) | Free | ready | always available; generative fill needs a local GPU |
| **Hugging Face ZeroGPU** | Free (shared, may queue) | **fully wired** | deploy the Space in `space/`, then `pip install gradio_client` + `HF_ZEROGPU_SPACE="you/space"` |
| Modal | Free within $30/mo credits | stub | `pip install modal`; `modal token set` |
| Replicate | ~$0.01 / op | stub | `pip install replicate`; `REPLICATE_API_TOKEN=…` |
| fal.ai | ~$0.02 / op | stub | `pip install fal-client`; `FAL_KEY=…` |

Cloud backends **upload your image off-device**; the app is local-first and only
uses them when you explicitly pick one.

### Hugging Face ZeroGPU — the wired backend

This one works end-to-end. The folder [`space/`](space/) contains a ready Gradio
app you deploy (once) to a **free ZeroGPU Space**; it runs Stable Diffusion
inpainting and img2img. The desktop client (`pixie/backends/cloud_backends.py`,
`HFZeroGPUBackend`) saves your image + mask, calls the Space's `/inpaint`
endpoint via `gradio_client`, and shows the result. See
[`space/README.md`](space/README.md) for the 3-step deploy. The other providers
are cost-showing stubs you can fill in the same shape.

---

## Project layout

```
pixie/
  run.py                     # entry point
  pixie.spec                 # PyInstaller build spec (-> dist/Pixie/Pixie.exe)
  build_exe.bat              # one-click Windows build
  pixie/                     # the application package
    system_check.py          # runtime hardware detection (torch-optional)
    config.py                # auto-configuration from the detected hardware
    engine.py                # orchestrates models + ops + history + backends
    backends/                # local + cloud compute backends, cost registry
    models/                  # SAM/GrabCut, LaMa/cv2.inpaint, OpenCV stylers
    ops/                     # brightness/contrast, compositing, undo/redo
    nlp/                     # rule-based command parser + LLM seam
    ui/                      # PySide6 window, canvas, backend picker, threads
  space/                     # deployable Gradio ZeroGPU backend (server side)
  tests/                     # unit tests for the headless core
```

The `Editor` class in `pixie/engine.py` has **no Qt imports**, so the whole
pipeline is testable headlessly:

```bash
pip install pytest
pytest -q
```

---

## Packaging to a Windows .exe

Build on a **Windows** machine (PyInstaller produces an exe for the OS it runs
on):

```bat
build_exe.bat
```

That sets up a venv, installs deps + PyInstaller, runs `pyinstaller pixie.spec`,
and leaves the app at `dist\Pixie\Pixie.exe` (zip the `dist\Pixie` folder to
share it). The packaged build includes the always-on feature set; to bundle the
optional neural stack (SAM / LaMa / Stable Diffusion), install those packages and
remove them from `excludes` in `pixie.spec`.

## Roadmap (beyond this starter)

- **Phase 4:** more styles; neural style-transfer presets.
- **Phase 5:** multi-point / box prompts, full-resolution re-projection of edits
  (currently edits are applied at the working/downscaled resolution), JPEG/PNG
  export options.
- **Phase 6:** ✅ ZeroGPU cloud backend wired; remaining — local SD on GPU,
  Ollama command bar, and filling in the other cloud stubs.
- **Phase 7:** ✅ PyInstaller packaging (`pixie.spec` + `build_exe.bat`).

## Notes & limitations

- Large images are downscaled to `max_process_dim` for the model pass (auto-set
  from your hardware). Full-resolution re-projection is a Phase 5 item.
- GrabCut selection is approximate; install SAM for precise click-to-select.
- Cloud backend `run()` methods are stubs by design — see the table above.
