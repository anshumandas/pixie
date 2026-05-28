---
title: Pixie ZeroGPU Backend
emoji: 🪄
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Pixie ZeroGPU backend

The GPU worker for the [Pixie](../README.md) desktop app. It runs Stable
Diffusion **inpainting** (generative fill) and **img2img** (neural stylise) on
Hugging Face's free **ZeroGPU** hardware, and exposes them as API endpoints the
desktop app calls.

## Deploy (one time)

1. Create a new Space at https://huggingface.co/new-space
   - **SDK:** Gradio
   - **Hardware:** **ZeroGPU** (free; you may need to enable it in
     *Settings → Hardware* after creating the Space)
2. Upload `app.py` and `requirements.txt` from this folder (drag-and-drop in the
   Space's *Files* tab, or `git push` to the Space repo).
3. Wait for it to build and show *Running*. Test it in the two tabs.

## Point Pixie at it

On the machine running the desktop app:

```bash
# Windows (PowerShell)
setx HF_ZEROGPU_SPACE "your-username/pixie-zerogpu"
# optional, for private Spaces or higher quota:
setx HF_TOKEN "hf_xxx"

pip install gradio_client
```

Restart Pixie, open **"Choose backend (shows cost)…"**, pick **Hugging Face
ZeroGPU Space**, then use **Generative fill**. The app saves your image + mask to
a temp file, calls the Space's `/inpaint` endpoint, and shows the result.

## Notes

- ZeroGPU is shared, so the first call may cold-start or queue briefly.
- Images are capped to 768 px on the longest side for speed, then resized back.
- The default models are `stabilityai/stable-diffusion-2-inpainting` and
  `stabilityai/stable-diffusion-2-1-base`; swap them in `app.py` (e.g. for an
  SDXL inpainting model) if you want higher quality and don't mind it being
  slower.
