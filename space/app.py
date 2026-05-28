"""Pixie ZeroGPU backend — a Hugging Face Space that does the GPU-heavy work.

Deploy this as a **Gradio** Space on **ZeroGPU** hardware (free). It exposes two
API endpoints that Pixie's desktop app calls via ``gradio_client``:

* ``/inpaint``  (image, mask, prompt)  → Stable Diffusion inpainting / generative fill
* ``/stylize``  (image, prompt)        → Stable Diffusion img2img stylisation

The desktop side is already wired in ``pixie/backends/cloud_backends.py``; just
set ``HF_ZEROGPU_SPACE="your-name/your-space"`` locally after deploying.
"""
from __future__ import annotations

import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image

INPAINT_MODEL = "stabilityai/stable-diffusion-2-inpainting"
IMG2IMG_MODEL = "stabilityai/stable-diffusion-2-1-base"
DTYPE = torch.float16
MAX_SIDE = 768  # cap the working size for speed; result is resized back

_inpaint = None
_img2img = None


def _get_inpaint():
    global _inpaint
    if _inpaint is None:
        from diffusers import StableDiffusionInpaintPipeline

        _inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            INPAINT_MODEL, torch_dtype=DTYPE, safety_checker=None
        )
    return _inpaint


def _get_img2img():
    global _img2img
    if _img2img is None:
        from diffusers import StableDiffusionImg2ImgPipeline

        _img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            IMG2IMG_MODEL, torch_dtype=DTYPE, safety_checker=None
        )
    return _img2img


def _fit(img: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    """Resize so the longest side <= MAX_SIDE and both sides are multiples of 8."""
    w, h = img.size
    scale = min(1.0, MAX_SIDE / float(max(w, h)))
    nw, nh = max(8, int(w * scale) // 8 * 8), max(8, int(h * scale) // 8 * 8)
    return img.resize((nw, nh)), (w, h)


@spaces.GPU(duration=120)
def inpaint(image: str, mask: str, prompt: str):
    pipe = _get_inpaint().to("cuda")
    img = Image.open(image).convert("RGB")
    msk = Image.open(mask).convert("L")
    work, original = _fit(img)
    msk = msk.resize(work.size)
    out = pipe(
        prompt=prompt or "fill the area naturally, photorealistic",
        image=work,
        mask_image=msk,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]
    return out.resize(original)


@spaces.GPU(duration=120)
def stylize(image: str, prompt: str):
    pipe = _get_img2img().to("cuda")
    img = Image.open(image).convert("RGB")
    work, original = _fit(img)
    out = pipe(
        prompt=prompt or "artistic ink illustration, high detail",
        image=work,
        strength=0.6,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]
    return out.resize(original)


with gr.Blocks(title="Pixie ZeroGPU backend") as demo:
    gr.Markdown(
        "# Pixie ZeroGPU backend\n"
        "GPU worker for the Pixie desktop app. Use the tabs to test, or call the "
        "`/inpaint` and `/stylize` API endpoints from `gradio_client`."
    )
    with gr.Tab("Inpaint / generative fill"):
        i_img = gr.Image(type="filepath", label="Image")
        i_mask = gr.Image(type="filepath", label="Mask (white = area to fill)")
        i_prompt = gr.Textbox(label="Prompt", placeholder="e.g. a potted plant")
        i_out = gr.Image(label="Result")
        gr.Button("Run inpaint").click(
            inpaint, [i_img, i_mask, i_prompt], i_out, api_name="inpaint"
        )
    with gr.Tab("Stylize (img2img)"):
        s_img = gr.Image(type="filepath", label="Image")
        s_prompt = gr.Textbox(label="Prompt", placeholder="e.g. ink illustration")
        s_out = gr.Image(label="Result")
        gr.Button("Run stylize").click(
            stylize, [s_img, s_prompt], s_out, api_name="stylize"
        )

if __name__ == "__main__":
    demo.queue().launch()
