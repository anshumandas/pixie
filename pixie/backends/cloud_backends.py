"""Cloud compute backends.

These report availability (based on credentials/config) and a representative
per-operation cost so the picker can show prices at runtime. Actual network
calls are intentionally left as clearly-marked integration points: wiring a
provider is a few lines once you add a key, and keeping them inert means the
app runs fully offline until you opt in.

Cost figures are approximate (mid-2026) and provider pricing changes — treat
them as ballpark, and override via the constructor if you want exact numbers.
"""
from __future__ import annotations

import os

from .base import Availability, ComputeBackend, CostEstimate, Operation

_HEAVY_OPS = (Operation.GENERATIVE_FILL, Operation.STYLIZE_NEURAL)


class _CloudBackend(ComputeBackend):
    kind = "hosted-api"
    sends_data_offsite = True
    env_var = ""           # credential env var to look for
    setup_hint = ""        # shown when unavailable

    def availability(self) -> Availability:
        if self.env_var and not os.environ.get(self.env_var):
            return Availability(ready=False, reason=self.setup_hint)
        return Availability(ready=True)

    def supports(self, op: Operation) -> bool:
        return op in _HEAVY_OPS

    def run(self, op: Operation, **payload):  # pragma: no cover - integration point
        raise NotImplementedError(
            f"{self.display_name}: add your integration in {type(self).__name__}.run(). "
            f"{self.setup_hint}"
        )


class ReplicateBackend(_CloudBackend):
    id = "replicate"
    display_name = "Replicate (hosted API)"
    env_var = "REPLICATE_API_TOKEN"
    setup_hint = "set REPLICATE_API_TOKEN and `pip install replicate`"

    def estimate_cost(self, op: Operation) -> CostEstimate:
        # SD inpaint/img2img on an A100 bills per second; ~$0.005–0.02 typical.
        return CostEstimate(
            is_free=False, amount_usd=0.011, unit="op",
            note="billed per second, varies by model",
        )


class FalBackend(_CloudBackend):
    id = "fal"
    display_name = "fal.ai (hosted API)"
    env_var = "FAL_KEY"
    setup_hint = "set FAL_KEY and `pip install fal-client`"

    def estimate_cost(self, op: Operation) -> CostEstimate:
        return CostEstimate(
            is_free=False, amount_usd=0.02, unit="op",
            note="per-second billing; fast; promo credits sometimes",
        )


class ModalBackend(_CloudBackend):
    id = "modal"
    display_name = "Modal (your function on their GPU)"
    kind = "self-hosted"
    env_var = "MODAL_TOKEN_ID"
    setup_hint = "configure `modal token set` then deploy the inpaint function"

    def estimate_cost(self, op: Operation) -> CostEstimate:
        return CostEstimate(
            is_free=True, note="within $30/mo free credits, then ~per-second",
        )


class HFZeroGPUBackend(_CloudBackend):
    """Fully wired backend: calls a Gradio ZeroGPU Space via gradio_client.

    Deploy the Space in this repo's ``space/`` folder (free ZeroGPU hardware),
    then set ``HF_ZEROGPU_SPACE="you/your-space"`` locally. The Space exposes two
    endpoints — ``/inpaint`` (image, mask, prompt) and ``/stylize`` (image,
    prompt) — which this backend drives for generative fill and neural stylise.
    """

    id = "hf_zerogpu"
    display_name = "Hugging Face ZeroGPU Space"
    kind = "self-hosted"
    env_var = "HF_ZEROGPU_SPACE"  # e.g. "your-name/your-space"
    setup_hint = "set HF_ZEROGPU_SPACE to your Gradio Space id and `pip install gradio_client`"

    def estimate_cost(self, op: Operation) -> CostEstimate:
        return CostEstimate(is_free=True, note="free shared GPU, may queue / cold-start")

    def _client(self):
        from gradio_client import Client  # imported lazily so the app runs without it

        space = os.environ.get(self.env_var)
        token = os.environ.get("HF_TOKEN")  # optional: private spaces / higher quota
        return Client(space, hf_token=token)

    def availability(self) -> Availability:
        if not os.environ.get(self.env_var):
            return Availability(ready=False, reason=self.setup_hint)
        try:
            import gradio_client  # noqa: F401
        except Exception:
            return Availability(ready=False, reason="`pip install gradio_client`")
        return Availability(ready=True)

    def run(self, op: Operation, **payload):
        """Send the image (+ mask) to the Space and return the result as RGB uint8."""
        import os as _os
        import tempfile

        import numpy as np
        from gradio_client import handle_file
        from PIL import Image

        image = np.asarray(payload["image"])[..., :3].astype("uint8")
        client = self._client()

        with tempfile.TemporaryDirectory() as d:
            img_path = _os.path.join(d, "image.png")
            Image.fromarray(image).save(img_path)

            if op is Operation.GENERATIVE_FILL:
                mask = np.asarray(payload["mask"])
                if mask.ndim == 3:
                    mask = mask[..., 0]
                mask255 = (mask > 0).astype("uint8") * 255
                mask_path = _os.path.join(d, "mask.png")
                Image.fromarray(mask255).save(mask_path)
                result = client.predict(
                    handle_file(img_path),
                    handle_file(mask_path),
                    payload.get("prompt", "") or "fill the area naturally",
                    api_name="/inpaint",
                )
            else:  # STYLIZE_NEURAL
                result = client.predict(
                    handle_file(img_path),
                    payload.get("prompt", "") or "artistic ink illustration",
                    api_name="/stylize",
                )

            out_path = _result_path(result)
            return np.array(Image.open(out_path).convert("RGB"))


def _result_path(result) -> str:
    """gradio_client may return a filepath, a dict with 'path'/'url', or a tuple."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return result.get("path") or result.get("url")  # type: ignore[return-value]
    if isinstance(result, (list, tuple)) and result:
        return _result_path(result[0])
    raise RuntimeError(f"Unexpected result from Space: {type(result)}")


ALL_CLOUD_BACKENDS = [
    ReplicateBackend,
    FalBackend,
    ModalBackend,
    HFZeroGPUBackend,
]
