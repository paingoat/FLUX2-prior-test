"""FLUX.2 [klein] 4B editing via diffusers.

Official Black Forest Labs sampling settings (flux2 ``FLUX2_MODEL_INFO``):

- distill (``FLUX.2-klein-4B``): 4 steps, guidance 1.0, both fixed
  because the checkpoint is step- and guidance-distilled.
- base (``FLUX.2-klein-base-4B``): 50 steps, guidance 4.0, classifier-free
  guidance with an empty negative prompt (the pipeline does this itself).

The two repos ship the same Qwen3-4B text encoder and the same VAE, so the
second pipeline reuses those modules and only loads its own transformer.
Output dimensions follow the first reference, capped at 1024*1024 pixels
and rounded down to multiples of 16, for comparable single/multi-image runs.
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from typing import Any

from app.common.env import load_env

load_env()

import diffusers
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

MAX_AREA = 1024 * 1024
MULTIPLE_OF = 16


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    dir_name: str
    steps: int
    guidance: float
    fixed: bool


MODEL_SPECS: dict[str, ModelSpec] = {
    "distill": ModelSpec(
        key="distill",
        repo="black-forest-labs/FLUX.2-klein-4B",
        dir_name="flux2-klein-distill",
        steps=4,
        guidance=1.0,
        fixed=True,
    ),
    "base": ModelSpec(
        key="base",
        repo="black-forest-labs/FLUX.2-klein-base-4B",
        dir_name="flux2-klein-base",
        steps=50,
        guidance=4.0,
        fixed=False,
    ),
}


def model_input_size(image: Image.Image, max_area: int = MAX_AREA) -> tuple[int, int]:
    """Size the pipeline will actually denoise, before the final center crop.

    Mirrors ``Flux2KleinPipeline``: downscale when the area exceeds ``max_area``,
    then round each side down to a multiple of 16.
    """
    width, height = image.size
    if width * height > max_area:
        scale = math.sqrt(max_area / (width * height))
        width = int(width * scale)
        height = int(height * scale)
    width = max(MULTIPLE_OF, (width // MULTIPLE_OF) * MULTIPLE_OF)
    height = max(MULTIPLE_OF, (height // MULTIPLE_OF) * MULTIPLE_OF)
    return width, height


class ModelManager:
    """Keeps one or both Klein pipelines resident, sharing text encoder and VAE."""

    def __init__(self, offload: bool = False) -> None:
        self.offload = offload
        self.pipes: dict[str, Flux2KleinPipeline] = {}
        self._text_encoder = None
        self._tokenizer = None
        self._vae = None

    def load(self, variant: str, *, quiet: bool = False) -> Flux2KleinPipeline:
        if variant in self.pipes:
            return self.pipes[variant]
        spec = _spec(variant)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required. FLUX.2 [klein] is not run on CPU.")

        # CPU offload hooks do not compose across two pipelines that share
        # modules, so offload mode keeps a single pipeline and reloads it
        # from scratch on every switch.
        if self.offload:
            for other in list(self.pipes):
                self.unload(other, release_shared=True)
            self._text_encoder = None
            self._tokenizer = None
            self._vae = None

        kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
        if self._text_encoder is not None:
            kwargs["text_encoder"] = self._text_encoder
            kwargs["tokenizer"] = self._tokenizer
            kwargs["vae"] = self._vae

        pipe = Flux2KleinPipeline.from_pretrained(spec.repo, **kwargs)
        if quiet:
            pipe.set_progress_bar_config(disable=True)
        if self.offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")

        self.pipes[variant] = pipe
        self._text_encoder = pipe.text_encoder
        self._tokenizer = pipe.tokenizer
        self._vae = pipe.vae
        return pipe

    def preload(self, which: str, *, quiet: bool = False) -> None:
        if which == "all":
            keys = list(MODEL_SPECS)
        elif which in MODEL_SPECS:
            keys = [which]
        else:
            raise ValueError(f"Unknown preload {which!r}. Choose all, distill, or base.")
        for key in keys:
            self.load(key, quiet=quiet)

    def unload(self, variant: str, *, release_shared: bool = False) -> None:
        """Drop one pipeline. The shared text encoder and VAE stay unless asked.

        Batch inference unloads the transformer between variants and reuses the
        encoder. Offload mode passes ``release_shared=True`` because the next
        pipeline is loaded on its own.
        """
        pipe = self.pipes.pop(variant, None)
        if pipe is None:
            return
        if release_shared:
            self._text_encoder = None
            self._tokenizer = None
            self._vae = None
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def edit(
        self,
        variant: str,
        image: Image.Image | list[Image.Image],
        prompt: str,
        seed: int,
        steps: int | None = None,
        guidance: float | None = None,
        *,
        quiet: bool = False,
    ) -> tuple[Image.Image, dict[str, Any]]:
        references = image if isinstance(image, list) else [image]
        if not references or not all(isinstance(ref, Image.Image) for ref in references):
            raise ValueError("Provide an image or a non-empty list of PIL images.")
        references = [ref.convert("RGB") for ref in references]
        seen = model_input_size(references[0])
        spec = _spec(variant)
        pipe = self.load(variant, quiet=quiet)
        if spec.fixed:
            steps = spec.steps
            guidance = spec.guidance
        else:
            steps = spec.steps if steps is None else int(steps)
            guidance = spec.guidance if guidance is None else float(guidance)

        generator = torch.Generator(device="cuda").manual_seed(int(seed))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = pipe(
            image=references if isinstance(image, list) else references[0],
            prompt=prompt,
            width=seen[0],
            height=seen[1],
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        meta = {
            "variant": spec.key,
            "repo": spec.repo,
            "dir_name": spec.dir_name,
            "steps": steps,
            "guidance": guidance,
            "guidance_fixed": spec.fixed,
            "seed": int(seed),
            "dtype": "bfloat16",
            "max_area": MAX_AREA,
            "input_size": list(references[0].size),
            "reference_count": len(references),
            "reference_sizes": [list(ref.size) for ref in references],
            "reference_model_sizes": [list(model_input_size(ref)) for ref in references],
            "model_input_size": [seen[0], seen[1]],
            "output_size": [result.width, result.height],
            "elapsed_sec": round(elapsed, 3),
            "peak_vram_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "offload": self.offload,
        }
        return result, meta


def _spec(variant: str) -> ModelSpec:
    try:
        return MODEL_SPECS[variant]
    except KeyError:
        known = ", ".join(MODEL_SPECS)
        raise KeyError(f"Unknown model {variant!r}. Choose one of: {known}.") from None
