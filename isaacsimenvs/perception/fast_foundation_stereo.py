"""Wrapper around NVlabs/Fast-FoundationStereo for the sim eval loop.

Loads either the PyTorch checkpoint (`model_best_bp2_serialize.pth`) OR the
pair of TRT engine files (`feature_runner.engine` + `post_runner.engine`),
exposes a single ``__call__(left, right, fx_px) -> depth`` that returns
metric depth in meters, and handles the bookkeeping (InputPadder, AMP
autocast, batch dim, disparity-to-depth, mask for sky/no-match pixels).

Construction-time choice:
    - If ``engine_dir`` is provided AND both engine files exist, use the
      TRT path via core.foundation_stereo.TrtRunner (~4 ms / frame at
      384x224 / 4 iters on RTX 6000 Ada).
    - Otherwise fall back to PyTorch via the serialized model under
      ``model_dir`` (~26 ms / frame at same config). Useful for bring-up.

This module is the single integration point for Fast-FS — sim
read_student_camera_image() and the eventual deployment node both go
through it. See deployment/FAST_FS_SETUP.md for setup steps.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import yaml


def _ensure_on_path() -> None:
    """Add the vendored Fast-FS repo to sys.path so its core.* imports resolve."""
    fs_root = (Path(__file__).resolve().parents[2] / "third_party" / "Fast-FoundationStereo")
    if not fs_root.is_dir():
        raise FileNotFoundError(
            f"Fast-FoundationStereo not vendored at {fs_root}. See "
            f"deployment/FAST_FS_SETUP.md for setup instructions."
        )
    s = str(fs_root)
    if s not in sys.path:
        sys.path.insert(0, s)


class FastFoundationStereoModule:
    """Stateful inference wrapper. Construct once, call per frame.

    Args:
        model_dir: directory containing ``model_best_bp2_serialize.pth`` +
            ``cfg.yaml`` (the .pth/.yaml pair downloaded for one of the
            ``HH-MM-SS``-named model variants).
        engine_dir: optional directory containing ``feature_runner.engine``
            + ``post_runner.engine`` + ``onnx.yaml`` (produced by
            ``scripts/make_onnx.py`` + ``deployment/build_trt_engine.py``).
            If both engine files are present here, TRT inference is used.
        valid_iters: number of refinement iterations (must match the value
            baked into the TRT engine, if using TRT).
        max_disp: maximum disparity used for the cost volume.
        device: torch device for tensors.
    """

    def __init__(
        self,
        model_dir: str | Path,
        engine_dir: str | Path | None = None,
        valid_iters: int = 4,
        max_disp: int = 192,
        device: str = "cuda:0",
    ):
        _ensure_on_path()
        from omegaconf import OmegaConf
        from core.utils.utils import InputPadder  # noqa: F401  (kept for re-export)
        from Utils import AMP_DTYPE

        self._device = device
        self._amp_dtype = AMP_DTYPE
        self._valid_iters = int(valid_iters)
        self._max_disp = int(max_disp)

        engine_dir = Path(engine_dir) if engine_dir else None
        if engine_dir is not None and (engine_dir / "feature_runner.engine").exists() \
                                    and (engine_dir / "post_runner.engine").exists():
            self.backend = "trt"
            cfg_path = engine_dir / "onnx.yaml"
            if not cfg_path.exists():
                raise FileNotFoundError(
                    f"TRT engine pair found at {engine_dir} but onnx.yaml is missing. "
                    f"Re-run scripts/make_onnx.py to regenerate metadata."
                )
            with open(cfg_path) as f:
                cfg = OmegaConf.create(yaml.safe_load(f))
            cfg.valid_iters = self._valid_iters
            cfg.max_disp = self._max_disp
            from core.foundation_stereo import TrtRunner
            self._model = TrtRunner(
                cfg,
                str(engine_dir / "feature_runner.engine"),
                str(engine_dir / "post_runner.engine"),
            )
            self._stereo_hw = tuple(int(v) for v in cfg.image_size)  # (H, W)
            self._padded_hw = self._stereo_hw  # TRT engine is fixed-resolution
        else:
            self.backend = "pytorch"
            ckpt_path = Path(model_dir) / "model_best_bp2_serialize.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Fast-FS checkpoint not found: {ckpt_path}. "
                    f"See deployment/FAST_FS_SETUP.md."
                )
            torch.autograd.set_grad_enabled(False)
            self._model = torch.load(
                str(ckpt_path), map_location="cpu", weights_only=False
            )
            self._model.args.valid_iters = self._valid_iters
            self._model.args.max_disp = self._max_disp
            self._model = self._model.to(device).eval()
            self._stereo_hw = None  # determined per call
            self._padded_hw = None

        self._InputPadder = InputPadder

    @torch.inference_mode()
    def __call__(
        self,
        left_rgb: torch.Tensor,
        right_rgb: torch.Tensor,
        fx_px: float,
        baseline_m: float,
        eps_disp: float = 1e-3,
    ) -> torch.Tensor:
        """Return metric depth `(B, 1, H, W)` in meters.

        Inputs:
            left_rgb, right_rgb: (B, 3, H, W) float32 in [0, 255], on CUDA.
            fx_px: focal length in pixels at the input resolution.
            baseline_m: stereo baseline in meters.
            eps_disp: clamp floor for disparity before division (prevents
                inf depth at zero-disparity sky regions).
        """
        if left_rgb.dim() != 4 or left_rgb.shape != right_rgb.shape:
            raise ValueError(
                f"left/right shapes must match (B, 3, H, W); "
                f"got {tuple(left_rgb.shape)} vs {tuple(right_rgb.shape)}"
            )

        if self.backend == "trt":
            # The compiled engine is batch=1. For B>1 we loop -- inference is
            # ~4 ms per (left,right) at 384x224, so the wall-clock cost scales
            # linearly with B but is still trivially fast for offline eval at
            # B<=16. Re-export with dynamic batch if you need higher B.
            H, W = self._stereo_hw
            B = left_rgb.shape[0]
            if B == 1:
                disp = self._model.forward(left_rgb, right_rgb).view(1, 1, H, W)
            else:
                disp_chunks = [
                    self._model.forward(left_rgb[i:i+1], right_rgb[i:i+1]).view(1, 1, H, W)
                    for i in range(B)
                ]
                disp = torch.cat(disp_chunks, dim=0)
        else:
            padder = self._InputPadder(
                left_rgb.shape, divis_by=32, force_square=False
            )
            l_p, r_p = padder.pad(left_rgb, right_rgb)
            with torch.amp.autocast("cuda", enabled=True, dtype=self._amp_dtype):
                disp = self._model.forward(
                    l_p, r_p,
                    iters=self._valid_iters,
                    test_mode=True,
                    optimize_build_volume="pytorch1",
                )
            disp = padder.unpad(disp.float())   # (B, 1, H, W)

        depth = float(fx_px) * float(baseline_m) / disp.clamp(min=eps_disp)
        return depth
