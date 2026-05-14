#!/usr/bin/env python
"""Capture one preprocessed depth frame from the real ZED and save it.

Run on the lab machine where the ZED is plugged in. The frame is the same
``70x70 float32 in [0, 1]`` view the depth-student policy receives, so you
can diff it against a sim-side dump (peg_in_hole_dynamic/capture_sim_depth.py)
to confirm the real scene matches what the policy saw during training.

Examples:
    python deployment/capture_real_first_depth.py --out-dir /tmp/zed_first
    python deployment/capture_real_first_depth.py --serial 15107 --out-dir .
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.zed_async_reader import (  # noqa: E402
    DEPTH_FAR_M,
    DEPTH_NEAR_M,
    ZED_CAMERA_FPS,
    ZED_DEPTH_MODE,
    ZED_EXPOSURE,
    ZED_GAIN,
    ZED_RESOLUTION,
    ZED_SERIAL_NUMBER,
    ZedAsyncReader,
    ZedReaderConfig,
)


def _save_frame(out_dir: Path, name: str, frame: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{name}.npz"
    png_path = out_dir / f"{name}.png"
    np.savez_compressed(npz_path, image=frame)
    from PIL import Image

    img = (np.clip(frame, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(img).save(png_path)
    return png_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="/tmp/depth_compare",
                   help="Where to save real_depth.png + real_depth.npz.")
    p.add_argument("--name", type=str, default="real_depth")
    p.add_argument("--warmup-frames", type=int, default=3,
                   help="Drop this many first frames before saving "
                        "(ZED auto-exposure/gain settles fast but not instant).")
    p.add_argument("--timeout-s", type=float, default=10.0)
    p.add_argument("--serial", type=str, default=ZED_SERIAL_NUMBER)
    p.add_argument("--resolution", type=str, default=ZED_RESOLUTION)
    p.add_argument("--depth-mode", type=str, default=ZED_DEPTH_MODE)
    p.add_argument("--camera-fps", type=int, default=ZED_CAMERA_FPS)
    p.add_argument("--exposure", type=int, default=ZED_EXPOSURE)
    p.add_argument("--gain", type=int, default=ZED_GAIN)
    p.add_argument("--depth-near-m", type=float, default=DEPTH_NEAR_M)
    p.add_argument("--depth-far-m", type=float, default=DEPTH_FAR_M)
    p.add_argument("--camera-upsidedown", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = ZedReaderConfig(
        serial_number=args.serial,
        resolution=args.resolution,
        depth_mode=args.depth_mode,
        camera_fps=args.camera_fps,
        grab_hz=float(args.camera_fps),
        exposure=args.exposure,
        gain=args.gain,
        depth_near_m=args.depth_near_m,
        depth_far_m=args.depth_far_m,
        camera_upsidedown=args.camera_upsidedown,
    )
    out_dir = Path(args.out_dir).resolve()
    deadline = time.perf_counter() + args.timeout_s

    with ZedAsyncReader(cfg) as reader:
        # Skip a few warmup frames so exposure/gain settles.
        last_id = -1
        kept = 0
        while time.perf_counter() < deadline:
            try:
                frame, frame_id, age_s = reader.get_latest(timeout_s=0.2)
            except RuntimeError as exc:
                print(f"[capture-real] waiting for frame: {exc}", flush=True)
                continue
            if frame_id == last_id:
                time.sleep(1.0 / max(args.camera_fps, 1))
                continue
            last_id = frame_id
            kept += 1
            if kept <= args.warmup_frames:
                continue

            png = _save_frame(out_dir, args.name, frame)
            print(
                f"[capture-real] saved {png} "
                f"(frame_id={frame_id} age_ms={age_s * 1000:.1f} "
                f"shape={frame.shape} min={float(frame.min()):.3f} "
                f"mean={float(frame.mean()):.3f} max={float(frame.max()):.3f})",
                flush=True,
            )
            return 0

    print(f"[capture-real] timed out after {args.timeout_s}s waiting for ZED frames",
          flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
