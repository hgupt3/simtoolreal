#!/usr/bin/env python
"""Capture one real-world depth frame from the ZED at sim camera resolution.

Saves four artifacts under ``--out-dir``:
  - ``<name>.npz`` / ``.png``       : 70x70 policy-preprocessed depth (window-
                                       normalized in [0, 1]) — the same view
                                       the depth-student receives.
  - ``<name>_raw.npz`` / ``_raw.png`` : 160x90 uncropped, unnormalized depth in
                                       meters (NaN/inf preserved). The PNG is
                                       a visualisation; the npz keeps the raw
                                       float32 values.

Opens the ZED inline (no multiprocessing) since this is a one-shot tool.
Uses the same SDK retrieve-on-resize overload as ``zed_async_reader.py`` so
the FOV stays calibrated.
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
    CROP_X0,
    CROP_X1,
    CROP_Y0,
    CROP_Y1,
    DEPTH_FAR_M,
    DEPTH_NEAR_M,
    SIM_CAMERA_HEIGHT,
    SIM_CAMERA_WIDTH,
    ZED_CAMERA_FPS,
    ZED_DEPTH_MODE,
    ZED_EXPOSURE,
    ZED_GAIN,
    ZED_RESOLUTION,
    ZED_SERIAL_NUMBER,
)


def _save_normalized_png(path: Path, frame: np.ndarray) -> None:
    from PIL import Image

    img = (np.clip(frame, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(img).save(path)


def _save_raw_visualization_png(path: Path, depth_m: np.ndarray,
                                vis_near: float, vis_far: float) -> None:
    """Save a uint8 PNG of the raw depth, stretched into [vis_near, vis_far]m."""
    from PIL import Image

    finite = np.isfinite(depth_m)
    safe = np.where(finite, depth_m, vis_far)
    norm = (safe - vis_near) / max(vis_far - vis_near, 1e-6)
    img = (np.clip(norm, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(img).save(path)


def _zed_enum_value(enum_cls, name: str):
    try:
        return getattr(enum_cls, str(name).upper())
    except AttributeError as exc:
        valid = [item for item in dir(enum_cls) if item.isupper()]
        raise ValueError(
            f"Invalid ZED enum value {name!r}; valid: {valid}"
        ) from exc


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="/tmp/depth_compare")
    p.add_argument("--name", type=str, default="real_depth")
    p.add_argument("--warmup-frames", type=int, default=5,
                   help="Discard this many first frames so ZED exposure settles.")
    p.add_argument("--timeout-s", type=float, default=10.0)
    p.add_argument("--serial", type=str, default=ZED_SERIAL_NUMBER)
    p.add_argument("--resolution", type=str, default=ZED_RESOLUTION)
    p.add_argument("--depth-mode", type=str, default=ZED_DEPTH_MODE)
    p.add_argument("--camera-fps", type=int, default=ZED_CAMERA_FPS)
    p.add_argument("--exposure", type=int, default=ZED_EXPOSURE)
    p.add_argument("--gain", type=int, default=ZED_GAIN)
    p.add_argument("--depth-near-m", type=float, default=DEPTH_NEAR_M)
    p.add_argument("--depth-far-m", type=float, default=DEPTH_FAR_M)
    p.add_argument("--raw-vis-near", type=float, default=0.30,
                   help="Lower bound (m) for the *_raw.png visualization stretch.")
    p.add_argument("--raw-vis-far", type=float, default=1.50,
                   help="Upper bound (m) for the *_raw.png visualization stretch.")
    p.add_argument("--camera-upsidedown", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        import pyzed.sl as sl
    except ModuleNotFoundError:
        print("[capture-real] pyzed.sl not installed in this venv.", flush=True)
        return 1

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    init = sl.InitParameters(input_t=sl.InputType())
    init.svo_real_time_mode = True
    init.camera_resolution = _zed_enum_value(sl.RESOLUTION, args.resolution)
    init.depth_mode = _zed_enum_value(sl.DEPTH_MODE, args.depth_mode)
    init.coordinate_units = sl.UNIT.MILLIMETER
    if args.camera_fps > 0:
        init.camera_fps = int(args.camera_fps)
    if args.serial:
        init.set_from_serial_number(int(args.serial))

    camera = sl.Camera()
    err = camera.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"[capture-real] failed to open ZED serial={args.serial}: {err}",
              flush=True)
        return 1
    if args.exposure >= 0:
        camera.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, int(args.exposure))
    if args.gain >= 0:
        camera.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, int(args.gain))

    retrieve_resolution = sl.Resolution(SIM_CAMERA_WIDTH, SIM_CAMERA_HEIGHT)
    depth_mat = sl.Mat()
    runtime = sl.RuntimeParameters()
    deadline = time.perf_counter() + args.timeout_s
    grabbed = 0

    print(
        f"[capture-real] opened ZED serial={args.serial} {args.resolution}@"
        f"{args.camera_fps}Hz depth_mode={args.depth_mode}; retrieving at "
        f"{SIM_CAMERA_WIDTH}x{SIM_CAMERA_HEIGHT}",
        flush=True,
    )

    try:
        while time.perf_counter() < deadline:
            if camera.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                time.sleep(1.0 / max(args.camera_fps, 1))
                continue
            try:
                camera.retrieve_measure(
                    depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU, retrieve_resolution
                )
            except TypeError:
                camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            depth_mm = np.array(depth_mat.get_data(), copy=True)
            if args.camera_upsidedown:
                import cv2

                depth_mm = cv2.flip(depth_mm, -1)
            if depth_mm.ndim == 3:
                depth_mm = depth_mm[..., 0]
            depth_m = depth_mm.astype(np.float32, copy=False) / 1000.0

            grabbed += 1
            if grabbed <= args.warmup_frames:
                continue

            # --- raw uncropped depth (in meters, NaN/inf preserved) ---
            raw_npz = out_dir / f"{args.name}_raw.npz"
            raw_png = out_dir / f"{args.name}_raw.png"
            np.savez_compressed(raw_npz, depth_m=depth_m)
            _save_raw_visualization_png(
                raw_png, depth_m, args.raw_vis_near, args.raw_vis_far
            )

            # --- policy-preprocessed 70x70 depth ---
            cropped = depth_m[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
            safe = np.nan_to_num(
                cropped, nan=args.depth_far_m,
                posinf=args.depth_far_m, neginf=args.depth_near_m,
            )
            policy = np.clip(
                (safe - args.depth_near_m)
                / max(args.depth_far_m - args.depth_near_m, 1e-6),
                0.0, 1.0,
            ).astype(np.float32)
            policy_npz = out_dir / f"{args.name}.npz"
            policy_png = out_dir / f"{args.name}.png"
            np.savez_compressed(policy_npz, image=policy)
            _save_normalized_png(policy_png, policy)

            valid = np.isfinite(depth_m)
            print(
                f"[capture-real] kept frame {grabbed}\n"
                f"  raw:    {raw_png}  shape={depth_m.shape} "
                f"  valid_m=[{float(depth_m[valid].min()) if valid.any() else float('nan'):.3f}, "
                f"{float(depth_m[valid].max()) if valid.any() else float('nan'):.3f}] "
                f"finite_frac={float(valid.mean()):.3f}\n"
                f"  policy: {policy_png} shape={policy.shape} "
                f"min={float(policy.min()):.3f} mean={float(policy.mean()):.3f} "
                f"max={float(policy.max()):.3f}",
                flush=True,
            )
            return 0
    finally:
        camera.close()

    print(f"[capture-real] timed out after {args.timeout_s}s with no good frame",
          flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
