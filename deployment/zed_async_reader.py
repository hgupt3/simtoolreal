#!/usr/bin/env python
"""Async ZED depth reader for the depth-student policy node.

A child process owns ``pyzed.sl`` (the SDK is not fork-safe and likes its own
thread). The parent reads the latest preprocessed depth frame from
double-buffered shared memory in O(H*W) without blocking on camera I/O.

Capture pipeline (producer process):
  ZED raw 1920x1080 mm depth
    -> resize nearest to sim camera resolution (160x90)
    -> crop x=[90,160) y=[0,70) -> 70x70
    -> window-normalize against (depth_near_m, depth_far_m) to [0, 1].
  Frame written into double-buffered shared memory; consumer copies it
  out under a lock.

Output frame contract -- must match what the student saw at training time:
  - 1-channel float32, shape ``(image_input_height, image_input_width)``.
  - Window-normalized into ``[0, 1]``:
        ``clamp((d - depth_near) / (depth_far - depth_near), 0, 1)``
    against the trained window ``(depth_near, depth_far)``. For the
    camera_noise_ablations checkpoints these are 0.70 m and 1.10 m, baked
    into ``isaacsimenvs/cfg/task/PegInHoleDepthStudent.yaml``.
  - Crop+resize ordering matches ``_crop_student_image`` +
    ``_preprocess_student_depth`` in
    ``isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py``.

Run standalone for a smoke test:
    python deployment/zed_async_reader.py --self-test --duration-s 5
    python deployment/zed_async_reader.py --self-test --save-dir /tmp/zed_smoke

Both modes require a connected ZED on serial ``ZED_SERIAL_NUMBER`` (default
15107).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional

import numpy as np


# Match isaacsimenvs/cfg/task/PegInHoleDepthStudent.yaml at the time the
# camera_noise_ablations checkpoints were trained. Both can be overridden via
# constructor args / CLI.
DEPTH_NEAR_M = 0.70
DEPTH_FAR_M = 1.10
DEPTH_INVALID_M = 10.0     # in case the SDK returns NaN/inf, map to far+

# Sim camera renders at 160x90; we crop x=[90,160) y=[0,70) -> 70x70.
SIM_CAMERA_WIDTH = 160
SIM_CAMERA_HEIGHT = 90
CROP_X0 = 90
CROP_Y0 = 0
CROP_X1 = 160
CROP_Y1 = 70
POLICY_WIDTH = CROP_X1 - CROP_X0
POLICY_HEIGHT = CROP_Y1 - CROP_Y0

# ZED defaults — mirror v3 settings the team validated on serial 15107.
ZED_SERIAL_NUMBER = "15107"
# Capture at 1080p (1920x1080); resize-nearest to sim camera res (160x90)
# inside the producer before cropping. HD1080 maxes out at 30 FPS on the
# ZED 2; consumer reads at policy rate via shared memory.
ZED_RESOLUTION = "HD1080"
ZED_DEPTH_MODE = "NEURAL"
ZED_CAMERA_FPS = 30
ZED_EXPOSURE = 25
ZED_GAIN = 40


@dataclass
class ZedReaderConfig:
    serial_number: str = ZED_SERIAL_NUMBER
    resolution: str = ZED_RESOLUTION
    depth_mode: str = ZED_DEPTH_MODE
    camera_fps: int = ZED_CAMERA_FPS
    grab_hz: float = float(ZED_CAMERA_FPS)
    exposure: int = ZED_EXPOSURE
    gain: int = ZED_GAIN

    # Depth window — passed to producer so the consumer can read a fully
    # preprocessed frame without any further math on the policy hot path.
    depth_near_m: float = DEPTH_NEAR_M
    depth_far_m: float = DEPTH_FAR_M

    # Retrieve at the sim camera's native resolution; crop applied per pixel.
    sim_camera_width: int = SIM_CAMERA_WIDTH
    sim_camera_height: int = SIM_CAMERA_HEIGHT
    crop_x0: int = CROP_X0
    crop_y0: int = CROP_Y0
    crop_x1: int = CROP_X1
    crop_y1: int = CROP_Y1

    # If true, vertically + horizontally flip the depth frame after retrieve.
    # The lab's ZED has been mounted upside down on at least one rig in the
    # past; defaulting off — toggle if your mount differs.
    camera_upsidedown: bool = False


# ============================================================
# Producer (child process). Imports pyzed.sl only inside the
# process so the parent stays clean if pyzed is unavailable
# (e.g. unit-testing utility code).
# ============================================================


def _zed_enum_value(enum_cls, name: str):
    try:
        return getattr(enum_cls, str(name).upper())
    except AttributeError as exc:
        valid = [item for item in dir(enum_cls) if item.isupper()]
        raise ValueError(
            f"Invalid ZED enum value {name!r}; valid values include {valid}"
        ) from exc


def _resize_nearest(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    if depth.shape == (height, width):
        return depth.astype(np.float32, copy=False)
    import cv2

    return cv2.resize(
        depth, (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(np.float32, copy=False)


def _preprocess_depth(
    depth_mm: np.ndarray,
    cfg: ZedReaderConfig,
) -> np.ndarray:
    """ZED depth (mm, already at sim 160x90 via SDK resize-on-retrieve) ->
    policy depth (float32, [0,1], cropped to 70x70).

    Mirrors ``_preprocess_student_depth`` (window_normalize mode) +
    ``_crop_student_image`` in scene_utils.py.
    """
    if depth_mm.ndim == 3:
        depth_mm = depth_mm[..., 0]
    depth_mm = depth_mm.astype(np.float32, copy=False)
    depth_m = depth_mm / 1000.0

    # Sentinel cleanup: NaN/inf -> far end (clipped to 1.0 after normalize).
    finite = np.isfinite(depth_m)
    depth_m = np.where(
        finite,
        depth_m,
        np.full_like(depth_m, DEPTH_INVALID_M),
    )

    # 1. If the SDK couldn't resize-on-retrieve (older API), fall back to a
    #    numpy nearest resize to sim resolution. Common case: input already
    #    160x90 from the retrieve overload and this is a no-op.
    resized = _resize_nearest(
        depth_m, cfg.sim_camera_width, cfg.sim_camera_height
    )

    # 2. Crop to the student's policy window. x = column, y = row.
    cropped = resized[cfg.crop_y0:cfg.crop_y1, cfg.crop_x0:cfg.crop_x1]

    # 3. Window normalize -> [0, 1].
    near, far = cfg.depth_near_m, cfg.depth_far_m
    if far <= near:
        raise ValueError(
            f"depth_far_m ({far}) must be > depth_near_m ({near})"
        )
    safe = np.nan_to_num(cropped, nan=far, posinf=far, neginf=near)
    normalized = (safe - near) / (far - near)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _producer_main(
    cfg: ZedReaderConfig,
    shm_name: str,
    shape: tuple[int, int, int],
    state: "SharedZedState",
    stop_event: mp.Event,
) -> None:
    """Child process: open ZED, loop grab/retrieve/preprocess, write SHM."""
    try:
        import pyzed.sl as sl
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyzed.sl is required in the ZED producer process."
        ) from exc

    shm = shared_memory.SharedMemory(name=shm_name)
    shared = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
    camera = sl.Camera()
    depth_mat = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    grab_period_s = 1.0 / cfg.grab_hz if cfg.grab_hz > 0.0 else 0.0
    # Ask the SDK to deliver depth already at the sim camera resolution.
    # ZED scales the calibrated intrinsics linearly with the retrieve size,
    # so the FOV stays consistent with sim (verified ~69° H, ~43° V from
    # the lab's HD1080 intrinsics fx=fy=694.03, cx=480.1, cy=282.77).
    retrieve_resolution = sl.Resolution(
        int(cfg.sim_camera_width), int(cfg.sim_camera_height)
    )
    low_res_retrieve_failed = False

    try:
        init_params = sl.InitParameters(input_t=sl.InputType())
        init_params.svo_real_time_mode = True
        init_params.camera_resolution = _zed_enum_value(
            sl.RESOLUTION, cfg.resolution
        )
        init_params.depth_mode = _zed_enum_value(sl.DEPTH_MODE, cfg.depth_mode)
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        if cfg.camera_fps > 0:
            init_params.camera_fps = int(cfg.camera_fps)
        if cfg.serial_number:
            init_params.set_from_serial_number(int(cfg.serial_number))

        err = camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"Failed to open ZED camera serial={cfg.serial_number!r}: {err}"
            )
        if cfg.exposure >= 0:
            camera.set_camera_settings(
                sl.VIDEO_SETTINGS.EXPOSURE, int(cfg.exposure)
            )
        if cfg.gain >= 0:
            camera.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, int(cfg.gain))

        print(
            "[zed-producer] opened "
            f"serial={cfg.serial_number} resolution={cfg.resolution} "
            f"depth_mode={cfg.depth_mode} fps={cfg.camera_fps} "
            f"grab_hz={cfg.grab_hz} exposure={cfg.exposure} gain={cfg.gain}",
            flush=True,
        )

        local_frame_id = 0
        while not stop_event.is_set():
            tick_start = time.perf_counter()
            err = camera.grab(runtime_parameters)
            if err != sl.ERROR_CODE.SUCCESS:
                with state.failures.get_lock():
                    state.failures.value += 1
                stop_event.wait(min(grab_period_s, 0.005) if grab_period_s > 0 else 0.005)
                continue
            t_grab = time.perf_counter()

            # Retrieve at the sim camera resolution (default 160x90) directly,
            # using the SDK's resize-on-retrieve overload. Older SDKs that
            # don't accept the (mat, measure, mem, resolution) signature fall
            # back to full-frame retrieve and let _preprocess_depth resize.
            if low_res_retrieve_failed:
                camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            else:
                try:
                    camera.retrieve_measure(
                        depth_mat,
                        sl.MEASURE.DEPTH,
                        sl.MEM.CPU,
                        retrieve_resolution,
                    )
                except TypeError:
                    low_res_retrieve_failed = True
                    print(
                        "[zed-producer] resize-on-retrieve overload not "
                        "available; falling back to full-frame retrieve.",
                        flush=True,
                    )
                    camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            t_retrieve = time.perf_counter()

            depth_mm = np.array(depth_mat.get_data(), copy=True)
            if cfg.camera_upsidedown:
                import cv2

                depth_mm = cv2.flip(depth_mm, -1)
            processed = _preprocess_depth(depth_mm, cfg)
            t_pre = time.perf_counter()

            if processed.shape != (shape[1], shape[2]):
                raise RuntimeError(
                    "ZED producer preprocessed depth shape "
                    f"{processed.shape} != expected {(shape[1], shape[2])}"
                )

            with state.lock:
                next_idx = 1 - int(state.active_idx.value)
                shared[next_idx] = processed
                state.active_idx.value = next_idx
                state.frame_id.value = local_frame_id
                state.wall_time_s.value = t_pre
                state.grab_ms.value = 1000.0 * (t_grab - tick_start)
                state.retrieve_ms.value = 1000.0 * (t_retrieve - t_grab)
                state.preprocess_ms.value = 1000.0 * (t_pre - t_retrieve)
            local_frame_id += 1

            elapsed_s = time.perf_counter() - tick_start
            if grab_period_s > elapsed_s:
                stop_event.wait(grab_period_s - elapsed_s)
    finally:
        try:
            camera.close()
        finally:
            shm.close()


# ============================================================
# Shared state container
# ============================================================


@dataclass
class SharedZedState:
    active_idx: "mp.Value"   # 0 or 1, index of the most-recently-written buffer
    frame_id: "mp.Value"     # monotonically increasing
    wall_time_s: "mp.Value"  # perf_counter when frame was finished
    grab_ms: "mp.Value"
    retrieve_ms: "mp.Value"
    preprocess_ms: "mp.Value"
    failures: "mp.Value"
    lock: "mp.Lock"


def _make_shared_state() -> SharedZedState:
    return SharedZedState(
        active_idx=mp.Value("i", 0),
        frame_id=mp.Value("q", -1),       # -1 = no frame written yet
        wall_time_s=mp.Value("d", 0.0),
        grab_ms=mp.Value("d", 0.0),
        retrieve_ms=mp.Value("d", 0.0),
        preprocess_ms=mp.Value("d", 0.0),
        failures=mp.Value("q", 0),
        lock=mp.Lock(),
    )


# ============================================================
# Public API
# ============================================================


class ZedAsyncReader:
    """Owns a child process that owns the ZED SDK.

    ``start()`` spawns the producer. ``get_latest(timeout_s)`` returns the
    most recent preprocessed depth frame the producer has written. ``stop()``
    signals the producer to exit and unlinks the shared memory.

    Usage:
        reader = ZedAsyncReader()
        reader.start()
        try:
            depth, frame_id, age_s = reader.get_latest(timeout_s=0.02)
            ...
        finally:
            reader.stop()
    """

    def __init__(
        self,
        cfg: Optional[ZedReaderConfig] = None,
        *,
        policy_height: Optional[int] = None,
        policy_width: Optional[int] = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else ZedReaderConfig()
        # Allow caller to override the policy frame dims (e.g. read from yaml).
        # Crop coords stay as configured; only the output shape is checked.
        self.policy_height = int(
            policy_height
            if policy_height is not None
            else (self.cfg.crop_y1 - self.cfg.crop_y0)
        )
        self.policy_width = int(
            policy_width
            if policy_width is not None
            else (self.cfg.crop_x1 - self.cfg.crop_x0)
        )
        self._shape = (2, self.policy_height, self.policy_width)
        self._bytes = int(np.prod(self._shape) * np.dtype(np.float32).itemsize)

        self._shm: Optional[shared_memory.SharedMemory] = None
        self._shared_arr: Optional[np.ndarray] = None
        self._state: Optional[SharedZedState] = None
        self._stop_event: Optional[mp.Event] = None
        self._proc: Optional[mp.Process] = None
        self._started = False
        self._t_started: Optional[float] = None

    # --- lifecycle ---

    def start(self) -> None:
        if self._started:
            return
        # ZED SDK is not fork-safe; force spawn so the producer has a fresh
        # Python interpreter / GIL.
        try:
            mp.set_start_method("spawn", force=False)
        except RuntimeError:
            # Already set elsewhere -- fine.
            pass

        self._shm = shared_memory.SharedMemory(create=True, size=self._bytes)
        self._shared_arr = np.ndarray(
            self._shape, dtype=np.float32, buffer=self._shm.buf
        )
        self._shared_arr.fill(0.0)
        self._state = _make_shared_state()
        self._stop_event = mp.Event()

        ctx = mp.get_context("spawn")
        self._proc = ctx.Process(
            target=_producer_main,
            args=(self.cfg, self._shm.name, self._shape, self._state, self._stop_event),
            daemon=True,
        )
        self._proc.start()
        self._t_started = time.perf_counter()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        if self._proc is not None:
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2.0)
        if self._shm is not None:
            try:
                self._shm.close()
            finally:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
        self._shm = None
        self._shared_arr = None
        self._state = None
        self._stop_event = None
        self._proc = None
        self._started = False

    def __enter__(self) -> "ZedAsyncReader":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # --- consumer API ---

    def get_latest(
        self, timeout_s: float = 0.02
    ) -> tuple[np.ndarray, int, float]:
        """Return ``(depth (H,W) float32 in [0,1], frame_id, age_s)``.

        Blocks up to ``timeout_s`` for the first frame after ``start()``.
        Raises ``RuntimeError`` if no frame appears within the deadline.
        """
        if not self._started:
            raise RuntimeError("ZedAsyncReader.get_latest() called before start()")
        deadline = time.perf_counter() + max(0.0, float(timeout_s))
        while True:
            assert self._state is not None and self._shared_arr is not None
            with self._state.lock:
                frame_id = int(self._state.frame_id.value)
                if frame_id >= 0:
                    idx = int(self._state.active_idx.value)
                    frame = self._shared_arr[idx].copy()
                    wall_time_s = float(self._state.wall_time_s.value)
                    break
            if time.perf_counter() >= deadline:
                raise RuntimeError(
                    "ZedAsyncReader: no frame received within "
                    f"{timeout_s:.3f}s of start (failures="
                    f"{int(self._state.failures.value)})"
                )
            time.sleep(0.001)
        age_s = max(0.0, time.perf_counter() - wall_time_s)
        return frame, frame_id, age_s

    def stats(self) -> dict[str, float]:
        """Snapshot of producer-side timing counters (ms)."""
        if not self._started or self._state is None:
            return {}
        with self._state.lock:
            return {
                "frame_id": float(self._state.frame_id.value),
                "grab_ms": float(self._state.grab_ms.value),
                "retrieve_ms": float(self._state.retrieve_ms.value),
                "preprocess_ms": float(self._state.preprocess_ms.value),
                "failures": float(self._state.failures.value),
            }


# ============================================================
# CLI: standalone smoke test
# ============================================================


def _save_frame(save_dir: Path, frame_id: int, frame: np.ndarray) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    np.savez_compressed(save_dir / f"zed_frame_{frame_id:06d}.npz", image=frame)
    Image.fromarray(
        (np.clip(frame, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    ).save(save_dir / f"zed_frame_{frame_id:06d}.png")


def _self_test(args: argparse.Namespace) -> int:
    cfg = ZedReaderConfig(
        serial_number=args.serial,
        resolution=args.resolution,
        depth_mode=args.depth_mode,
        camera_fps=args.camera_fps,
        grab_hz=args.grab_hz,
        exposure=args.exposure,
        gain=args.gain,
        depth_near_m=args.depth_near_m,
        depth_far_m=args.depth_far_m,
        camera_upsidedown=args.camera_upsidedown,
    )
    save_dir = Path(args.save_dir) if args.save_dir else None
    last_saved_id = -1
    consumer_hz = max(1.0, args.consumer_hz)
    consumer_dt = 1.0 / consumer_hz
    end_time = time.perf_counter() + args.duration_s
    n_reads = 0
    n_reused = 0
    last_frame_id = -1

    with ZedAsyncReader(cfg) as reader:
        next_tick = time.perf_counter()
        next_status = time.perf_counter() + args.status_interval_s
        while time.perf_counter() < end_time:
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(min(0.002, next_tick - now))
                continue
            try:
                frame, frame_id, age_s = reader.get_latest(timeout_s=0.05)
            except RuntimeError as exc:
                print(f"[zed-smoke] {exc}", flush=True)
                next_tick += consumer_dt
                continue
            n_reads += 1
            if frame_id == last_frame_id:
                n_reused += 1
            last_frame_id = frame_id
            if save_dir is not None and (frame_id - last_saved_id) >= args.save_every:
                _save_frame(save_dir, frame_id, frame)
                last_saved_id = frame_id
            if time.perf_counter() >= next_status:
                stats = reader.stats()
                print(
                    f"[zed-smoke] reads={n_reads} reused={n_reused} "
                    f"last_id={frame_id} age_ms={1000 * age_s:.1f} "
                    f"min={float(frame.min()):.3f} mean={float(frame.mean()):.3f} "
                    f"max={float(frame.max()):.3f} | producer "
                    f"grab={stats.get('grab_ms', 0.0):.2f}ms "
                    f"retrieve={stats.get('retrieve_ms', 0.0):.2f}ms "
                    f"pre={stats.get('preprocess_ms', 0.0):.2f}ms "
                    f"failures={int(stats.get('failures', 0))}",
                    flush=True,
                )
                next_status = time.perf_counter() + args.status_interval_s
            next_tick += consumer_dt

    print(f"[zed-smoke] done; consumer reads={n_reads} reused={n_reused}", flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true",
                   help="Open ZED, stream preprocessed frames for --duration-s.")
    p.add_argument("--duration-s", type=float, default=5.0)
    p.add_argument("--consumer-hz", type=float, default=60.0)
    p.add_argument("--status-interval-s", type=float, default=1.0)
    p.add_argument("--save-dir", type=str, default=None,
                   help="If set, write every --save-every-th frame as PNG+npz.")
    p.add_argument("--save-every", type=int, default=15)

    p.add_argument("--serial", type=str, default=ZED_SERIAL_NUMBER)
    p.add_argument("--resolution", type=str, default=ZED_RESOLUTION)
    p.add_argument("--depth-mode", type=str, default=ZED_DEPTH_MODE)
    p.add_argument("--camera-fps", type=int, default=ZED_CAMERA_FPS)
    p.add_argument("--grab-hz", type=float, default=float(ZED_CAMERA_FPS))
    p.add_argument("--exposure", type=int, default=ZED_EXPOSURE)
    p.add_argument("--gain", type=int, default=ZED_GAIN)
    p.add_argument("--depth-near-m", type=float, default=DEPTH_NEAR_M)
    p.add_argument("--depth-far-m", type=float, default=DEPTH_FAR_M)
    p.add_argument("--camera-upsidedown", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.self_test:
        return _self_test(args)
    print("Nothing to do; pass --self-test to run the standalone smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
