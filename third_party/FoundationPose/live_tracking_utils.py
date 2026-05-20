"""Shared helpers for FoundationPose live tracking with the ZED camera.

NOTE: when using ROS, the caller must `import rospy` BEFORE importing this
module. Loading rospy AFTER the FoundationPose stack (which this module
imports transitively) reliably segfaults — some native lib in the FP stack
clobbers rospy's C bindings on subsequent import. The combined
`live_tracking.py` handles this via a `--ros in sys.argv` gate.
"""

import os
import sys
import datetime as _dt
import importlib.util
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import cv2
import pyzed.sl as sl
import trimesh
import imageio
import nvdiffrast.torch as dr
from estimater import *  # noqa: F401,F403  (FoundationPose, ScorePredictor, etc.)
from generate_mask import generate_binary_mask_box


# ----------------------------- camera helpers --------------------------------

def construct_camera_intrinsics(camera_params, target_width, target_height,
                                camera_upsidedown=False):
    fx = camera_params.fx * target_width / camera_params.image_size.width
    fy = camera_params.fy * target_height / camera_params.image_size.height
    cx = camera_params.cx * target_width / camera_params.image_size.width
    cy = camera_params.cy * target_height / camera_params.image_size.height
    if camera_upsidedown:
        cx = target_width - cx
        cy = target_height - cy
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def init_zed(serial_number: str, exposure: int, gain: int,
             resolution: str = 'HD1080', camera_fps: int = 0,
             depth_mode=sl.DEPTH_MODE.NEURAL,
             coordinate_units=sl.UNIT.MILLIMETER):
    """Open the ZED. `camera_fps=0` asks the SDK for that resolution's default
    max (30 / 60 / 100 for HD1080 / HD720 / VGA on a ZED 2). For Fast-FS use
    `depth_mode=sl.DEPTH_MODE.NONE` and `coordinate_units=sl.UNIT.METER`."""
    zed = sl.Camera()
    init_params = sl.InitParameters(input_t=sl.InputType())
    init_params.svo_real_time_mode = True
    init_params.camera_resolution = getattr(sl.RESOLUTION, resolution)
    init_params.depth_mode = depth_mode
    init_params.coordinate_units = coordinate_units
    if camera_fps:
        init_params.camera_fps = int(camera_fps)
    init_params.set_from_serial_number(int(serial_number))

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {err}")

    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, exposure)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, gain)

    runtime_parameters = sl.RuntimeParameters()
    return zed, runtime_parameters, sl.Mat(), sl.Mat()


def _zed_view_to_rgb(mat, width, height, camera_upsidedown):
    img = mat.get_data()
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    else:
        img = img[..., :3]
    if camera_upsidedown:
        img = cv2.flip(img, -1)
    return cv2.resize(img, (width, height))


def read_zed_depth_frame(zed, runtime_parameters, image_mat, depth_mat,
                         width, height, camera_upsidedown=False):
    """Grab LEFT + ZED neural depth. Depth from the ZED is in `coordinate_units`
    (mm if `init_zed` used the default); we convert to meters here."""
    if zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS:
        return None, None

    zed.retrieve_image(image_mat, sl.VIEW.LEFT)
    img_rgb = _zed_view_to_rgb(image_mat, width, height, camera_upsidedown)

    zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
    depth_mm = depth_mat.get_data()
    if camera_upsidedown:
        depth_mm = cv2.flip(depth_mm, -1)
    depth_mm = cv2.resize(depth_mm, (width, height), interpolation=cv2.INTER_NEAREST)
    depth_m = depth_mm.astype(np.float32) / 1000.0
    depth_m[(depth_m < 0.001) | (~np.isfinite(depth_m))] = 0.0
    return img_rgb, depth_m


def read_zed_stereo_frame(zed, runtime_parameters, left_mat, right_mat,
                          width, height, camera_upsidedown=False):
    """Grab synchronized rectified LEFT + RIGHT at (width, height)."""
    if zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS:
        return None, None
    zed.retrieve_image(left_mat, sl.VIEW.LEFT)
    zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
    left = _zed_view_to_rgb(left_mat, width, height, camera_upsidedown)
    right = _zed_view_to_rgb(right_mat, width, height, camera_upsidedown)
    return left, right


# ----------------------------- SAM mask --------------------------------------

def select_mask_with_sam(rgb):
    """Interactive SAM box-based mask. SAM helper expects BGR."""
    mask = generate_binary_mask_box(rgb[..., ::-1], polygon_refinement=True)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8)


# ----------------------------- ROS publisher ---------------------------------

class RosPosePublisher:
    """ROS publisher wrapper; instantiated only when --ros is passed.

    The caller is responsible for `import rospy` BEFORE importing this module
    (see module docstring). The `import rospy` inside __init__ is then a no-op
    since rospy is already in sys.modules, avoiding the segfault that comes
    from loading rospy after the FoundationPose stack.
    """

    def __init__(self, frame_id, publish_robot_frame):
        import rospy
        from geometry_msgs.msg import PoseStamped
        self._rospy = rospy
        self._PoseStamped = PoseStamped
        rospy.init_node('foundationpose_live_tracking', anonymous=True)
        self.frame_id = frame_id
        self.cam_pub = rospy.Publisher(
            'camera_frame/current_object_pose', PoseStamped, queue_size=1)
        self.robot_pub = (
            rospy.Publisher('robot_frame/current_object_pose',
                            PoseStamped, queue_size=1)
            if publish_robot_frame else None
        )

    def _msg(self, T, frame_id):
        msg = self._PoseStamped()
        msg.header.stamp = self._rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(T[0, 3])
        msg.pose.position.y = float(T[1, 3])
        msg.pose.position.z = float(T[2, 3])
        quat_wxyz = trimesh.transformations.quaternion_from_matrix(T)
        msg.pose.orientation.x = float(quat_wxyz[1])
        msg.pose.orientation.y = float(quat_wxyz[2])
        msg.pose.orientation.z = float(quat_wxyz[3])
        msg.pose.orientation.w = float(quat_wxyz[0])
        return msg

    def publish(self, pose_cam, T_RC=None):
        self.cam_pub.publish(self._msg(pose_cam, self.frame_id))
        if self.robot_pub is not None and T_RC is not None:
            self.robot_pub.publish(self._msg(T_RC @ pose_cam, 'robot_frame'))

    def shutdown_requested(self):
        return self._rospy.is_shutdown()


# ----------------------------- Fast-FS depth ---------------------------------

# Fast-FoundationStereo lives at <repo>/third_party/Fast-FoundationStereo/.
# This file is at <repo>/third_party/FoundationPose/live_tracking_utils.py,
# so the sibling vendored copy is one level up.
_FP_DIR = Path(__file__).resolve().parent
_FAST_FS_DIR = _FP_DIR.parent / 'Fast-FoundationStereo'

# Matches Fast-FoundationStereo/Utils.py:10. Inlined because both repos
# expose a top-level `Utils` module — see `_swap_in_fast_fs_utils` below.
import torch  # noqa: E402
_AMP_DTYPE = torch.float16


@contextmanager
def _swap_in_fast_fs_utils(fast_fs_dir: Path):
    """Temporarily swap sys.modules['Utils'] to Fast-FS's Utils.py.

    Both third_party/FoundationPose/Utils.py and
    third_party/Fast-FoundationStereo/Utils.py claim the module name `Utils`.
    `from estimater import *` at the top of this file populates
    sys.modules['Utils'] with FoundationPose's Utils — so when `torch.load`
    later triggers `import core.foundation_stereo` (which does
    `from Utils import AMP_DTYPE` deep in Fast-FS internals), the import fails
    because FP's Utils has no AMP_DTYPE. Swap in Fast-FS's Utils for the
    duration of the load, then put FP's Utils back. The loaded modules
    captured AMP_DTYPE at load time, so the swap is safe to undo after
    `torch.load` returns.
    """
    saved = sys.modules.get('Utils')
    spec = importlib.util.spec_from_file_location('Utils', fast_fs_dir / 'Utils.py')
    fs_utils = importlib.util.module_from_spec(spec)
    sys.modules['Utils'] = fs_utils
    spec.loader.exec_module(fs_utils)
    try:
        yield
    finally:
        if saved is not None:
            sys.modules['Utils'] = saved
        else:
            del sys.modules['Utils']


DEFAULT_FAST_FS_MODEL = _FAST_FS_DIR / 'weights' / '23-36-37' / 'model_best_bp2_serialize.pth'


class FastFsDepth:
    """Fast-FoundationStereo inference: stereo RGB -> depth (meters).

    Two backends:
      - **PyTorch** (default): loads ``model_best_bp2_serialize.pth`` via
        ``torch.load`` and calls ``model.forward(left, right, iters=...,
        test_mode=True)``. ~27 ms at 384x224 on RTX 6000 Ada.
      - **TensorRT** (when ``trt_dir`` is set): loads a pre-built engine pair
        (``feature_runner.engine`` + ``post_runner.engine``) and the matching
        ``onnx.yaml``. ~4 ms at 384x224 on RTX 6000 Ada — a ~7x speedup.

    Forward runs at ``fs_size``. With TRT, ``fs_size`` is forced to the
    engine's baked-in resolution (read from ``onnx.yaml``). Disparity is
    converted to depth using fx at that processing size, then resized up to
    ``fp_size`` (FoundationPose's input resolution).
    """

    def __init__(self, model_path, iters: int, max_disp: int,
                 baseline_m: float, fp_size, fs_size, fx_fp: float,
                 trt_dir=None):
        # sys.path is appended here (lazy) so importing live_tracking_utils
        # for non-fast_fs use doesn't require Fast-FS to be present.
        sys.path.insert(0, str(_FAST_FS_DIR))
        from core.utils.utils import InputPadder
        self._InputPadder = InputPadder

        import yaml
        from omegaconf import OmegaConf

        self.trt_dir = Path(trt_dir) if trt_dir else None
        self.iters = iters
        self.baseline_m = float(baseline_m)
        self.fp_size = tuple(fp_size)

        if self.trt_dir is not None:
            feat = self.trt_dir / 'feature_runner.engine'
            post = self.trt_dir / 'post_runner.engine'
            cfg_path = self.trt_dir / 'onnx.yaml'
            for p in (feat, post, cfg_path):
                if not p.is_file():
                    raise FileNotFoundError(
                        f"Fast-FS TRT artifact missing at {p}. Build engines "
                        f"per docs/venv_isaacsim_install.md §2c.")
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            self.args = OmegaConf.create(cfg)
            img_H, img_W = cfg['image_size']
            self.fs_size = (int(img_W), int(img_H))
            with _swap_in_fast_fs_utils(_FAST_FS_DIR):
                from core.foundation_stereo import TrtRunner
                self.model = TrtRunner(self.args, str(feat), str(post))
            self.backend = 'trt'
        else:
            if not Path(model_path).is_file():
                raise FileNotFoundError(
                    f"Fast-FS weights not found at {model_path}. Download per "
                    f"docs/venv_isaacsim_install.md §2b.")
            cfg_path = Path(model_path).parent / 'cfg.yaml'
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            cfg['valid_iters'] = iters
            cfg['max_disp'] = max_disp
            self.args = OmegaConf.create(cfg)
            with _swap_in_fast_fs_utils(_FAST_FS_DIR):
                self.model = torch.load(model_path, map_location='cpu',
                                        weights_only=False)
            self.model.args.valid_iters = iters
            self.model.args.max_disp = max_disp
            self.model.cuda().eval()
            self.backend = 'pytorch'
            self.fs_size = tuple(fs_size)

        # fx scales linearly with horizontal resolution.
        self.fx_fs = float(fx_fp) * self.fs_size[0] / self.fp_size[0]

    @torch.no_grad()
    def __call__(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        """Inputs at ``fp_size``. Returns depth (H_fp, W_fp) float32 in meters."""
        left_small = cv2.resize(left_rgb, self.fs_size, interpolation=cv2.INTER_AREA)
        right_small = cv2.resize(right_rgb, self.fs_size, interpolation=cv2.INTER_AREA)
        fs_W, fs_H = self.fs_size

        left = torch.as_tensor(left_small).cuda().float()[None].permute(0, 3, 1, 2)
        right = torch.as_tensor(right_small).cuda().float()[None].permute(0, 3, 1, 2)

        if self.backend == 'pytorch':
            padder = self._InputPadder(left.shape, divis_by=32, force_square=False)
            left, right = padder.pad(left, right)
            with torch.amp.autocast('cuda', enabled=True, dtype=_AMP_DTYPE):
                disp = self.model.forward(left, right, iters=self.iters,
                                          test_mode=True,
                                          optimize_build_volume='pytorch1')
            disp = padder.unpad(disp.float())
        else:
            disp = self.model.forward(left, right).float()

        disp = disp.cpu().numpy().reshape(fs_H, fs_W).clip(0, None)

        # Right-view correspondences that fall off the image are invalid.
        yy, xx = np.meshgrid(np.arange(fs_H), np.arange(fs_W), indexing='ij')
        invalid = (xx - disp) < 0
        depth_small = np.zeros((fs_H, fs_W), dtype=np.float32)
        valid = (disp > 1e-3) & (~invalid)
        depth_small[valid] = (self.fx_fs * self.baseline_m) / disp[valid]

        return cv2.resize(depth_small, self.fp_size, interpolation=cv2.INTER_NEAREST)


# ----------------------------- run output ------------------------------------

# Axis endpoints in object frame for the overlay (10 cm).
_AXIS_X = np.array([0.1, 0.0, 0.0, 1.0])
_AXIS_Y = np.array([0.0, 0.1, 0.0, 1.0])
_AXIS_Z = np.array([0.0, 0.0, 0.1, 1.0])
_ORIGIN_H = np.array([0.0, 0.0, 0.0, 1.0])


def _stage_stats_ms(times):
    t = np.asarray(times)
    return {
        'mean': float(t.mean() * 1e3),
        'p50': float(np.median(t) * 1e3),
        'p95': float(np.percentile(t, 95) * 1e3),
        'min': float(t.min() * 1e3),
        'max': float(t.max() * 1e3),
    }


def save_run(save_dir, backend_name, resolution,
             raw_frames, pose_log, K, loop_start_time, frame_times,
             target_fps, stage_times=None):
    """Write `raw.mp4` + `overlay.mp4` into a fresh datetimed subdir of
    `save_dir` and print loop-timing stats. Returns the run directory."""
    if not raw_frames or loop_start_time is None:
        return None

    elapsed = max(time.time() - loop_start_time, 1e-6)
    measured_fps = max(1, int(round(len(raw_frames) / elapsed)))
    stamp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(
        save_dir,
        f'{backend_name}_{resolution}_fps{measured_fps}_{stamp}')
    os.makedirs(run_dir, exist_ok=True)
    raw_path = os.path.join(run_dir, 'raw.mp4')
    overlay_path = os.path.join(run_dir, 'overlay.mp4')

    raw_writer = imageio.get_writer(raw_path, fps=measured_fps,
                                    codec='libx264', quality=8,
                                    macro_block_size=1)
    for f in raw_frames:
        raw_writer.append_data(f)
    raw_writer.close()

    overlay_writer = imageio.get_writer(overlay_path, fps=measured_fps,
                                        codec='libx264', quality=8,
                                        macro_block_size=1)
    for rgb_frame, pose in zip(raw_frames, pose_log):
        if pose is None or K is None:
            overlay_writer.append_data(rgb_frame)
            continue
        overlay = rgb_frame.copy()
        o = tuple(project_3d_to_2d(_ORIGIN_H, K, pose))
        px = tuple(project_3d_to_2d(_AXIS_X, K, pose))
        py = tuple(project_3d_to_2d(_AXIS_Y, K, pose))
        pz = tuple(project_3d_to_2d(_AXIS_Z, K, pose))
        cv2.arrowedLine(overlay, o, px, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.arrowedLine(overlay, o, py, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.arrowedLine(overlay, o, pz, (0, 0, 255), 3, cv2.LINE_AA)
        overlay_writer.append_data(overlay)
    overlay_writer.close()

    print(f'Saved: {raw_path} and {overlay_path} '
          f'({len(raw_frames)} frames @ {measured_fps} FPS over {elapsed:.2f}s)')

    if frame_times:
        ft = np.asarray(frame_times)
        fps_arr = 1.0 / np.clip(ft, 1e-6, None)
        hit_pct = float((fps_arr >= target_fps).mean() * 100.0)
        stages = {
            name: _stage_stats_ms(ts)
            for name, ts in (stage_times or {}).items() if ts
        }
        metrics = {
            'backend': backend_name,
            'resolution': resolution,
            'timestamp': stamp,
            'n_frames': int(len(raw_frames)),
            'elapsed_s': float(elapsed),
            'target_fps': float(target_fps),
            'achieved_fps': float(measured_fps),
            'per_frame_ms': _stage_stats_ms(frame_times),
            'stages_ms': stages,
            'hit_target_pct': hit_pct,
            'budget_ms': float(1e3 / target_fps),
        }
        metrics_path = os.path.join(run_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        lines = [
            '--- Loop timing (excludes sleep-to-target) ---',
            f'  target FPS:           {target_fps:.1f}',
            f'  achieved (wall) FPS:  {measured_fps:.1f}',
            f'  per-frame work (ms):  '
            f'mean={ft.mean()*1e3:.1f}  p50={np.median(ft)*1e3:.1f}  '
            f'p95={np.percentile(ft, 95)*1e3:.1f}  '
            f'min={ft.min()*1e3:.1f}  max={ft.max()*1e3:.1f}',
            f'  frames with work <= 1/target ({1e3/target_fps:.1f}ms): '
            f'{hit_pct:.1f}%',
        ]
        if stages:
            lines.append('  per-stage (ms): '
                         '         mean    p50    p95    min    max')
            for name, s in stages.items():
                lines.append(
                    f"    {name:<10s}{s['mean']:7.1f}{s['p50']:7.1f}"
                    f"{s['p95']:7.1f}{s['min']:7.1f}{s['max']:7.1f}")
        lines.append(f'  metrics: {metrics_path}')
        print('\n'.join(lines))
    return run_dir
