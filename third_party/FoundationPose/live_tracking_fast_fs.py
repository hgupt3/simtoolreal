"""Live 6-DoF object pose tracking with depth from Fast-FoundationStereo.

Sister script: ``live_tracking_zed_depth.py`` uses the ZED SDK's onboard
neural depth instead of stereo matching from the left+right images.

This variant grabs the ZED's left+right rectified images, runs Fast-FS to
produce a per-pixel disparity, and converts to depth via
``depth = fx * baseline / disp``. The resulting depth is fed to
FoundationPose's normal ``register()`` / ``track_one()`` flow.

Pass ``--ros`` to publish ``PoseStamped`` on:
  - ``camera_frame/current_object_pose``
  - ``robot_frame/current_object_pose``  (only if ``--calibration`` is given)

Defaults to Fast-FS PyTorch inference (model_best_bp2_serialize.pth). TRT
deployment is documented in ``docs/venv_isaacsim_install.md`` §2c — once
engines are built, plumb them in by passing ``--fast_fs_trt_dir``.
"""

import os
import sys
import time
import argparse
import importlib.util
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import cv2
import pyzed.sl as sl
import trimesh
import torch
import nvdiffrast.torch as dr
import imageio
import yaml
from omegaconf import OmegaConf

from estimater import *
from generate_mask import generate_binary_mask_box


@contextmanager
def _swap_in_fast_fs_utils(fast_fs_dir: Path):
    """Temporarily swap sys.modules['Utils'] to Fast-FS's Utils.py.

    Both third_party/FoundationPose/Utils.py and third_party/Fast-FoundationStereo/Utils.py
    claim the module name `Utils`. `from estimater import *` at the top of this
    file populates sys.modules['Utils'] with FoundationPose's Utils — so when
    `torch.load` later triggers `import core.foundation_stereo` (which does
    `from Utils import AMP_DTYPE` deep in Fast-FS internals like core/submodule.py
    and core/distill_block.py), the import fails because FP's Utils has no
    AMP_DTYPE. Swap in Fast-FS's Utils for the duration of the load, then put
    FP's Utils back. The loaded modules captured AMP_DTYPE at load time, so the
    swap is safe to undo after `torch.load` returns.
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


# Fast-FoundationStereo lives at <repo>/third_party/Fast-FoundationStereo/.
# This file is at <repo>/third_party/FoundationPose/live_tracking_fast_fs.py,
# so the sibling vendored copy is one ../.
_FP_DIR = Path(__file__).resolve().parent
_FAST_FS_DIR = _FP_DIR.parent / 'Fast-FoundationStereo'
sys.path.insert(0, str(_FAST_FS_DIR))
from core.utils.utils import InputPadder  # noqa: E402

# Both third_party/FoundationPose/Utils.py and third_party/Fast-FoundationStereo/Utils.py
# claim the module name `Utils`. The former wins because `from estimater import *`
# above already imported it, so we inline Fast-FS's AMP_DTYPE rather than try to
# `from Utils import AMP_DTYPE`.
AMP_DTYPE = torch.float16  # matches Fast-FoundationStereo/Utils.py:10

DEFAULT_FAST_FS_MODEL = _FAST_FS_DIR / 'weights' / '23-36-37' / 'model_best_bp2_serialize.pth'


def construct_camera_intrinsics(camera_params, target_width, target_height, camera_upsidedown=False):
    fx = camera_params.fx * target_width / camera_params.image_size.width
    fy = camera_params.fy * target_height / camera_params.image_size.height
    cx = camera_params.cx * target_width / camera_params.image_size.width
    cy = camera_params.cy * target_height / camera_params.image_size.height

    if camera_upsidedown:
        cx = target_width - cx
        cy = target_height - cy

    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def init_zed(serial_number: str, exposure: int, gain: int,
             resolution: str = 'HD1080', camera_fps: int = 0):
    """Open the ZED with DEPTH disabled (Fast-FS computes its own).
    `resolution` is one of 'HD1080', 'HD720', 'VGA'. `camera_fps=0` asks the SDK
    for that resolution's default max (30 / 60 / 100 on a ZED 2)."""
    zed = sl.Camera()
    init_params = sl.InitParameters(input_t=sl.InputType())
    init_params.svo_real_time_mode = True
    init_params.camera_resolution = getattr(sl.RESOLUTION, resolution)
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # we run our own stereo
    init_params.coordinate_units = sl.UNIT.METER
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


def read_stereo_frame(zed, runtime_parameters, left_mat, right_mat,
                      width, height, camera_upsidedown=False):
    """Grab synchronized left+right rectified RGB at (width, height)."""
    if zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS:
        return None, None
    zed.retrieve_image(left_mat, sl.VIEW.LEFT)
    zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
    left = _zed_view_to_rgb(left_mat, width, height, camera_upsidedown)
    right = _zed_view_to_rgb(right_mat, width, height, camera_upsidedown)
    return left, right


class FastFsDepth:
    """Fast-FoundationStereo inference: stereo RGB -> depth (meters).

    Two backends:
      - **PyTorch** (default): loads ``model_best_bp2_serialize.pth`` via
        ``torch.load`` and calls ``model.forward(left, right, iters=..., test_mode=True)``.
        ~27 ms at 384x224 on RTX 6000 Ada.
      - **TensorRT** (when ``trt_dir`` is set): loads a pre-built engine pair
        (``feature_runner.engine`` + ``post_runner.engine``) and the matching
        ``onnx.yaml``. ~4 ms at 384x224 on RTX 6000 Ada — a ~7x speedup.
        Build engines per ``docs/venv_isaacsim_install.md`` §2c.

    The Fast-FS forward pass runs at ``fs_size``. With TRT, ``fs_size`` is
    forced to the engine's baked-in resolution (read from ``onnx.yaml``).
    The resulting disparity is converted to depth using the fx at that
    processing size, then resized up to ``fp_size`` (FoundationPose's input
    resolution).
    """

    def __init__(self, model_path: Path, iters: int, max_disp: int,
                 baseline_m: float, fp_size, fs_size, fx_fp: float,
                 trt_dir=None):
        self.trt_dir = Path(trt_dir) if trt_dir else None
        self.iters = iters
        self.baseline_m = float(baseline_m)
        self.fp_size = tuple(fp_size)   # (W, H) where FP runs

        if self.trt_dir is not None:
            self._init_trt(self.trt_dir)
        else:
            self._init_pytorch(model_path, iters, max_disp, fs_size)

        # fx scales linearly with horizontal resolution since the principal
        # ray geometry is invariant under proportional resampling.
        self.fx_fs = float(fx_fp) * self.fs_size[0] / self.fp_size[0]

    def _init_pytorch(self, model_path, iters, max_disp, fs_size):
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

        # The serialized .pth is a pickled FastFoundationStereo nn.Module,
        # not a state_dict. weights_only=False is required. Wrap the load in
        # the Utils-swap so Fast-FS's internal `from Utils import AMP_DTYPE`
        # resolves against Fast-FS's Utils, not FoundationPose's.
        with _swap_in_fast_fs_utils(_FAST_FS_DIR):
            self.model = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model.args.valid_iters = iters
        self.model.args.max_disp = max_disp
        self.model.cuda().eval()
        self.backend = 'pytorch'
        self.fs_size = tuple(fs_size)

    def _init_trt(self, trt_dir: Path):
        feat = trt_dir / 'feature_runner.engine'
        post = trt_dir / 'post_runner.engine'
        cfg_path = trt_dir / 'onnx.yaml'
        for p in (feat, post, cfg_path):
            if not p.is_file():
                raise FileNotFoundError(
                    f"Fast-FS TRT artifact missing at {p}. Build engines per "
                    f"docs/venv_isaacsim_install.md §2c.")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        self.args = OmegaConf.create(cfg)

        # image_size is [H, W] in onnx.yaml; engine is built for that exact size.
        img_H, img_W = cfg['image_size']
        self.fs_size = (int(img_W), int(img_H))

        with _swap_in_fast_fs_utils(_FAST_FS_DIR):
            from core.foundation_stereo import TrtRunner  # lazy: requires Utils swap
            self.model = TrtRunner(self.args, str(feat), str(post))
        self.backend = 'trt'

    @torch.no_grad()
    def __call__(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        """Inputs at ``fp_size`` (the size returned by ``read_stereo_frame``).
        Returns depth (H_fp, W_fp) float32 in meters, ready for FoundationPose."""
        # Downsize to fs_size for the forward pass.
        left_small = cv2.resize(left_rgb, self.fs_size, interpolation=cv2.INTER_AREA)
        right_small = cv2.resize(right_rgb, self.fs_size, interpolation=cv2.INTER_AREA)
        fs_W, fs_H = self.fs_size

        left = torch.as_tensor(left_small).cuda().float()[None].permute(0, 3, 1, 2)
        right = torch.as_tensor(right_small).cuda().float()[None].permute(0, 3, 1, 2)

        if self.backend == 'pytorch':
            padder = InputPadder(left.shape, divis_by=32, force_square=False)
            left, right = padder.pad(left, right)
            with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
                disp = self.model.forward(left, right, iters=self.iters,
                                          test_mode=True, optimize_build_volume='pytorch1')
            disp = padder.unpad(disp.float())
        else:  # trt
            # TRT engines are built for an exact shape (already a multiple of 32
            # via make_onnx.py), so no padding needed.
            disp = self.model.forward(left, right).float()

        disp = disp.cpu().numpy().reshape(fs_H, fs_W).clip(0, None)

        # Fast-FS marks pixels where the right-view correspondence falls off
        # the image (left-side occlusion / out-of-frame) as invisible. Treat
        # them as 0 depth so FoundationPose ignores them.
        yy, xx = np.meshgrid(np.arange(fs_H), np.arange(fs_W), indexing='ij')
        invalid = (xx - disp) < 0
        depth_small = np.zeros((fs_H, fs_W), dtype=np.float32)
        valid = (disp > 1e-3) & (~invalid)
        depth_small[valid] = (self.fx_fs * self.baseline_m) / disp[valid]

        # Upsample depth to fp_size for FoundationPose. NEAREST so we don't
        # smear depth across object/background boundaries.
        return cv2.resize(depth_small, self.fp_size, interpolation=cv2.INTER_NEAREST)


def select_mask_with_sam(rgb):
    mask = generate_binary_mask_box(rgb[..., ::-1], polygon_refinement=True)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8)


class RosPosePublisher:
    def __init__(self, frame_id, publish_robot_frame):
        import rospy
        from geometry_msgs.msg import PoseStamped
        self._rospy = rospy
        self._PoseStamped = PoseStamped
        rospy.init_node('foundationpose_live_tracking', anonymous=True)
        self.frame_id = frame_id
        self.cam_pub = rospy.Publisher('camera_frame/current_object_pose', PoseStamped, queue_size=1)
        self.robot_pub = (
            rospy.Publisher('robot_frame/current_object_pose', PoseStamped, queue_size=1)
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


def main():
    parser = argparse.ArgumentParser(description="Live 6D object pose tracking (Fast-FS depth).")
    parser.add_argument('--mesh_path', type=str, required=True,
                        help='Path to object mesh file (.obj, in meters)')
    parser.add_argument('--ros', action='store_true',
                        help='Publish PoseStamped on camera_frame/current_object_pose '
                             '(and robot_frame/current_object_pose if --calibration is set).')
    parser.add_argument('--calibration', type=str, default=None,
                        help='Optional 4x4 camera-to-robot transform (.npy or .txt). '
                             'Only used when --ros is set.')
    parser.add_argument('--serial_number', type=str, default='15107')
    parser.add_argument('--camera_upsidedown', action='store_true')
    parser.add_argument('--camera_resolution', choices=['HD1080', 'HD720', 'VGA'],
                        default='HD1080',
                        help='ZED capture resolution. HD1080=30fps max, HD720=60fps, VGA=100fps.')
    parser.add_argument('--camera_fps', type=int, default=0,
                        help='ZED capture FPS. 0 = use the resolution default '
                             '(30/60/100 for HD1080/HD720/VGA on ZED 2).')
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--exposure', type=int, default=25)
    parser.add_argument('--gain', type=int, default=40)
    parser.add_argument('--fps', type=float, default=40.0,
                        help='Target tracking-loop FPS (distinct from --camera_fps).')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=0,
                        help='Debug level (0=off, 1=show vis with depth heatmap)')
    parser.add_argument('--save_dir', type=str, default='live_tracking_results')
    parser.add_argument('--frame_id', type=str, default='camera_frame')

    # Fast-FS-specific knobs
    parser.add_argument('--fast_fs_model', type=str, default=str(DEFAULT_FAST_FS_MODEL),
                        help='Path to Fast-FS model_best_bp2_serialize.pth (PyTorch path)')
    parser.add_argument('--fast_fs_iters', type=int, default=4,
                        help='Fast-FS refinement iterations (ignored in TRT mode — the '
                             'engine bakes the iter count in at build time).')
    parser.add_argument('--fast_fs_max_disp', type=int, default=192)
    parser.add_argument('--fast_fs_width', type=int, default=384,
                        help='Fast-FS processing width (forced to engine size in TRT mode).')
    parser.add_argument('--fast_fs_height', type=int, default=224,
                        help='Fast-FS processing height (forced to engine size in TRT mode).')
    parser.add_argument('--fast_fs_trt_dir', type=str, default=None,
                        help='Directory containing feature_runner.engine, post_runner.engine, '
                             'and onnx.yaml from docs/venv_isaacsim_install.md §2c. When set, '
                             'uses the TRT path (~4 ms vs ~27 ms PyTorch on RTX 6000 Ada).')
    args = parser.parse_args()

    T_RC = None
    if args.calibration is not None:
        if not args.ros:
            print('[warn] --calibration ignored (only used with --ros)')
        else:
            arr = np.load(args.calibration) if args.calibration.endswith('.npy') \
                  else np.loadtxt(args.calibration)
            T_RC = arr.reshape(4, 4)

    ros_pub = RosPosePublisher(args.frame_id, publish_robot_frame=(T_RC is not None)) \
              if args.ros else None

    if args.debug >= 1:
        cv2.namedWindow('FoundationPose Live (Fast-FS depth)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('FoundationPose Live (Fast-FS depth)', 640, 360)

    zed, runtime_parameters, left_mat, right_mat = init_zed(
        args.serial_number, args.exposure, args.gain,
        resolution=args.camera_resolution, camera_fps=args.camera_fps)

    vis_array = []
    try:
        camera_info = zed.get_camera_information()
        left_cam_params = camera_info.camera_configuration.calibration_parameters.left_cam
        K = construct_camera_intrinsics(left_cam_params, args.width, args.height,
                                        args.camera_upsidedown)
        # Baseline in meters. coordinate_units=METER above, so get_camera_baseline()
        # already returns meters. abs() because some ZED firmwares report it signed.
        baseline_m = abs(camera_info.camera_configuration
                         .calibration_parameters.get_camera_baseline())
        fx_target = K[0, 0]
        print(f'ZED stereo: fx={fx_target:.2f} px at {args.width}x{args.height}, '
              f'baseline={baseline_m*1000:.2f} mm')

        fast_fs = FastFsDepth(
            model_path=args.fast_fs_model,
            iters=args.fast_fs_iters,
            max_disp=args.fast_fs_max_disp,
            baseline_m=baseline_m,
            fp_size=(args.width, args.height),
            fs_size=(args.fast_fs_width, args.fast_fs_height),
            fx_fp=fx_target,
            trt_dir=args.fast_fs_trt_dir,
        )
        print(f'Fast-FS backend: {fast_fs.backend} at {fast_fs.fs_size}; '
              f'depth upsampled to {fast_fs.fp_size} for FP')

        mesh = trimesh.load(args.mesh_path, process=False)
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        est = FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
            mesh=mesh, scorer=scorer, refiner=refiner,
            debug=args.debug, glctx=glctx, debug_dir=args.save_dir,
        )

        left, right = read_stereo_frame(zed, runtime_parameters, left_mat, right_mat,
                                        args.width, args.height, args.camera_upsidedown)
        depth = fast_fs(left, right)
        mask = select_mask_with_sam(left)
        if mask is None or mask.sum() == 0:
            print('Empty ROI selected. Exiting.')
            return

        print('Registering initial pose...')
        t0 = time.time()
        pose = est.register(K=K, rgb=left, depth=depth, ob_mask=mask.astype(bool),
                            iteration=args.est_refine_iter)
        print(f'Initial registration done in {time.time()-t0:.3f}s')
        if ros_pub is not None:
            ros_pub.publish(pose, T_RC)

        target_frame_time = 1.0 / args.fps if args.fps > 0 else 0.0

        def keep_running():
            return ros_pub.shutdown_requested() is False if ros_pub is not None else True

        while keep_running():
            loop_start = time.time()
            left, right = read_stereo_frame(zed, runtime_parameters, left_mat, right_mat,
                                            args.width, args.height, args.camera_upsidedown)
            depth = fast_fs(left, right)
            pose = est.track_one(rgb=left, depth=depth, K=K,
                                 iteration=args.track_refine_iter)
            if ros_pub is not None:
                ros_pub.publish(pose, T_RC)

            if args.debug >= 1:
                if pose is not None:
                    vis = draw_xyz_axis(left, ob_in_cam=pose, scale=0.1, K=K,
                                        thickness=3, transparency=0, is_input_rgb=True)
                else:
                    vis = left
                # Depth heatmap (clip to 0.2-1.5 m for visualization)
                d_clip = np.clip(depth, 0.2, 1.5)
                d_norm = ((d_clip - 0.2) / (1.5 - 0.2) * 255).astype(np.uint8)
                d_vis = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
                d_vis[depth <= 0] = 0
                vis_combined = np.concatenate([vis[..., ::-1], d_vis], axis=1)
                cv2.imshow('FoundationPose Live (Fast-FS depth)', vis_combined)
                vis_array.append(cv2.cvtColor(vis_combined, cv2.COLOR_BGR2RGB))
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            elapsed = time.time() - loop_start
            print(f"Tracking FPS: {1.0/elapsed:.1f} (Fast-FS+FP per frame {elapsed*1000:.1f} ms)")
            sleep_time = target_frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        zed.close()
        cv2.destroyAllWindows()
        if vis_array:
            os.makedirs(args.save_dir, exist_ok=True)
            imageio.mimsave(f'{args.save_dir}/vis.mp4', vis_array, fps=10)


if __name__ == '__main__':
    main()
