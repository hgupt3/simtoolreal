"""Live 6-DoF object pose tracking with depth from the ZED SDK's neural depth.

Sister script: ``live_tracking_fast_fs.py`` uses Fast-FoundationStereo on the
ZED's left+right images instead of the ZED's onboard depth.

Pass ``--ros`` to publish ``PoseStamped`` on:
  - ``camera_frame/current_object_pose``
  - ``robot_frame/current_object_pose``  (only if ``--calibration`` is given)

Without ``--ros`` the script just runs the FoundationPose pipeline against
the live ZED feed; useful for bring-up + ``--debug 1`` overlays.
"""

import os
import time
import argparse
import numpy as np
import cv2
import pyzed.sl as sl
import trimesh
import nvdiffrast.torch as dr
import imageio
from estimater import *
from generate_mask import generate_binary_mask_box


def construct_camera_intrinsics(camera_params, target_width, target_height, camera_upsidedown=False):
    """3x3 K scaled to the target resolution; flip principal point if camera is upside-down."""
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
    """Open the ZED. `resolution` is one of 'HD1080', 'HD720', 'VGA'. `camera_fps=0`
    asks the SDK for that resolution's default max (30 / 60 / 100 on a ZED 2)."""
    zed = sl.Camera()
    init_params = sl.InitParameters(input_t=sl.InputType())
    init_params.svo_real_time_mode = True
    init_params.camera_resolution = getattr(sl.RESOLUTION, resolution)
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    if camera_fps:
        init_params.camera_fps = int(camera_fps)
    init_params.set_from_serial_number(int(serial_number))

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {err}")

    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, exposure)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, gain)

    runtime_parameters = sl.RuntimeParameters()
    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    return zed, runtime_parameters, image_mat, depth_mat


def read_rgbd_frame(zed, runtime_parameters, image_mat, depth_mat, width, height, camera_upsidedown=False):
    if zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS:
        return None, None

    zed.retrieve_image(image_mat, sl.VIEW.LEFT)
    img = image_mat.get_data()
    if img.ndim == 3 and img.shape[2] == 4:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=-1)
    else:
        img_rgb = img[..., :3]

    if camera_upsidedown:
        img_rgb = cv2.flip(img_rgb, -1)
    img_rgb = cv2.resize(img_rgb, (width, height))

    zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
    depth_mm = depth_mat.get_data()
    if camera_upsidedown:
        depth_mm = cv2.flip(depth_mm, -1)
    depth_mm = cv2.resize(depth_mm, (width, height), interpolation=cv2.INTER_NEAREST)
    depth_m = depth_mm.astype(np.float32) / 1000.0
    depth_m[(depth_m < 0.001) | (~np.isfinite(depth_m))] = 0.0

    return img_rgb, depth_m


def select_mask_with_sam(rgb):
    """Interactive SAM box-based mask. SAM helper expects BGR; convert from RGB."""
    mask = generate_binary_mask_box(rgb[..., ::-1], polygon_refinement=True)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8)


class RosPosePublisher:
    """ROS publisher wrapper; instantiated only when --ros is passed."""

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
    parser = argparse.ArgumentParser(description="Live 6D object pose tracking (ZED depth).")
    parser.add_argument('--mesh_path', type=str, required=True,
                        help='Path to object mesh file (.obj, in meters)')
    parser.add_argument('--ros', action='store_true',
                        help='Publish PoseStamped on camera_frame/current_object_pose '
                             '(and robot_frame/current_object_pose if --calibration is set).')
    parser.add_argument('--calibration', type=str, default=None,
                        help='Optional path to 4x4 camera-to-robot transform '
                             '(.npy or .txt). Only used when --ros is set.')
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
    parser.add_argument('--est_refine_iter', type=int, default=5,
                        help='Refinement iterations for initial registration')
    parser.add_argument('--track_refine_iter', type=int, default=2,
                        help='Refinement iterations per tracking step')
    parser.add_argument('--debug', type=int, default=0,
                        help='Debug level (0=off, 1=show vis)')
    parser.add_argument('--save_dir', type=str, default='live_tracking_results')
    parser.add_argument('--frame_id', type=str, default='camera_frame')
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
        cv2.namedWindow('FoundationPose Live (ZED depth)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('FoundationPose Live (ZED depth)', 640, 360)

    zed, runtime_parameters, image_mat, depth_mat = init_zed(
        args.serial_number, args.exposure, args.gain,
        resolution=args.camera_resolution, camera_fps=args.camera_fps)

    vis_array = []
    try:
        left_cam_params = zed.get_camera_information() \
            .camera_configuration.calibration_parameters.left_cam
        K = construct_camera_intrinsics(left_cam_params, args.width, args.height,
                                        args.camera_upsidedown)

        mesh = trimesh.load(args.mesh_path, process=False)
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        est = FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
            mesh=mesh, scorer=scorer, refiner=refiner,
            debug=args.debug, glctx=glctx, debug_dir=args.save_dir,
        )

        rgb, depth = read_rgbd_frame(zed, runtime_parameters, image_mat, depth_mat,
                                     args.width, args.height, args.camera_upsidedown)
        mask = select_mask_with_sam(rgb)
        if mask is None or mask.sum() == 0:
            print('Empty ROI selected. Exiting.')
            return

        print('Registering initial pose...')
        t0 = time.time()
        pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask.astype(bool),
                            iteration=args.est_refine_iter)
        print(f'Initial registration done in {time.time()-t0:.3f}s')
        if ros_pub is not None:
            ros_pub.publish(pose, T_RC)

        target_frame_time = 1.0 / args.fps if args.fps > 0 else 0.0

        def keep_running():
            return ros_pub.shutdown_requested() is False if ros_pub is not None else True

        while keep_running():
            loop_start = time.time()
            rgb, depth = read_rgbd_frame(zed, runtime_parameters, image_mat, depth_mat,
                                         args.width, args.height, args.camera_upsidedown)
            pose = est.track_one(rgb=rgb, depth=depth, K=K,
                                 iteration=args.track_refine_iter)
            if ros_pub is not None:
                ros_pub.publish(pose, T_RC)

            if args.debug >= 1:
                if pose is not None:
                    vis = draw_xyz_axis(rgb, ob_in_cam=pose, scale=0.1, K=K,
                                        thickness=3, transparency=0, is_input_rgb=True)
                    vis_bgr = vis[..., ::-1]
                    vis_array.append(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB))
                else:
                    vis_bgr = rgb[..., ::-1]
                cv2.imshow('FoundationPose Live (ZED depth)', vis_bgr)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            elapsed = time.time() - loop_start
            print(f"Tracking FPS: {1.0/elapsed:.1f}")
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
