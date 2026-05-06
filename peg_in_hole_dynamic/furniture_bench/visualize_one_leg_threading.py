#!/usr/bin/env python3
"""Viser debug viewer for FurnitureBench one-leg thread alignment.

Renders only:
  * the local SDF hole-detail patch around the one-leg hole
  * the one-leg inserter geometry split into thread and body meshes

The leg starts at the registered assembled/final pose. GUI sliders apply
translation and world/receptive-frame roll, pitch, yaw offsets relative to
that assembled pose.
"""

from __future__ import annotations

import argparse
import colorsys
import socket
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as R

from peg_in_hole_dynamic import PROBLEM_REGISTRY


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_FB = REPO_ROOT / "assets" / "urdf" / "furniture_bench"
PROBLEM_NAME = "furniture_bench.one_leg_sdf_hybrid_dense"

THREAD_PITCH_M = 0.00937368684342171
YAW_DEG_PER_MM = 360.0 / (THREAD_PITCH_M * 1000.0)
MAX_PHASE_POINTS = 4500


def _phase_colors(theta: np.ndarray) -> np.ndarray:
    """Map angular phase in radians to HSV colors."""
    hue = ((theta % (2.0 * np.pi)) / (2.0 * np.pi)).reshape(-1)
    colors = np.empty((len(hue), 3), dtype=np.uint8)
    for i, h in enumerate(hue):
        r, g, b = colorsys.hsv_to_rgb(float(h), 0.95, 1.0)
        colors[i] = (int(255 * r), int(255 * g), int(255 * b))
    return colors


def _limit_points(points: np.ndarray, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    if len(points) <= MAX_PHASE_POINTS:
        return (points, *arrays)
    idx = np.linspace(0, len(points) - 1, MAX_PHASE_POINTS).astype(np.int64)
    return (points[idx], *(arr[idx] for arr in arrays))


def _free_port(start: int = 8045, stop: int = 8099) -> int | None:
    for port in range(start, stop + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    return None


def _xyzw_to_wxyz(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    qx, qy, qz, qw = q
    return (float(qw), float(qx), float(qy), float(qz))


class ThreadingDebugViewer:
    def __init__(self, server: viser.ViserServer):
        if PROBLEM_NAME not in PROBLEM_REGISTRY:
            raise KeyError(f"{PROBLEM_NAME!r} is not registered")

        self.server = server
        self.problem = PROBLEM_REGISTRY[PROBLEM_NAME]
        self.assets_dir = ASSETS_FB / "square_table"
        final_pose = self.problem.final_insert_pose_rel_receptive
        self.final_pos = np.asarray(final_pose[:3], dtype=float)
        self.final_rot = R.from_quat(final_pose[3:7])
        self.hole_xy = self.final_pos[:2].copy()
        self.top_z = float(self.problem.hole_z_offset)

        self._leg_frame = None
        self._hole_handle = None
        self._thread_handle = None
        self._body_handle = None
        self._diagnostic_handles = []
        self._info = None
        self._sliders = {}
        self._display_sliders = {}
        self._diagnostic_controls = {}
        self._screw_backoff_slider = None

        self._load_meshes()
        self._precompute_diagnostic_points()
        self._build_scene()
        self._build_gui()
        self._apply_pose()

    def _load_meshes(self) -> None:
        patch_path = (
            self.assets_dir / "hole_patches" / "sdf_hybrid"
            / "one_leg_hole_detail.obj"
        )
        leg_dir = self.assets_dir / "square_table_leg4" / "sdf_hybrid"
        thread_path = leg_dir / "square_table_leg4_thread.obj"
        body_path = leg_dir / "square_table_leg4_body.obj"

        self.patch_mesh = trimesh.load(str(patch_path), force="mesh", process=False)
        self.thread_mesh = trimesh.load(str(thread_path), force="mesh", process=False)
        self.body_mesh = trimesh.load(str(body_path), force="mesh", process=False)

    def _precompute_diagnostic_points(self) -> None:
        patch_v = np.asarray(self.patch_mesh.vertices, dtype=np.float64)
        patch_delta = patch_v[:, :2] - self.hole_xy[None, :]
        patch_radius = np.linalg.norm(patch_delta, axis=1)
        patch_theta = np.arctan2(patch_delta[:, 1], patch_delta[:, 0])
        patch_mask = (
            (patch_radius >= np.percentile(patch_radius, 72))
            & (patch_v[:, 2] >= self.patch_mesh.bounds[0, 2] + 0.0002)
            & (patch_v[:, 2] <= self.patch_mesh.bounds[1, 2] - 0.0002)
        )
        self.patch_phase_points, self.patch_phase_theta = _limit_points(
            patch_v[patch_mask].astype(np.float32),
            patch_theta[patch_mask].astype(np.float32),
        )
        self.patch_phase_colors = _phase_colors(self.patch_phase_theta)

        thread_v = np.asarray(self.thread_mesh.vertices, dtype=np.float64)
        thread_radius = np.linalg.norm(thread_v[:, 1:3], axis=1)
        thread_mask = thread_radius >= np.percentile(thread_radius, 82)
        thread_points = thread_v[thread_mask]
        # Color the leg by its phase at the assembled/final pose, so matching
        # colors should touch matching colors when the slider pose is aligned.
        thread_final = self.final_rot.apply(thread_points) + self.final_pos
        thread_delta = thread_final[:, :2] - self.hole_xy[None, :]
        thread_theta = np.arctan2(thread_delta[:, 1], thread_delta[:, 0])
        self.thread_phase_points, self.thread_phase_theta = _limit_points(
            thread_points.astype(np.float32),
            thread_theta.astype(np.float32),
        )
        self.thread_phase_colors = _phase_colors(self.thread_phase_theta)

    def _build_scene(self) -> None:
        self.server.scene.add_grid("/grid", width=0.16, height=0.16, cell_size=0.01)
        self.server.scene.add_frame(
            "/receptive",
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            show_axes=True,
            axes_length=0.035,
            axes_radius=0.0008,
        )
        self.server.scene.add_frame(
            "/hole_center_top",
            position=(float(self.hole_xy[0]), float(self.hole_xy[1]), self.top_z),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            show_axes=True,
            axes_length=0.025,
            axes_radius=0.0007,
        )
        self._hole_handle = self.server.scene.add_mesh_simple(
            "/hole/detail_patch",
            vertices=np.asarray(self.patch_mesh.vertices, dtype=np.float32),
            faces=np.asarray(self.patch_mesh.faces, dtype=np.int32),
            color=(35, 135, 210),
            opacity=0.55,
        )

        self._leg_frame = self.server.scene.add_frame(
            "/leg",
            position=tuple(float(v) for v in self.final_pos),
            wxyz=_xyzw_to_wxyz(tuple(self.final_rot.as_quat())),
            show_axes=True,
            axes_length=0.035,
            axes_radius=0.0009,
        )
        self._thread_handle = self.server.scene.add_mesh_simple(
            "/leg/thread",
            vertices=np.asarray(self.thread_mesh.vertices, dtype=np.float32),
            faces=np.asarray(self.thread_mesh.faces, dtype=np.int32),
            color=(245, 145, 25),
            opacity=0.95,
        )
        self._body_handle = self.server.scene.add_mesh_simple(
            "/leg/body",
            vertices=np.asarray(self.body_mesh.vertices, dtype=np.float32),
            faces=np.asarray(self.body_mesh.faces, dtype=np.int32),
            color=(125, 125, 125),
            opacity=0.35,
        )

    def _build_gui(self) -> None:
        self.server.gui.add_markdown("## One-Leg Thread Debug")
        self._info = self.server.gui.add_markdown("")

        with self.server.gui.add_folder("Translation Offset", expand_by_default=True):
            self._sliders["dx"] = self.server.gui.add_slider(
                "dx (mm)", min=-30.0, max=30.0, step=0.1, initial_value=0.0,
            )
            self._sliders["dy"] = self.server.gui.add_slider(
                "dy (mm)", min=-30.0, max=30.0, step=0.1, initial_value=0.0,
            )
            self._sliders["dz"] = self.server.gui.add_slider(
                "dz (mm)", min=-40.0, max=60.0, step=0.1, initial_value=0.0,
            )

        with self.server.gui.add_folder("Rotation Offset", expand_by_default=True):
            self._sliders["roll"] = self.server.gui.add_slider(
                "roll X (deg)", min=-180.0, max=180.0, step=0.25, initial_value=0.0,
            )
            self._sliders["pitch"] = self.server.gui.add_slider(
                "pitch Y (deg)", min=-180.0, max=180.0, step=0.25, initial_value=0.0,
            )
            self._sliders["yaw"] = self.server.gui.add_slider(
                "yaw Z (deg)", min=-1080.0, max=1080.0, step=0.25, initial_value=0.0,
            )

        with self.server.gui.add_folder("Display", expand_by_default=True):
            self._display_sliders["hole"] = self.server.gui.add_slider(
                "hole opacity", min=0.0, max=1.0, step=0.02, initial_value=0.55,
            )
            self._display_sliders["thread"] = self.server.gui.add_slider(
                "thread opacity", min=0.0, max=1.0, step=0.02, initial_value=0.95,
            )
            self._display_sliders["body"] = self.server.gui.add_slider(
                "body opacity", min=0.0, max=1.0, step=0.02, initial_value=0.35,
            )
            self._diagnostic_controls["phase3d"] = self.server.gui.add_checkbox(
                "phase-colored 3D points", initial_value=True,
            )
            self._diagnostic_controls["unwrap"] = self.server.gui.add_checkbox(
                "cylindrical unwrap", initial_value=True,
            )
            self._diagnostic_controls["point_size"] = self.server.gui.add_slider(
                "diagnostic point size", min=0.0004, max=0.0030, step=0.0001,
                initial_value=0.0012,
            )

        with self.server.gui.add_folder("Threaded Path", expand_by_default=True):
            self._screw_backoff_slider = self.server.gui.add_slider(
                "screw backoff (mm)",
                min=0.0,
                max=1000.0 * float(self.problem.pre_insert_offset),
                step=0.1,
                initial_value=0.0,
            )
            self.server.gui.add_markdown(
                f"`yaw = {YAW_DEG_PER_MM:.3f} deg/mm * backoff_mm`"
            )

        with self.server.gui.add_folder("Presets", expand_by_default=True):
            reset_btn = self.server.gui.add_button("Reset assembled")
            current_pre_btn = self.server.gui.add_button("Current pre-insert")
            screw_pre_btn = self.server.gui.add_button("Screw pre-insert")

        for slider in self._sliders.values():
            slider.on_update(lambda _: self._apply_pose())
        for slider in self._display_sliders.values():
            slider.on_update(lambda _: self._apply_opacity())
        for control in self._diagnostic_controls.values():
            control.on_update(lambda _: self._render_diagnostics())
        self._screw_backoff_slider.on_update(lambda _: self._apply_screw_backoff())

        @reset_btn.on_click
        def _(_) -> None:
            self._set_offsets(dx=0.0, dy=0.0, dz=0.0, roll=0.0, pitch=0.0, yaw=0.0)
            self._screw_backoff_slider.value = 0.0

        @current_pre_btn.on_click
        def _(_) -> None:
            self._set_offsets(dx=0.0, dy=0.0, dz=25.0, roll=0.0, pitch=0.0, yaw=0.0)

        @screw_pre_btn.on_click
        def _(_) -> None:
            self._screw_backoff_slider.value = 1000.0 * float(self.problem.pre_insert_offset)
            self._apply_screw_backoff()

    def _apply_screw_backoff(self) -> None:
        backoff_mm = float(self._screw_backoff_slider.value)
        self._sliders["dz"].value = backoff_mm
        self._sliders["yaw"].value = backoff_mm * YAW_DEG_PER_MM
        self._apply_pose()

    def _apply_opacity(self) -> None:
        if self._hole_handle is not None:
            self._hole_handle.opacity = float(self._display_sliders["hole"].value)
        if self._thread_handle is not None:
            self._thread_handle.opacity = float(self._display_sliders["thread"].value)
        if self._body_handle is not None:
            self._body_handle.opacity = float(self._display_sliders["body"].value)

    def _current_pose(self) -> Tuple[np.ndarray, R, np.ndarray, np.ndarray]:
        trans_mm, rpy_deg = self._slider_values()
        pos = self.final_pos + trans_mm / 1000.0
        rot = R.from_euler("xyz", rpy_deg, degrees=True) * self.final_rot
        return pos, rot, trans_mm, rpy_deg

    def _clear_diagnostics(self) -> None:
        for handle in reversed(self._diagnostic_handles):
            try:
                handle.remove()
            except Exception:
                pass
        self._diagnostic_handles.clear()

    def _track_diag(self, handle):
        self._diagnostic_handles.append(handle)
        return handle

    def _render_diagnostics(self) -> None:
        if not self._diagnostic_controls:
            return
        self._clear_diagnostics()

        point_size = float(self._diagnostic_controls["point_size"].value)
        show_phase3d = bool(self._diagnostic_controls["phase3d"].value)
        show_unwrap = bool(self._diagnostic_controls["unwrap"].value)

        pos, rot, _, _ = self._current_pose()
        thread_world = rot.apply(self.thread_phase_points) + pos
        thread_delta = thread_world[:, :2] - self.hole_xy[None, :]
        thread_theta = (np.arctan2(thread_delta[:, 1], thread_delta[:, 0])
                        + 2.0 * np.pi) % (2.0 * np.pi)

        if show_phase3d:
            self._track_diag(self.server.scene.add_point_cloud(
                "/hole/phase_points",
                points=self.patch_phase_points,
                colors=self.patch_phase_colors,
                point_size=point_size,
                point_shape="circle",
            ))
            self._track_diag(self.server.scene.add_point_cloud(
                "/leg/phase_points",
                points=self.thread_phase_points,
                colors=self.thread_phase_colors,
                point_size=point_size,
                point_shape="circle",
            ))

        if not show_unwrap:
            return

        origin = np.array([0.030, -0.115, -0.010], dtype=np.float64)
        width = 0.095
        height = 0.050
        z_min_mm = -12.0
        z_max_mm = 65.0

        def unwrap_points(z_m: np.ndarray, theta: np.ndarray) -> np.ndarray:
            z_mm = z_m * 1000.0
            x = np.clip((z_mm - z_min_mm) / (z_max_mm - z_min_mm), 0.0, 1.0) * width
            y = (theta % (2.0 * np.pi)) / (2.0 * np.pi) * height
            return np.column_stack([
                origin[0] + x,
                origin[1] + y,
                np.full_like(x, origin[2]),
            ]).astype(np.float32)

        hole_theta = (self.patch_phase_theta + 2.0 * np.pi) % (2.0 * np.pi)
        hole_unwrap = unwrap_points(self.patch_phase_points[:, 2], hole_theta)
        thread_unwrap = unwrap_points(thread_world[:, 2], thread_theta)

        self._track_diag(self.server.scene.add_point_cloud(
            "/unwrap/hole_points",
            points=hole_unwrap,
            colors=np.array([50, 150, 255], dtype=np.uint8),
            point_size=point_size * 1.35,
            point_shape="circle",
        ))
        self._track_diag(self.server.scene.add_point_cloud(
            "/unwrap/thread_points",
            points=thread_unwrap,
            colors=np.array([255, 150, 25], dtype=np.uint8),
            point_size=point_size * 1.15,
            point_shape="circle",
        ))

        corners = np.array([
            [origin[0], origin[1], origin[2]],
            [origin[0] + width, origin[1], origin[2]],
            [origin[0] + width, origin[1] + height, origin[2]],
            [origin[0], origin[1] + height, origin[2]],
        ], dtype=np.float32)
        border = np.array([
            [corners[0], corners[1]],
            [corners[1], corners[2]],
            [corners[2], corners[3]],
            [corners[3], corners[0]],
        ], dtype=np.float32)
        self._track_diag(self.server.scene.add_line_segments(
            "/unwrap/border",
            points=border,
            colors=np.array([230, 230, 230], dtype=np.uint8),
            line_width=2.0,
        ))

        grid_lines = []
        for frac in (0.25, 0.50, 0.75):
            y = origin[1] + frac * height
            grid_lines.append([
                [origin[0], y, origin[2]],
                [origin[0] + width, y, origin[2]],
            ])
        self._track_diag(self.server.scene.add_line_segments(
            "/unwrap/theta_grid",
            points=np.asarray(grid_lines, dtype=np.float32),
            colors=np.array([95, 95, 95], dtype=np.uint8),
            line_width=1.0,
        ))
        self._track_diag(self.server.scene.add_label(
            "/unwrap/title",
            text="unwrap: x=z depth, y=phase",
            position=tuple(origin + np.array([0.0, height + 0.006, 0.0])),
            font_size_mode="screen",
            font_screen_scale=0.7,
        ))
        self._track_diag(self.server.scene.add_label(
            "/unwrap/legend_hole",
            text="blue: hole",
            position=tuple(origin + np.array([width + 0.004, height * 0.72, 0.0])),
            font_size_mode="screen",
            font_screen_scale=0.65,
        ))
        self._track_diag(self.server.scene.add_label(
            "/unwrap/legend_thread",
            text="orange: leg thread",
            position=tuple(origin + np.array([width + 0.004, height * 0.50, 0.0])),
            font_size_mode="screen",
            font_screen_scale=0.65,
        ))

    def _set_offsets(
        self, *, dx: float, dy: float, dz: float,
        roll: float, pitch: float, yaw: float,
    ) -> None:
        self._sliders["dx"].value = dx
        self._sliders["dy"].value = dy
        self._sliders["dz"].value = dz
        self._sliders["roll"].value = roll
        self._sliders["pitch"].value = pitch
        self._sliders["yaw"].value = yaw
        self._apply_pose()

    def _slider_values(self) -> Tuple[np.ndarray, np.ndarray]:
        trans_mm = np.array([
            float(self._sliders["dx"].value),
            float(self._sliders["dy"].value),
            float(self._sliders["dz"].value),
        ])
        rpy_deg = np.array([
            float(self._sliders["roll"].value),
            float(self._sliders["pitch"].value),
            float(self._sliders["yaw"].value),
        ])
        return trans_mm, rpy_deg

    def _apply_pose(self) -> None:
        pos, rot, trans_mm, rpy_deg = self._current_pose()
        q_xyzw = rot.as_quat()

        self._leg_frame.position = tuple(float(v) for v in pos)
        self._leg_frame.wxyz = _xyzw_to_wxyz(tuple(q_xyzw))
        self._render_diagnostics()

        tip_local = np.array([self.thread_mesh.bounds[0, 0], 0.0, 0.0])
        shoulder_local = np.array([self.thread_mesh.bounds[1, 0], 0.0, 0.0])
        tip = rot.apply(tip_local) + pos
        shoulder = rot.apply(shoulder_local) + pos

        if self._info is not None:
            self._info.content = (
                f"**Problem:** `{PROBLEM_NAME}`\n"
                f"- final xyz: `({self.final_pos[0]:+.5f}, {self.final_pos[1]:+.5f}, {self.final_pos[2]:+.5f})`\n"
                f"- current xyz: `({pos[0]:+.5f}, {pos[1]:+.5f}, {pos[2]:+.5f})`\n"
                f"- offset mm: `({trans_mm[0]:+.1f}, {trans_mm[1]:+.1f}, {trans_mm[2]:+.1f})`\n"
                f"- offset rpy deg: `({rpy_deg[0]:+.2f}, {rpy_deg[1]:+.2f}, {rpy_deg[2]:+.2f})`\n"
                f"- predicted thread ratio: `{YAW_DEG_PER_MM:.3f} deg yaw / mm z-backoff`\n"
                f"- screw residual: `{rpy_deg[2] - trans_mm[2] * YAW_DEG_PER_MM:+.2f} deg`\n"
                f"- quat xyzw: `({q_xyzw[0]:+.5f}, {q_xyzw[1]:+.5f}, {q_xyzw[2]:+.5f}, {q_xyzw[3]:+.5f})`\n"
                f"- tip z / shoulder z: `{tip[2]:+.5f}` / `{shoulder[2]:+.5f}`\n"
                f"- tabletop top z: `{self.top_z:+.5f}`\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    port = args.port if args.port is not None else _free_port()
    if port is None:
        raise SystemExit("Could not find a free port in [8045, 8099]")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    print(f"[thread-debug] viser server listening on http://localhost:{port}")
    print(f"[thread-debug] problem: {PROBLEM_NAME}")

    @server.on_client_connect
    def _on_connect(client: viser.ClientHandle) -> None:
        client.camera.position = (0.03, -0.10, 0.12)
        client.camera.look_at = (-0.056, 0.057, 0.015)

    ThreadingDebugViewer(server)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
