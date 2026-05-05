#!/usr/bin/env python3
"""Standalone viser viewer for `peg_in_hole_dynamic.PROBLEM_REGISTRY`.

Loads the receptive URDF at the origin and the insertion URDF at either
the final insert pose or the pre-insert pose (selectable via dropdown),
with optional keypoint markers for the insertion object's bounding-box
corners. No Isaac Gym, no policy.

Usage:
    python peg_in_hole_dynamic/visualize_problems.py [--port 8043]
"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf

from dextoolbench.objects import NAME_TO_OBJECT
from peg_in_hole_dynamic import PROBLEM_REGISTRY, Problem

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_ROOT = REPO_ROOT / "assets"

# Defaults from SimToolReal.yaml — used to reproduce env-side keypoint
# math for visualization purposes.
OBJECT_BASE_SIZE = 0.04
KEYPOINT_SCALE = 1.5

# Env defaults (PegInHoleDynamicEnv / SimToolReal): table base at world z=0.38,
# table half-height 0.15 → top at world z=0.53. Robot base at world (0, 0.8, 0).
TABLE_RESET_Z = 0.38
TABLE_HALF_HEIGHT = 0.15
TABLE_TOP_Z = TABLE_RESET_Z + TABLE_HALF_HEIGHT
TABLE_DIMENSIONS = (0.475, 0.4, 0.3)  # X, Y, Z extents from urdf/table_narrow.urdf
ROBOT_POSITION = (0.0, 0.8, 0.0)
ROBOT_URDF = REPO_ROOT / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
DEFAULT_ARM_DOFS = np.array([-1.571, 1.571 - np.deg2rad(10), 0.0,
                             1.376 + np.deg2rad(10), 0.0, 1.485, 1.308])


def _xyzw_to_wxyz(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    qx, qy, qz, qw = q
    return (qw, qx, qy, qz)


def _keypoint_half_extents(insertion_object_name: str) -> Optional[Tuple[float, float, float]]:
    obj = NAME_TO_OBJECT.get(insertion_object_name)
    if obj is None:
        return None
    return tuple(
        obj.scale[i] * OBJECT_BASE_SIZE * KEYPOINT_SCALE / 2 for i in range(3)
    )


def _keypoint_corners(half_extents: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
    hx, hy, hz = half_extents
    out = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                out.append((sx * hx, sy * hy, sz * hz))
    return out


def _parse_fixture_urdf(urdf_path: Path) -> List[Tuple[str, Tuple[float, float, float], Tuple[float, float, float, float], List[Tuple[Path, Tuple[int, int, int]]]]]:
    """Parse one of our generated fixture URDFs.

    Returns a list of ``(part_label, joint_xyz, joint_wxyz, [(mesh_abs, rgb), ...])``
    entries — one per non-root link. Each visual mesh gets its own color from
    a palette (each CoACD hull rendered distinctly), with hue rotated per
    link so different parts use different segments of the palette.
    """
    tree = ET.parse(str(urdf_path))
    robot = tree.getroot()
    urdf_dir = urdf_path.parent

    joints = {}
    for j in robot.findall("joint"):
        if j.find("child") is None or j.find("origin") is None:
            continue
        child = j.find("child").get("link")
        origin = j.find("origin")
        xyz = tuple(float(x) for x in origin.get("xyz", "0 0 0").split())
        rpy = tuple(float(x) for x in origin.get("rpy", "0 0 0").split())
        joints[child] = (xyz, rpy)

    # 16-color palette — distinct enough that adjacent CoACD hulls don't
    # blend visually. Each link starts at a different palette offset so two
    # parts in the same fixture don't reuse the same opening colors.
    palette = [
        (217,  76,  76), ( 76, 165, 217), (102, 191,  89), (229, 178,  51),
        (179, 102, 204), ( 76, 204, 191), (217, 127,  51), (140, 140, 140),
        (242, 110, 165), ( 51, 130,  76), (255, 217, 102), ( 89,  89, 191),
        (191,  89, 165), (140, 217, 217), (217, 191,  89), (114, 114, 178),
    ]

    parts = []
    palette_offset = 0
    for link in robot.findall("link"):
        link_name = link.get("name")
        if link_name == "root":
            continue

        jxyz, jrpy = joints.get(link_name, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        rxyzw = R.from_euler("xyz", jrpy).as_quat()
        wxyz = (float(rxyzw[3]), float(rxyzw[0]), float(rxyzw[1]), float(rxyzw[2]))

        meshes: List[Tuple[Path, Tuple[int, int, int]]] = []
        for vi, visual in enumerate(link.findall("visual")):
            mesh_tag = visual.find("geometry/mesh")
            if mesh_tag is None:
                continue
            mesh_abs = (urdf_dir / mesh_tag.get("filename")).resolve()
            color = palette[(palette_offset + vi) % len(palette)]
            meshes.append((mesh_abs, color))

        if not meshes:
            continue
        parts.append((link_name, tuple(map(float, jxyz)), wxyz, meshes))
        # Shift the palette so the next part's hulls don't reuse the same
        # leading colors as this part's hulls.
        palette_offset += max(len(meshes), 1) + 3
    return parts


class ProblemVisualizer:
    """Re-builds the dynamic scene under a fresh `/v{counter}` namespace each
    render, so that even if individual `.remove()` calls don't fully clean
    up viser-side state, the next render's nodes never collide with the
    previous render's."""

    POSES = ("final", "pre_insert")

    def __init__(self, server: viser.ViserServer):
        self.server = server
        self._handles: List = []
        self._fixture_mesh_handles: List = []
        self._insertion_mesh_handles: List = []
        self._render_counter = 0
        self._info_md = None
        # UI state — set after gui is built.
        self._current_problem: Optional[str] = None
        self._current_pose: str = "final"
        self._show_insertion_keypoints: bool = False
        self._show_receptive_keypoints: bool = False
        self._fixture_opacity: float = 0.5
        self._insertion_opacity: float = 1.0
        self._build_static()

    # ---- static (built once) ----
    def _build_static(self) -> None:
        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
        self.server.scene.add_frame(
            "/world", position=(0, 0, 0), wxyz=(1, 0, 0, 0), show_axes=True,
            axes_length=0.1, axes_radius=0.003,
        )

        # Table — modelled as a primitive box matching urdf/table_narrow.urdf,
        # placed so its top sits at world z = TABLE_TOP_Z.
        self.server.scene.add_frame(
            "/table", position=(0, 0, TABLE_RESET_Z), wxyz=(1, 0, 0, 0), show_axes=False,
        )
        self.server.scene.add_box(
            "/table/wood",
            color=(180, 130, 70),
            dimensions=TABLE_DIMENSIONS,
            position=(0, 0, 0),
            opacity=0.85,
        )

        # Robot — pose-only ViserUrdf at the env's robot location.
        self.server.scene.add_frame(
            "/robot", position=ROBOT_POSITION, wxyz=(1, 0, 0, 0), show_axes=False,
        )
        try:
            robot = ViserUrdf(self.server, ROBOT_URDF, root_node_name="/robot")
            cfg = np.zeros(len(robot.get_actuated_joint_names()))
            cfg[: len(DEFAULT_ARM_DOFS)] = DEFAULT_ARM_DOFS
            robot.update_cfg(cfg)
        except Exception as e:
            print(f"[visualize] failed to load robot URDF {ROBOT_URDF}: {e}")

    # ---- handle bookkeeping ----
    def _track(self, handle):
        self._handles.append(handle)
        return handle

    def _clear_dynamic(self) -> None:
        # Remove deepest-first so parents don't auto-orphan children.
        for h in reversed(self._handles):
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        self._fixture_mesh_handles.clear()
        self._insertion_mesh_handles.clear()
        # Belt-and-suspenders: also remove every prior version subtree by
        # path, in case some scene nodes aren't tracked by .remove() handles
        # (ViserUrdf-internal frames, etc.).
        for n in range(1, self._render_counter + 1):
            try:
                self.server.scene.remove_by_name(f"/v{n}")
            except Exception:
                pass

    # ---- pose helpers ----
    @staticmethod
    def _pre_insert_position(p: Problem) -> Tuple[float, float, float]:
        x, y, z = p.insert_pose_rel_receptive[:3]
        idir = np.array(p.insertion_direction, dtype=np.float64)
        idir = idir / (np.linalg.norm(idir) + 1e-12)
        return tuple(np.array((x, y, z)) - idir * p.pre_insert_offset)

    def _selected_position_quat(self, p: Problem) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        x, y, z, qx, qy, qz, qw = p.insert_pose_rel_receptive
        if self._current_pose == "pre_insert":
            x, y, z = self._pre_insert_position(p)
        return (x, y, z), (qx, qy, qz, qw)

    # ---- main render ----
    def render(self) -> None:
        self._clear_dynamic()
        self._render_counter += 1
        ns = f"/v{self._render_counter}"

        if self._current_problem not in PROBLEM_REGISTRY:
            return
        p = PROBLEM_REGISTRY[self._current_problem]

        # --- receptive object at world (0, 0, table_top_z + hole_z_offset) ---
        # This matches what the env will do: hole_pos = (rand_x, rand_y,
        # table_top_z + holeZOffset). For visualization we fix XY at 0 and
        # use the per-problem hole_z_offset.
        recv_world_z = TABLE_TOP_Z + p.hole_z_offset
        receptive_abs = ASSETS_ROOT / p.receptive_urdf
        recv_frame = self._track(self.server.scene.add_frame(
            f"{ns}/receptive",
            position=(0, 0, recv_world_z),
            wxyz=(1, 0, 0, 0),
            show_axes=True, axes_length=0.1, axes_radius=0.003,
        ))
        try:
            for part_label, jxyz, jwxyz, mesh_entries in _parse_fixture_urdf(receptive_abs):
                part_node = f"{ns}/receptive/{part_label}"
                self._track(self.server.scene.add_frame(
                    part_node, position=jxyz, wxyz=jwxyz, show_axes=False,
                ))
                for mi, (mp, rgb) in enumerate(mesh_entries):
                    try:
                        mesh = trimesh.load_mesh(str(mp), process=False)
                    except Exception as e:
                        print(f"[visualize] failed to load mesh {mp}: {e}")
                        continue
                    h = self.server.scene.add_mesh_simple(
                        f"{part_node}/m{mi}",
                        vertices=np.asarray(mesh.vertices, dtype=np.float32),
                        faces=np.asarray(mesh.faces, dtype=np.int32),
                        color=rgb,
                        opacity=self._fixture_opacity,
                    )
                    self._track(h)
                    self._fixture_mesh_handles.append(h)
        except Exception as e:
            print(f"[visualize] failed to load receptive {receptive_abs}: {e}")

        # --- insertion object at the selected pose (relative to receptive) ---
        (ix, iy, iz), (qx, qy, qz, qw) = self._selected_position_quat(p)
        # Lift to world frame by adding the receptive's world z.
        ix_world, iy_world, iz_world = ix, iy, iz + recv_world_z
        ins_frame = self._track(self.server.scene.add_frame(
            f"{ns}/insertion",
            position=(ix_world, iy_world, iz_world),
            wxyz=_xyzw_to_wxyz((qx, qy, qz, qw)),
            show_axes=True, axes_length=0.05, axes_radius=0.002,
        ))
        try:
            obj = NAME_TO_OBJECT.get(p.insertion_object_name)
            if obj is None:
                raise KeyError(f"{p.insertion_object_name!r} not in NAME_TO_OBJECT")
            for part_label, jxyz, jwxyz, mesh_entries in _parse_fixture_urdf(Path(obj.urdf_path)):
                ins_part_node = f"{ns}/insertion/{part_label}"
                self._track(self.server.scene.add_frame(
                    ins_part_node, position=jxyz, wxyz=jwxyz, show_axes=False,
                ))
                for mi, (mp, rgb) in enumerate(mesh_entries):
                    try:
                        mesh = trimesh.load_mesh(str(mp), process=False)
                    except Exception as e:
                        print(f"[visualize] failed to load mesh {mp}: {e}")
                        continue
                    h = self.server.scene.add_mesh_simple(
                        f"{ins_part_node}/m{mi}",
                        vertices=np.asarray(mesh.vertices, dtype=np.float32),
                        faces=np.asarray(mesh.faces, dtype=np.int32),
                        color=rgb,
                        opacity=self._insertion_opacity,
                    )
                    self._track(h)
                    self._insertion_mesh_handles.append(h)
        except Exception as e:
            print(f"[visualize] failed to load insertion: {e}")

        # --- "other" pose marker (small frame at the non-selected pose) ---
        if self._current_pose == "final":
            other_pos = self._pre_insert_position(p)
            other_label = "pre_insert"
        else:
            other_pos = p.insert_pose_rel_receptive[:3]
            other_label = "final"
        other_world = (other_pos[0], other_pos[1], other_pos[2] + recv_world_z)
        self._track(self.server.scene.add_frame(
            f"{ns}/other_{other_label}",
            position=other_world,
            wxyz=_xyzw_to_wxyz((qx, qy, qz, qw)),
            show_axes=True, axes_length=0.04, axes_radius=0.0015,
        ))

        # --- optional keypoint spheres ---
        half_ext = _keypoint_half_extents(p.insertion_object_name)

        # "Insertion keypoints": the 8 bbox corners attached to the insertion
        # frame so they ride along with the rendered insertion mesh.
        if self._show_insertion_keypoints and half_ext is not None:
            for k, (cx, cy, cz) in enumerate(_keypoint_corners(half_ext)):
                self._track(self.server.scene.add_icosphere(
                    f"{ns}/insertion/kp_{k}",
                    position=(cx, cy, cz),
                    radius=0.006,
                    color=(40, 200, 255),
                ))

        # "Goal-pose keypoints": same 8 corners but expressed in the
        # receptive frame at the FINAL insert pose — this is what the
        # policy is trying to land each corner on.
        if self._show_receptive_keypoints and half_ext is not None:
            # Keypoints at the FINAL insert pose — this is where the policy
            # is trying to land each insertion-object corner.
            from scipy.spatial.transform import Rotation as _R
            x_f, y_f, z_f, qx_f, qy_f, qz_f, qw_f = p.insert_pose_rel_receptive
            R_ins = _R.from_quat([qx_f, qy_f, qz_f, qw_f]).as_matrix()
            for k, c in enumerate(_keypoint_corners(half_ext)):
                rec_local = R_ins @ np.asarray(c) + np.array([x_f, y_f, z_f])
                world = (rec_local[0], rec_local[1], rec_local[2] + recv_world_z)
                self._track(self.server.scene.add_icosphere(
                    f"{ns}/receptive_kp_{k}",
                    position=world,
                    radius=0.006,
                    color=(255, 180, 40),
                ))

        # --- info panel ---
        if self._info_md is not None:
            x_f, y_f, z_f, qx_f, qy_f, qz_f, qw_f = p.insert_pose_rel_receptive
            pre = self._pre_insert_position(p)
            self._info_md.content = (
                f"### {self._current_problem}  ({self._current_pose})\n"
                f"- **insertion**: `{p.insertion_object_name}`\n"
                f"- **receptive**: `{p.receptive_urdf}`\n"
                f"- **final** xyz `({x_f:+.4f}, {y_f:+.4f}, {z_f:+.4f})`\n"
                f"- **pre_insert** xyz `({pre[0]:+.4f}, {pre[1]:+.4f}, {pre[2]:+.4f})`\n"
                f"- **quat (xyzw)** `({qx_f:+.4f}, {qy_f:+.4f}, {qz_f:+.4f}, {qw_f:+.4f})`\n"
                f"- **insertion_direction**: `{tuple(p.insertion_direction)}`\n"
                f"- **pre_insert_offset**: `{p.pre_insert_offset}`\n"
            )

    # ---- UI integration ----
    def attach_info(self, info_handle) -> None:
        self._info_md = info_handle

    def set_problem(self, name: str) -> None:
        self._current_problem = name
        self.render()

    def set_pose(self, pose: str) -> None:
        if pose not in self.POSES:
            return
        self._current_pose = pose
        self.render()

    def set_show_insertion_keypoints(self, val: bool) -> None:
        self._show_insertion_keypoints = bool(val)
        self.render()

    def set_show_receptive_keypoints(self, val: bool) -> None:
        self._show_receptive_keypoints = bool(val)
        self.render()

    def set_fixture_opacity(self, val: float) -> None:
        self._fixture_opacity = float(np.clip(val, 0.0, 1.0))
        for h in self._fixture_mesh_handles:
            try:
                h.opacity = self._fixture_opacity
            except Exception:
                pass

    def set_insertion_opacity(self, val: float) -> None:
        self._insertion_opacity = float(np.clip(val, 0.0, 1.0))
        for h in self._insertion_mesh_handles:
            try:
                h.opacity = self._insertion_opacity
            except Exception:
                pass


def _free_port(start: int = 8043, span: int = 20) -> Optional[int]:
    import socket
    for p in range(start, start + span):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("0.0.0.0", p))
            s.close()
            return p
        except OSError:
            s.close()
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=None,
                        help="viser port; if unset, picks a free port near 8043")
    parser.add_argument("--initial", type=str, default=None,
                        help="initial problem name; defaults to first in registry")
    args = parser.parse_args()

    problems = sorted(PROBLEM_REGISTRY)
    if not problems:
        raise SystemExit("PROBLEM_REGISTRY is empty — nothing to visualize.")
    initial = args.initial if args.initial in problems else problems[0]

    port = args.port if args.port is not None else _free_port()
    if port is None:
        raise SystemExit("Could not find a free port near 8043.")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    print(f"[visualize] viser server listening on http://localhost:{port}")
    print(f"[visualize] {len(problems)} problem(s) registered")

    @server.on_client_connect
    def _on_connect(client: viser.ClientHandle) -> None:
        client.camera.position = (0.6, 0.6, 0.4)
        client.camera.look_at = (0.0, 0.0, 0.05)

    viz = ProblemVisualizer(server)

    with server.gui.add_folder("Problem"):
        problem_dropdown = server.gui.add_dropdown(
            "Select", options=problems, initial_value=initial,
        )
        pose_dropdown = server.gui.add_dropdown(
            "Pose", options=list(ProblemVisualizer.POSES), initial_value="final",
        )
        kp_ins_checkbox = server.gui.add_checkbox(
            "Show insertion keypoints", initial_value=False,
        )
        kp_recv_checkbox = server.gui.add_checkbox(
            "Show goal-pose keypoints", initial_value=False,
        )
        opacity_slider = server.gui.add_slider(
            "Fixture opacity", min=0.0, max=1.0, step=0.05, initial_value=0.5,
        )
        ins_opacity_slider = server.gui.add_slider(
            "Insertion opacity", min=0.0, max=1.0, step=0.05, initial_value=1.0,
        )
        info = server.gui.add_markdown("")
        viz.attach_info(info)

    @problem_dropdown.on_update
    def _on_problem(_) -> None:
        viz.set_problem(problem_dropdown.value)

    @pose_dropdown.on_update
    def _on_pose(_) -> None:
        viz.set_pose(pose_dropdown.value)

    @kp_ins_checkbox.on_update
    def _on_kp_ins(_) -> None:
        viz.set_show_insertion_keypoints(kp_ins_checkbox.value)

    @kp_recv_checkbox.on_update
    def _on_kp_recv(_) -> None:
        viz.set_show_receptive_keypoints(kp_recv_checkbox.value)

    @opacity_slider.on_update
    def _on_opacity(_) -> None:
        viz.set_fixture_opacity(opacity_slider.value)

    @ins_opacity_slider.on_update
    def _on_ins_opacity(_) -> None:
        viz.set_insertion_opacity(ins_opacity_slider.value)

    viz.set_fixture_opacity(opacity_slider.value)
    viz.set_insertion_opacity(ins_opacity_slider.value)
    viz.set_problem(initial)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
