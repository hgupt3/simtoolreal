"""Render init-state distribution overlays for fig 4 panel (a) — all 4 tasks.

Top-down 3D render of:
    - Kuka Sharpa robot at default arm pose (yourdfpy → trimesh)
    - Table box (0.475 × 0.4 × 0.3 m)
    - N=10 GOAL hole poses (gray) on the table, drawn from the env's
      hole_x_range × hole_y_range (peg_in_hole_env_cfg.py:33-34, plus
      per-problem overrides at peg_in_hole_dynamic_env.py:576-586).
    - N=10 OBJECT start poses (green), lying flat on the table at random
      XY in (±reset_position_noise_x, ±reset_position_noise_y) and random yaw.
      The env actually spawns the object 25 cm above the table at full random
      SO(3) (simtoolreal_env_cfg.py:ResetCfg + reset_utils.py:_reset_object_pose),
      but the policy can settle the object — we approximate the *settled* pose
      as lying-flat-with-random-yaw, which is the dominant resting orientation
      for these elongated parts.

Outputs:
    plot_figures/fig4/inputs/init_dist_peg_in_hole.png
    plot_figures/fig4/inputs/init_dist_asm_pillar.png
    plot_figures/fig4/inputs/init_dist_asm_beam.png
    plot_figures/fig4/inputs/init_dist_screw_leg.png

Layout constants from CLAUDE.md:
    TABLE_Z = 0.38 (table center z; table top ≈ 0.53)
    Robot base at (0, 0.8, 0)
    Fixture at (0.12, -0.152, 0.15) relative to table
"""

import json
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pyrender
import trimesh
import yourdfpy
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets" / "urdf"
OUT = REPO / "plot_figures" / "fig4" / "inputs"

# Scene constants
TABLE_Z = 0.38
TABLE_SIZE = (0.475, 0.4, 0.3)
TABLE_TOP_Z = TABLE_Z + TABLE_SIZE[2] / 2  # 0.53
ROBOT_POS = (0.0, 0.8, 0.0)
FIXTURE_POS_REL_TABLE = (0.12, -0.152, 0.15)
FIXTURE_POS = (
    FIXTURE_POS_REL_TABLE[0],
    FIXTURE_POS_REL_TABLE[1],
    TABLE_TOP_Z + FIXTURE_POS_REL_TABLE[2],
)

# Object spawn noise (simtoolreal_env_cfg.py:ResetCfg defaults)
OBJ_XY_NOISE = (0.1, 0.1)   # ±0.1 m

# Robot default arm pose (mirrors viser_viz.py)
_ARM_DEFAULT = np.array([-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308])
_ARM_DEFAULT[1] -= np.deg2rad(10)
_ARM_DEFAULT[3] += np.deg2rad(10)
URDF_PATH = ASSETS / "kuka_sharpa_description" / "iiwa14_left_sharpa_adjusted_restricted.urdf"

# Colors (RGBA, 0-255)
COLOR_ROBOT = [180, 180, 185, 255]
COLOR_TABLE = [209, 143, 89, 255]
COLOR_GOAL = [120, 120, 130, 130]    # gray, low alpha — many overlays
COLOR_OBJECT = [70, 200, 90, 130]    # green, low alpha — many overlays

N_OVERLAYS = 10
RNG = np.random.default_rng(42)


def _merge_scene(s):
    if isinstance(s, trimesh.Scene):
        return trimesh.util.concatenate(tuple(s.geometry.values()))
    return s


def _colored_mesh(m, color, alpha_mode="OPAQUE", smooth=True):
    base = np.array(color, dtype=np.float32) / 255.0
    return pyrender.Mesh.from_trimesh(
        m,
        material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=base.tolist(),
            metallicFactor=0.1,
            roughnessFactor=0.7,
            doubleSided=True,
            alphaMode=alpha_mode,
        ),
        smooth=smooth,
    )


def _table_mesh() -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=TABLE_SIZE)
    box.apply_translation([0, 0, TABLE_Z])
    return box


def _robot_meshes() -> list[trimesh.Trimesh]:
    urdf = yourdfpy.URDF.load(str(URDF_PATH))
    cfg = {n: 0.0 for n in urdf.actuated_joint_names}
    for i in range(7):
        cfg[urdf.actuated_joint_names[i]] = _ARM_DEFAULT[i]
    urdf.update_cfg(cfg)
    out = []
    for geom_name, geom in urdf.scene.geometry.items():
        T = urdf.scene.graph.get(geom_name)[0]
        m = geom.copy()
        m.apply_transform(T)
        m.apply_translation(np.asarray(ROBOT_POS))
        out.append(m)
    return out


def _lay_flat(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rotate so the mesh's shortest local axis ends up along world +Z."""
    extents = mesh.extents
    shortest = int(np.argmin(extents))
    if shortest == 2:
        return mesh.copy()
    # Rotate so axis `shortest` becomes axis 2 (Z).
    axis_for_rot = {0: [0, 1, 0], 1: [1, 0, 0]}[shortest]
    R = trimesh.transformations.rotation_matrix(np.pi / 2, axis_for_rot)
    m = mesh.copy()
    m.apply_transform(R)
    return m


def _drop_to_table(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Translate so the mesh's z_min sits on the table top (+1 mm buffer)."""
    m = mesh.copy()
    m.apply_translation([0, 0, TABLE_TOP_Z + 0.001 - m.bounds[0, 2]])
    return m


def _yaw_about_centroid(mesh: trimesh.Trimesh, yaw_rad: float) -> trimesh.Trimesh:
    """Rotate mesh by yaw around its own XY centroid (Z axis)."""
    c = mesh.centroid
    R = trimesh.transformations.rotation_matrix(yaw_rad, [0, 0, 1])
    T_to_origin = trimesh.transformations.translation_matrix([-c[0], -c[1], 0])
    T_back = trimesh.transformations.translation_matrix([c[0], c[1], 0])
    m = mesh.copy()
    m.apply_transform(T_back @ R @ T_to_origin)
    return m


def _sample_obj_poses(n):
    xs = RNG.uniform(-OBJ_XY_NOISE[0], OBJ_XY_NOISE[0], n)
    ys = RNG.uniform(-OBJ_XY_NOISE[1], OBJ_XY_NOISE[1], n)
    yaws = RNG.uniform(-np.pi, np.pi, n)
    return list(zip(xs, ys, yaws))


def _sample_goal_xy(n, x_range, y_range):
    xs = RNG.uniform(x_range[0], x_range[1], n)
    ys = RNG.uniform(y_range[0], y_range[1], n)
    return list(zip(xs, ys))


# ---------- Per-task asset loading ---------------------------------------


def _load_peg_in_hole():
    insert = trimesh.load(ASSETS / "peg_in_hole" / "peg" / "peg.obj")
    receptive = trimesh.load(ASSETS / "peg_in_hole" / "holes" / "hole_tol0p5mm" / "hole.obj")
    return insert, receptive


def _load_fabrica(insert_id, receptive_ids):
    insert = _merge_scene(trimesh.load(ASSETS / "fabrica" / "beam_2x" / insert_id / f"{insert_id}.obj"))
    pieces = [
        _merge_scene(trimesh.load(ASSETS / "fabrica" / "beam_2x" / pid / f"{pid}.obj"))
        for pid in receptive_ids
    ]
    receptive = trimesh.util.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    return insert, receptive


def _load_screw_leg():
    sqt = ASSETS / "furniture_bench" / "square_table"
    top = _merge_scene(trimesh.load(sqt / "square_table_top" / "square_table_top.obj",
                                    force="mesh", process=False))
    leg = _merge_scene(trimesh.load(sqt / "square_table_leg4" / "square_table_leg4.obj",
                                    force="mesh", process=False))
    # y-up -> z-up (R_x(+90 deg) per assembly.json)
    R_yz = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    top.apply_transform(R_yz)
    leg.apply_transform(R_yz)
    return leg, top


# ---------- Per-task config -----------------------------------------------


@dataclass
class TaskSpec:
    name: str
    loader: Callable
    goal_x_range: tuple
    goal_y_range: tuple
    receptive_translucent: bool = False


TASKS = [
    TaskSpec(
        name="peg_in_hole",
        loader=_load_peg_in_hole,
        goal_x_range=(-0.1875, 0.1875),
        goal_y_range=(-0.1, 0.1),
    ),
    TaskSpec(
        name="asm_pillar",
        loader=lambda: _load_fabrica("2", ["6"]),
        goal_x_range=(-0.1875, 0.1875),
        goal_y_range=(-0.1, 0.2),
    ),
    TaskSpec(
        name="asm_beam",
        loader=lambda: _load_fabrica("0", ["6", "2"]),
        goal_x_range=(-0.1875, 0.1875),
        goal_y_range=(-0.1, 0.1),
    ),
    TaskSpec(
        name="screw_leg",
        loader=_load_screw_leg,
        goal_x_range=(-0.1875, 0.1875),
        goal_y_range=(-0.1, 0.2),
        receptive_translucent=True,
    ),
]


# ---------- Render orchestration -----------------------------------------


def _topdown_camera():
    eye = np.array([FIXTURE_POS[0], FIXTURE_POS[1], TABLE_TOP_Z + 1.2])
    target = np.array([FIXTURE_POS[0], FIXTURE_POS[1], TABLE_TOP_Z])
    fw = target - eye
    fw /= np.linalg.norm(fw)
    up = np.array([0, 1, 0.0])
    rt = np.cross(fw, up)
    rt /= np.linalg.norm(rt)
    uc = np.cross(rt, fw)
    cp = np.eye(4)
    cp[:3, 0] = rt
    cp[:3, 1] = uc
    cp[:3, 2] = -fw
    cp[:3, 3] = eye
    return cp


def render_task(task: TaskSpec, robot_meshes, table_mesh) -> Image.Image:
    insert_raw, receptive_raw = task.loader()

    # Lay flat (so the object rests on its widest face) and centre at origin
    # so subsequent translations land it at exactly (x, y, table_top).
    insert_flat = _lay_flat(insert_raw)
    insert_centered = insert_flat.copy()
    insert_centered.apply_translation([-insert_centered.centroid[0],
                                       -insert_centered.centroid[1], 0])
    insert_centered = _drop_to_table(insert_centered)

    receptive_flat = _lay_flat(receptive_raw)
    receptive_centered = receptive_flat.copy()
    receptive_centered.apply_translation([-receptive_centered.centroid[0],
                                          -receptive_centered.centroid[1], 0])
    receptive_centered = _drop_to_table(receptive_centered)

    scene = pyrender.Scene(bg_color=[1, 1, 1, 0], ambient_light=[0.6] * 3)
    for m in robot_meshes:
        scene.add(_colored_mesh(m, COLOR_ROBOT))
    scene.add(_colored_mesh(table_mesh, COLOR_TABLE))

    # 10 goal/hole/receptive overlays
    for gx, gy in _sample_goal_xy(N_OVERLAYS, task.goal_x_range, task.goal_y_range):
        m = receptive_centered.copy()
        m.apply_translation([FIXTURE_POS[0] + gx, FIXTURE_POS[1] + gy, 0])
        scene.add(_colored_mesh(m, COLOR_GOAL, alpha_mode="BLEND"))

    # 10 object/insert start overlays (lying flat, random yaw, random XY)
    for ox, oy, yaw in _sample_obj_poses(N_OVERLAYS):
        m = _yaw_about_centroid(insert_centered, yaw)
        m.apply_translation([FIXTURE_POS[0] + ox, FIXTURE_POS[1] + oy, 0])
        scene.add(_colored_mesh(m, COLOR_OBJECT, alpha_mode="BLEND"))

    cam_pose = _topdown_camera()
    scene.add(pyrender.OrthographicCamera(xmag=0.40, ymag=0.40), pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.5), pose=cam_pose)
    side = np.eye(4)
    side[:3, 3] = [FIXTURE_POS[0] + 0.5, FIXTURE_POS[1], TABLE_TOP_Z + 0.5]
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.5), pose=side)

    r = pyrender.OffscreenRenderer(800, 800)
    col, _ = r.render(
        scene,
        flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES,
    )
    r.delete()
    return Image.fromarray(col)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    robot_meshes = _robot_meshes()
    table_mesh = _table_mesh()
    for task in TASKS:
        img = render_task(task, robot_meshes, table_mesh)
        img.save(OUT / f"init_dist_{task.name}.png")
        print(f"wrote init_dist_{task.name}.png")


if __name__ == "__main__":
    main()
