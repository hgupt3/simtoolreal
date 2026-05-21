"""Render fig 2 panel-(a) task-config insets from OBJ assets.

Outputs (PNGs land in plot_figures/fig2/inputs/, since they are consumed as
inputs by make_panel_a.py):
    plot_figures/fig2/inputs/task_inset_peg_in_hole.png
    plot_figures/fig2/inputs/task_inset_asm_pillar.png
    plot_figures/fig2/inputs/task_inset_asm_beam.png
    plot_figures/fig2/inputs/task_inset_screw_leg.png

Convention:
    bright green = part being inserted
    dull gray    = part(s) being inserted into
"""

import json
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from pathlib import Path

import numpy as np
import pyrender
import trimesh
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets" / "urdf"
OUT = REPO / "plot_figures" / "fig2" / "inputs"

GRAY = [150, 150, 155, 255]
GREEN = [70, 200, 90, 255]
GRAY_TRANSLUCENT = [150, 150, 155, 200]  # alpha < 255 so cavity walls show through


def _as_trimesh(loaded) -> trimesh.Trimesh:
    """Coalesce trimesh.load output (Trimesh or Scene) into a single Trimesh."""
    if isinstance(loaded, trimesh.Scene):
        return trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return loaded


def _colored_mesh(m: trimesh.Trimesh, color, alpha_mode="OPAQUE", smooth=False):
    base = np.array(color, dtype=np.float32) / 255.0
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=base.tolist(),
        metallicFactor=0.1,
        roughnessFactor=0.6,
        doubleSided=True,
        alphaMode=alpha_mode,
    )
    return pyrender.Mesh.from_trimesh(m, material=material, smooth=smooth)


def render(meshes_colors, size=(480, 480), azim_deg=35, elev_deg=25, distance_factor=1.7):
    """Bounds-fitted camera; world up = +Z. Used by the 3 Z-up renders."""
    scene = pyrender.Scene(bg_color=[1, 1, 1, 0], ambient_light=[0.55, 0.55, 0.55])
    bounds_all = []
    for m, color in meshes_colors:
        scene.add(_colored_mesh(m, color))
        bounds_all.append(m.bounds)
    bounds = np.stack(bounds_all).reshape(-1, 3)
    center = (bounds.max(0) + bounds.min(0)) / 2
    extent = float((bounds.max(0) - bounds.min(0)).max())
    radius = extent * distance_factor

    az, el = np.deg2rad(azim_deg), np.deg2rad(elev_deg)
    eye = center + radius * np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
    )
    forward = center - eye
    forward /= np.linalg.norm(forward)
    up = np.array([0, 0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up_corr = np.cross(right, forward)
    cam_pose = np.eye(4)
    cam_pose[:3, 0] = right
    cam_pose[:3, 1] = up_corr
    cam_pose[:3, 2] = -forward
    cam_pose[:3, 3] = eye

    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4), pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0), pose=cam_pose)
    top = np.eye(4)
    top[:3, 3] = center + np.array([0, 0, extent * 3])
    top[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.0), pose=top)

    r = pyrender.OffscreenRenderer(*size)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()
    return Image.fromarray(color)


def render_yup(mesh_specs, size=(700, 700), azim_deg=-60, elev_deg=30, ext=0.35,
               fov=np.pi / 4.5, ambient=0.45):
    """Render in Y-up world frame (for furniture_bench assets) with per-mesh alpha.

    mesh_specs: list of (mesh, color_rgba, alpha_mode) tuples.
    """
    scene = pyrender.Scene(bg_color=[1, 1, 1, 0], ambient_light=[ambient] * 3)
    for mesh, color, alpha_mode in mesh_specs:
        scene.add(_colored_mesh(mesh, color, alpha_mode=alpha_mode, smooth=True))

    target = np.array([0, 0, 0.0])
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    eye = target + ext * np.array(
        [np.cos(elev) * np.cos(azim), np.sin(elev), np.cos(elev) * np.sin(azim)]
    )
    up = np.array([0, 1, 0.0])
    fw = target - eye
    fw /= np.linalg.norm(fw)
    rt = np.cross(fw, up)
    rt /= np.linalg.norm(rt)
    uc = np.cross(rt, fw)
    cam_pose = np.eye(4)
    cam_pose[:3, 0] = rt
    cam_pose[:3, 1] = uc
    cam_pose[:3, 2] = -fw
    cam_pose[:3, 3] = eye

    scene.add(pyrender.PerspectiveCamera(yfov=fov), pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.5), pose=cam_pose)
    top_light = np.eye(4)
    top_light[:3, 3] = np.array([0, 0.3, 0])
    top_light[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0), pose=top_light)

    r = pyrender.OffscreenRenderer(*size)
    color, _ = r.render(
        scene,
        flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES,
    )
    r.delete()
    return Image.fromarray(color)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Peg-In-Hole: peg (green) hovering above hole (gray), pointing down.
    # Original peg.obj is a flat board (X long, Z thin). Rotate to vertical,
    # then flip upside-down so the post points into the hole.
    peg = trimesh.load(ASSETS / "peg_in_hole" / "peg" / "peg.obj")
    hole = trimesh.load(ASSETS / "peg_in_hole" / "holes" / "hole_tol0p5mm" / "hole.obj")
    R_y = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])  # X -> Z (vertical)
    R_x = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])      # flip upside down
    R_z = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])      # spin around vertical
    peg.apply_transform(R_z @ R_x @ R_y)
    peg.apply_transform(trimesh.transformations.translation_matrix([0, 0, 0.06 + 0.13]))
    img = render([(hole, GRAY), (peg, GREEN)], distance_factor=1.6, azim_deg=30, elev_deg=20)
    img.save(OUT / "task_inset_peg_in_hole.png")

    # Asm-Pillar: base 6 (gray) + part 2 (green)
    b6 = trimesh.load(ASSETS / "fabrica" / "beam_2x" / "6" / "6.obj")
    p2 = trimesh.load(ASSETS / "fabrica" / "beam_2x" / "2" / "2.obj")
    img = render([(b6, GRAY), (p2, GREEN)], distance_factor=1.7, azim_deg=35, elev_deg=20)
    img.save(OUT / "task_inset_asm_pillar.png")

    # Asm-Beam: base 6 (gray) + part 2 (gray) + part 0 (green)
    p0 = trimesh.load(ASSETS / "fabrica" / "beam_2x" / "0" / "0.obj")
    img = render(
        [(b6, GRAY), (p2, GRAY), (p0, GREEN)],
        distance_factor=1.7,
        azim_deg=35,
        elev_deg=20,
    )
    img.save(OUT / "task_inset_asm_beam.png")

    # Screw-Leg: square_table_top (translucent gray) + square_table_leg4 (green).
    # Render in native Y-up frame (furniture_bench convention).
    # The leg is placed at its canonical assembled pose from one_leg_setup.json
    # (active_corner_pos_yup_obj, rpy=(0,0,0)) — leg's bottom sits at the
    # cavity floor. The table top OBJ has cavities that are only visible when
    # rendered with translucent material + double-sided + culling disabled, so
    # the cavity interior walls show through the outer surface.
    sqt_dir = ASSETS / "furniture_bench" / "square_table"
    top_mesh = _as_trimesh(trimesh.load(
        sqt_dir / "square_table_top" / "square_table_top.obj",
        force="mesh", process=False,
    ))
    leg_mesh = _as_trimesh(trimesh.load(
        sqt_dir / "square_table_leg4" / "square_table_leg4.obj",
        force="mesh", process=False,
    ))
    setup = json.load(open(sqt_dir / "one_leg_setup.json"))
    leg_pos_yup = np.array(setup["active_corner_pos_yup_obj"])
    leg_mesh.apply_transform(trimesh.transformations.translation_matrix(leg_pos_yup))

    img = render_yup(
        [(top_mesh, GRAY_TRANSLUCENT, "BLEND"), (leg_mesh, GREEN, "OPAQUE")],
        azim_deg=-60, elev_deg=30, ext=0.35,
    )
    img.save(OUT / "task_inset_screw_leg.png")

    print(f"wrote 4 insets to {OUT}")


if __name__ == "__main__":
    main()
