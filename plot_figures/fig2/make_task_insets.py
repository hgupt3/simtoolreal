"""Render fig 2 panel-(a) task-config insets from OBJ assets.

Outputs:
    outputs/fig2_panel_bcd/task_insets/peg_in_hole.png
    outputs/fig2_panel_bcd/task_insets/asm_pillar.png
    outputs/fig2_panel_bcd/task_insets/asm_beam.png

Convention:
    bright green = part being inserted
    dull gray    = part(s) being inserted into

Screw-Leg inset is not generated here; the square_table_leg / square_table_top
OBJs live on a different branch (the "future" branch with screw assets).
"""

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from pathlib import Path

import numpy as np
import pyrender
import trimesh
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets" / "urdf"
OUT = REPO / "outputs" / "fig2_panel_bcd" / "task_insets"

GRAY = [150, 150, 155, 255]
GREEN = [70, 200, 90, 255]


def _colored_mesh(m: trimesh.Trimesh, color):
    base = np.array(color, dtype=np.float32) / 255.0
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=base.tolist(), metallicFactor=0.1, roughnessFactor=0.6
    )
    return pyrender.Mesh.from_trimesh(m, material=material, smooth=False)


def render(meshes_colors, size=(480, 480), azim_deg=35, elev_deg=25, distance_factor=1.7):
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Peg-In-Hole: peg (green) hovering above hole (gray)
    peg = trimesh.load(ASSETS / "peg_in_hole" / "peg" / "peg.obj")
    hole = trimesh.load(ASSETS / "peg_in_hole" / "holes" / "hole_tol0p5mm" / "hole.obj")
    R_y = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    R_z = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
    peg.apply_transform(R_z @ R_y)
    peg.apply_transform(trimesh.transformations.translation_matrix([0, 0, 0.06 + 0.13]))
    img = render([(hole, GRAY), (peg, GREEN)], distance_factor=1.6, azim_deg=30, elev_deg=20)
    img.save(OUT / "peg_in_hole.png")

    # Asm-Pillar: base 6 (gray) + part 2 (green)
    b6 = trimesh.load(ASSETS / "fabrica" / "beam_2x" / "6" / "6.obj")
    p2 = trimesh.load(ASSETS / "fabrica" / "beam_2x" / "2" / "2.obj")
    img = render([(b6, GRAY), (p2, GREEN)], distance_factor=1.7, azim_deg=35, elev_deg=20)
    img.save(OUT / "asm_pillar.png")

    # Asm-Beam: base 6 (gray) + part 2 (gray) + part 0 (green)
    p0 = trimesh.load(ASSETS / "fabrica" / "beam_2x" / "0" / "0.obj")
    img = render(
        [(b6, GRAY), (p2, GRAY), (p0, GREEN)],
        distance_factor=1.7,
        azim_deg=35,
        elev_deg=20,
    )
    img.save(OUT / "asm_beam.png")

    print(f"wrote 3 insets to {OUT}")


if __name__ == "__main__":
    main()
