#!/usr/bin/env python3
"""Generate peg + multiple hole variants for peg-in-hole benchmark.

Peg: cuboidal hammer (handle 25x3x2 cm, perpendicular head 10x2x2 cm).
Holes: rectangular slot with varying clearance tolerances.

Outputs to assets/urdf/peg_in_hole/:
  peg/{peg.obj, peg.mtl, peg_texture.png, peg.stl, peg.urdf}
  holes/hole_tol{T}mm/{hole.obj, hole.mtl, hole_texture.png, hole.stl,
                       hole_tol{T}mm.urdf, scene.urdf,
                       trajectories/peg/pick_place.json}

Peg body frame: handle long axis along +X, tip at X=0, head at +X end.
Hole body frame: centered on X/Y, block base at Z=0, top opening at Z=floor_t+depth.

World frame (matches fabrica conventions):
  TABLE_Z = 0.38      — scene-origin z in world
  Table box = 0.475 x 0.4 x 0.3, centered at scene origin (top at world z=0.53)
  Hole placed at scene (0.12, -0.152, 0.15) — fabrica fixture location
"""

import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets/urdf/peg_in_hole"

HANDLE_EXTENTS = (0.25, 0.03, 0.02)
HEAD_EXTENTS = (0.02, 0.10, 0.02)
# Body origin is at the HANDLE GEOMETRIC CENTER so the URDF origin lands on
# the natural grasp point (policy scale uses handle extents — see objects.py).
HANDLE_CENTER = (0.0, 0.0, 0.0)
HEAD_CENTER = (0.115, 0.0, 0.0)   # head outer face flush with handle +X end (0.125)

HOLE_FOOTPRINT_X = 0.08
HOLE_FOOTPRINT_Y = 0.08
HOLE_SLOT_CORE_X = 0.02
HOLE_SLOT_CORE_Y = 0.03
HOLE_FLOOR_THICKNESS = 0.01
HOLE_DEPTH = 0.05

TOLERANCES_MM = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0]

PEG_COLOR = (204, 40, 40)
HOLE_COLOR = (120, 120, 120)
DENSITY = 1250.0

# World/scene placement (fabrica conventions)
TABLE_Z = 0.38
TABLE_SIZE = (0.475, 0.4, 0.3)
# Scene-local center of hole base. Z=0.15 puts hole on table top. XY places it just
# off the peg's -X end, same Y row as the peg (peg start at (0.135, 0.08) spans
# x in [0.01, 0.26]; hole spans x in [-0.075, -0.025], y in [0.055, 0.105] => ~3.5 cm gap).
HOLE_SCENE_OFFSET = (-0.05, 0.08, 0.15)

# Peg start pose (world frame) — flat on table, handle along +X, in robot reach
PEG_START_XYZ = (0.135, 0.08, 0.54)       # tip at this pos; handle bottom face rests on table top (0.53)
PEG_START_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)

# Orientation for vertical insertion: body +X -> world +Z (rotation -90° about Y)
PEG_INSERT_QUAT_XYZW = (0.0, -0.70710678, 0.0, 0.70710678)


def fmt_tol(tol_mm: float) -> str:
    if tol_mm == int(tol_mm):
        return str(int(tol_mm))
    return f"{tol_mm}".replace(".", "p")


def box_mesh(center, extents):
    m = trimesh.creation.box(extents=np.asarray(extents, dtype=float))
    m.apply_translation(np.asarray(center, dtype=float))
    return m


def peg_boxes():
    return [
        (HANDLE_CENTER, HANDLE_EXTENTS),
        (HEAD_CENTER, HEAD_EXTENTS),
    ]


def hole_boxes(tol_m):
    slot_x = HOLE_SLOT_CORE_X + 2 * tol_m
    slot_y = HOLE_SLOT_CORE_Y + 2 * tol_m
    t = HOLE_FLOOR_THICKNESS
    d = HOLE_DEPTH
    ox, oy = HOLE_FOOTPRINT_X, HOLE_FOOTPRINT_Y

    assert slot_x < ox, f"slot_x {slot_x} >= outer {ox} — increase HOLE_FOOTPRINT_X or reduce tol"
    assert slot_y <= oy, f"slot_y {slot_y} > outer {oy} — increase HOLE_FOOTPRINT_Y or reduce tol"

    boxes = [((0.0, 0.0, t / 2), (ox, oy, t))]
    zc = t + d / 2
    ew = (ox - slot_x) / 2
    if ew > 1e-6:
        boxes.append(((slot_x / 2 + ew / 2, 0.0, zc), (ew, oy, d)))
        boxes.append(((-(slot_x / 2 + ew / 2), 0.0, zc), (ew, oy, d)))
    nl = (oy - slot_y) / 2
    if nl > 1e-6:
        boxes.append(((0.0, slot_y / 2 + nl / 2, zc), (slot_x, nl, d)))
        boxes.append(((0.0, -(slot_y / 2 + nl / 2), zc), (slot_x, nl, d)))
    return boxes


def boxes_to_mesh(boxes):
    return trimesh.util.concatenate([box_mesh(c, e) for c, e in boxes])


def write_obj_mtl_png(out_dir: Path, name: str, mesh, rgb):
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / f"{name}.obj"
    mtl_path = out_dir / f"{name}.mtl"
    png_path = out_dir / f"{name}_texture.png"

    Image.new("RGB", (32, 32), rgb).save(png_path)

    mtl_lines = [
        f"newmtl {name}_mat",
        "Ka 0.2 0.2 0.2",
        f"Kd {rgb[0]/255:.4f} {rgb[1]/255:.4f} {rgb[2]/255:.4f}",
        "Ks 0.0 0.0 0.0",
        "d 1.0",
        "illum 2",
        f"map_Kd {name}_texture.png",
    ]
    mtl_path.write_text("\n".join(mtl_lines) + "\n")

    V = mesh.vertices
    F = mesh.faces
    FN = mesh.face_normals
    obj_lines = [f"mtllib {name}.mtl", f"o {name}", f"usemtl {name}_mat"]
    for v in V:
        obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    obj_lines.append("vt 0.0 0.0")
    for n in FN:
        obj_lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
    for fi, face in enumerate(F):
        v1, v2, v3 = face
        ni = fi + 1
        obj_lines.append(
            f"f {v1+1}/1/{ni} {v2+1}/1/{ni} {v3+1}/1/{ni}"
        )
    obj_path.write_text("\n".join(obj_lines) + "\n")


def write_stl(out_dir: Path, name: str, mesh):
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_dir / f"{name}.stl"))


def box_xml(boxes, tag: str, material_name=None, indent="    "):
    parts = []
    for c, e in boxes:
        mat = f"\n{indent}  <material name=\"{material_name}\"/>" if (material_name and tag == "visual") else ""
        parts.append(
            f'{indent}<{tag}>\n'
            f'{indent}  <origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" rpy="0 0 0"/>\n'
            f'{indent}  <geometry><box size="{e[0]:.6f} {e[1]:.6f} {e[2]:.6f}"/></geometry>{mat}\n'
            f'{indent}</{tag}>'
        )
    return "\n".join(parts)


def write_peg_urdf(out_dir: Path):
    boxes = peg_boxes()
    h_c, h_e = boxes[0]
    d_c, d_e = boxes[1]
    vol_h = h_e[0] * h_e[1] * h_e[2]
    vol_d = d_e[0] * d_e[1] * d_e[2]
    vol = vol_h + vol_d
    mass = DENSITY * vol
    com_x = (h_c[0] * vol_h + d_c[0] * vol_d) / vol
    ixx = mass * (h_e[1] ** 2 + h_e[2] ** 2) / 12
    iyy = mass * (h_e[0] ** 2 + h_e[2] ** 2) / 12
    izz = mass * (h_e[0] ** 2 + h_e[1] ** 2) / 12

    vis = box_xml(boxes, "visual", material_name="peg_red")
    col = box_xml(boxes, "collision")
    r, g, b = PEG_COLOR
    xml = f'''<?xml version="1.0"?>
<robot name="peg">
  <material name="peg_red">
    <color rgba="{r/255:.4f} {g/255:.4f} {b/255:.4f} 1.0"/>
  </material>
  <link name="peg">
{vis}
{col}
    <inertial>
      <mass value="{mass:.6f}"/>
      <origin xyz="{com_x:.6f} 0 0" rpy="0 0 0"/>
      <inertia ixx="{ixx:.6e}" ixy="0" ixz="0" iyy="{iyy:.6e}" iyz="0" izz="{izz:.6e}"/>
    </inertial>
  </link>
</robot>
'''
    (out_dir / "peg.urdf").write_text(xml)


def _scene_box_xml(center_scene, extents, material_name=None, indent="    "):
    """Emit visual+collision XML for a single box in the scene URDF's 'box' link."""
    cx, cy, cz = center_scene
    ex, ey, ez = extents
    mat = f'\n{indent}  <material name="{material_name}"/>' if material_name else ""
    return (
        f'{indent}<visual>\n'
        f'{indent}  <origin xyz="{cx:.6f} {cy:.6f} {cz:.6f}" rpy="0 0 0"/>\n'
        f'{indent}  <geometry><box size="{ex:.6f} {ey:.6f} {ez:.6f}"/></geometry>{mat}\n'
        f'{indent}</visual>\n'
        f'{indent}<collision>\n'
        f'{indent}  <origin xyz="{cx:.6f} {cy:.6f} {cz:.6f}" rpy="0 0 0"/>\n'
        f'{indent}  <geometry><box size="{ex:.6f} {ey:.6f} {ez:.6f}"/></geometry>\n'
        f'{indent}</collision>'
    )


def write_scene_urdf(out_dir: Path, tol_m: float, tol_tag: str):
    """Scene URDF: fixed table + hole (at fabrica fixture location).

    Mirrors assets/urdf/fabrica/beam/environments/{pid}/scene.urdf: single link 'box',
    heavy mass, fixed base. The hole boxes are positioned in scene-local coords by
    offsetting each hole-local box origin by HOLE_SCENE_OFFSET.
    """
    ox_s, oy_s, oz_s = HOLE_SCENE_OFFSET
    parts = [_scene_box_xml((0.0, 0.0, 0.0), TABLE_SIZE, material_name="wood")]
    for (cx, cy, cz), ext in hole_boxes(tol_m):
        parts.append(_scene_box_xml(
            (cx + ox_s, cy + oy_s, cz + oz_s), ext, material_name="hole_grey",
        ))
    geom_xml = "\n".join(parts)
    r, g, b = HOLE_COLOR
    xml = f'''<?xml version="1.0"?>
<robot name="peg_in_hole_scene_tol{tol_tag}mm">
  <material name="wood"><color rgba="0.82 0.56 0.35 1.0"/></material>
  <material name="hole_grey"><color rgba="{r/255:.4f} {g/255:.4f} {b/255:.4f} 1.0"/></material>
  <link name="box">
{geom_xml}
    <inertial>
      <mass value="500"/>
      <friction value="1.0"/>
      <inertia ixx="1000.0" ixy="0.0" ixz="0.0" iyy="1000.0" iyz="0.0" izz="1000.0"/>
    </inertial>
  </link>
</robot>
'''
    (out_dir / "scene.urdf").write_text(xml)


def write_pick_place_json(out_dir: Path):
    """Hardcoded peg pick-place waypoints in world frame.

    Sequence: lift (3 steps) -> lift+rotate -> above-hole -> pre-insert -> inserted.
    Poses are for the peg URDF origin (= handle geometric center); quaternion xyzw.
    With the insert quat (body +X -> world +Z), the tip is at (origin_z - 0.125)
    and the head-end at (origin_z + 0.125), so "tip at world Z = T" means
    center_z = T + 0.125.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sx, sy, sz = PEG_START_XYZ
    qx, qy, qz, qw = PEG_INSERT_QUAT_XYZW

    hx, hy, _ = HOLE_SCENE_OFFSET
    hole_floor_world_z = TABLE_Z + HOLE_SCENE_OFFSET[2] + HOLE_FLOOR_THICKNESS   # 0.54
    hole_top_world_z = hole_floor_world_z + HOLE_DEPTH                           # 0.59
    half_handle = HANDLE_EXTENTS[0] / 2                                          # 0.125

    start = [sx, sy, sz, 0.0, 0.0, 0.0, 1.0]
    goals = [
        [sx, sy, sz + 0.05, 0.0, 0.0, 0.0, 1.0],                                 # lift 5cm
        [sx, sy, sz + 0.10, 0.0, 0.0, 0.0, 1.0],                                 # lift 10cm
        [sx, sy, sz + 0.15, 0.0, 0.0, 0.0, 1.0],                                 # lift 15cm
        [sx, sy, sz + 0.15 + half_handle, qx, qy, qz, qw],                       # rotate to vertical (tip lifted to old lift-15cm height)
        [hx, hy, hole_top_world_z + 0.10 + half_handle, qx, qy, qz, qw],         # above hole (tip 10cm above opening)
        [hx, hy, hole_top_world_z + 0.01 + half_handle, qx, qy, qz, qw],         # pre-insert (tip 1cm above opening)
        [hx, hy, hole_floor_world_z + 0.001 + half_handle, qx, qy, qz, qw],      # inserted (tip 1mm above floor)
    ]
    (out_dir / "pick_place.json").write_text(
        json.dumps({"start_pose": start, "goals": goals}, indent=4) + "\n"
    )


def write_hole_urdf(out_dir: Path, tol_m: float, urdf_name: str, robot_name: str):
    boxes = hole_boxes(tol_m)
    vis = box_xml(boxes, "visual", material_name="hole_grey")
    col = box_xml(boxes, "collision")
    r, g, b = HOLE_COLOR
    xml = f'''<?xml version="1.0"?>
<robot name="{robot_name}">
  <material name="hole_grey">
    <color rgba="{r/255:.4f} {g/255:.4f} {b/255:.4f} 1.0"/>
  </material>
  <link name="hole">
{vis}
{col}
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.055" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
</robot>
'''
    (out_dir / urdf_name).write_text(xml)


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    peg_dir = ASSETS_DIR / "peg"
    peg_mesh = boxes_to_mesh(peg_boxes())
    write_obj_mtl_png(peg_dir, "peg", peg_mesh, PEG_COLOR)
    write_stl(peg_dir, "peg", peg_mesh)
    write_peg_urdf(peg_dir)
    print(f"[peg] extents={peg_mesh.extents}  -> {peg_dir}")

    for tol_mm in TOLERANCES_MM:
        tol_m = tol_mm * 1e-3
        tag = fmt_tol(tol_mm)
        hname = f"hole_tol{tag}mm"
        hdir = ASSETS_DIR / "holes" / hname
        hmesh = boxes_to_mesh(hole_boxes(tol_m))
        write_obj_mtl_png(hdir, "hole", hmesh, HOLE_COLOR)
        write_stl(hdir, "hole", hmesh)
        write_hole_urdf(hdir, tol_m, urdf_name=f"{hname}.urdf", robot_name=hname)
        write_scene_urdf(hdir, tol_m, tag)
        write_pick_place_json(hdir / "trajectories" / "peg")
        slot = (HOLE_SLOT_CORE_X + 2 * tol_m, HOLE_SLOT_CORE_Y + 2 * tol_m)
        print(f"[{hname}] slot=({slot[0]:.4f}, {slot[1]:.4f}) -> {hdir}")

    print("Done.")


if __name__ == "__main__":
    main()
