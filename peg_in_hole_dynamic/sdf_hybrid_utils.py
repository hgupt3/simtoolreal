"""Shared helpers for the SDF + CoACD hybrid asset pipelines.

The fabrica.beam_2x and fmb.peg_board_1 setup scripts both need:
  * trimesh ↔ manifold3d conversions for clean watertight slicing,
  * axis-range and XY-bbox boolean intersections to extract the
    inserter's "tip" and the receiver's "hole patch" out of canonical
    meshes,
  * a tiny XML emitter that appends an SDF collision element to a URDF
    `<link>`.

This module is intentionally framework-free: it deals only with
trimesh / manifold3d / xml.etree types and has no dependency on
PROBLEM_REGISTRY or Isaac Gym, so the asset-generation scripts can
import it cheaply.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Tuple

import numpy as np
import trimesh


# ─────────────────────────────────────────────────────────────────────────────
# Mesh ↔ manifold3d round-trip + boolean intersections
# ─────────────────────────────────────────────────────────────────────────────

def trimesh_to_manifold(m: trimesh.Trimesh):
    """Convert a trimesh.Trimesh into a manifold3d.Manifold (xyz only)."""
    import manifold3d
    mesh = manifold3d.Mesh(
        vert_properties=m.vertices.astype(np.float32),
        tri_verts=m.faces.astype(np.uint32),
    )
    return manifold3d.Manifold(mesh)


def manifold_to_trimesh(mf) -> trimesh.Trimesh:
    mesh = mf.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=np.float64)
    if verts.shape[1] > 3:
        verts = verts[:, :3]
    return trimesh.Trimesh(
        vertices=verts,
        faces=np.asarray(mesh.tri_verts, dtype=np.int64),
        process=False,
    )


def _box_around(extents: Tuple[float, float, float],
                center:  Tuple[float, float, float]) -> trimesh.Trimesh:
    return trimesh.creation.box(
        extents=extents,
        transform=trimesh.transformations.translation_matrix(list(center)),
    )


def slice_mesh_by_axis_range(mesh: trimesh.Trimesh, axis: int,
                              lo: float, hi: float,
                              other_pad: float = 0.001,
                              ) -> trimesh.Trimesh:
    """Boolean-intersect ``mesh`` with the slab ``axis ∈ [lo, hi]``.

    Used to peel the inserter's "tip" off its canonical mesh — pick
    the axis that has the largest extent (the long axis), then slice
    a `[max - tip_height, max]` slab.

    Returns a trimesh whose vertices already live in the same coordinate
    frame as the input (no re-centering). The result is watertight if
    the input was.
    """
    assert axis in (0, 1, 2)
    bb_min, bb_max = mesh.bounds.copy()
    extents = bb_max - bb_min
    extents[axis] = hi - lo
    center = (bb_min + bb_max) / 2.0
    center[axis] = (lo + hi) / 2.0
    # Pad the *other* axes by `other_pad` so the box certainly contains
    # all mesh vertices in those axes; this avoids manifold3d producing
    # tiny edge artefacts when the box exactly grazes the mesh boundary.
    extents[(axis + 1) % 3] += 2 * other_pad
    extents[(axis + 2) % 3] += 2 * other_pad
    box = _box_around(tuple(extents), tuple(center))
    a = trimesh_to_manifold(mesh)
    b = trimesh_to_manifold(box)
    return manifold_to_trimesh(a ^ b)   # ^ = boolean intersection


def _bbox_to_centered_box(bb_min, bb_max, pad: float) -> trimesh.Trimesh:
    bb_min = np.asarray(bb_min, dtype=float)
    bb_max = np.asarray(bb_max, dtype=float)
    extents = (bb_max - bb_min) + 2 * pad
    center = (bb_min + bb_max) / 2.0
    return _box_around(tuple(extents), tuple(center))


def slice_mesh_by_3d_bbox(mesh: trimesh.Trimesh,
                          bb_min, bb_max,
                          pad: float = 0.001,
                          ) -> trimesh.Trimesh:
    """Boolean-intersect ``mesh`` with the 3D axis-aligned box
    ``[bb_min, bb_max]`` (expanded by ``pad`` on each face)."""
    a = trimesh_to_manifold(mesh)
    b = trimesh_to_manifold(_bbox_to_centered_box(bb_min, bb_max, pad))
    return manifold_to_trimesh(a ^ b)


def subtract_3d_bbox_from_mesh(mesh: trimesh.Trimesh,
                                bb_min, bb_max,
                                pad: float = 0.001,
                                ) -> trimesh.Trimesh:
    """Boolean-subtract the 3D axis-aligned box ``[bb_min, bb_max]``
    (expanded by ``pad``) from ``mesh``. Returns the watertight remainder."""
    a = trimesh_to_manifold(mesh)
    b = trimesh_to_manifold(_bbox_to_centered_box(bb_min, bb_max, pad))
    return manifold_to_trimesh(a - b)


def slice_mesh_by_xy_bbox(mesh: trimesh.Trimesh,
                          x_lo: float, x_hi: float,
                          y_lo: float, y_hi: float,
                          z_pad: float = 0.001,
                          ) -> trimesh.Trimesh:
    """Boolean-intersect ``mesh`` with the column
    ``[x_lo, x_hi] × [y_lo, y_hi] × R``.

    Used to extract the receiver's "hole patch" from its canonical
    mesh (the rest of the receiver is approximated by axis-aligned
    boxes elsewhere).
    """
    bb_min, bb_max = mesh.bounds
    z_lo = float(bb_min[2]) - z_pad
    z_hi = float(bb_max[2]) + z_pad
    box = _box_around(
        extents=(x_hi - x_lo, y_hi - y_lo, z_hi - z_lo),
        center=((x_lo + x_hi) / 2, (y_lo + y_hi) / 2, (z_lo + z_hi) / 2),
    )
    a = trimesh_to_manifold(mesh)
    b = trimesh_to_manifold(box)
    return manifold_to_trimesh(a ^ b)


# ─────────────────────────────────────────────────────────────────────────────
# URDF emitter
# ─────────────────────────────────────────────────────────────────────────────

def _set_origin(elem: ET.Element,
                xyz: Tuple[float, float, float],
                rpy: Tuple[float, float, float]) -> None:
    ET.SubElement(elem, "origin", attrib={
        "xyz": f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}",
        "rpy": f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}",
    })


def add_mesh_visual(parent_link: ET.Element,
                    mesh_filename: str,
                    origin_xyz=(0.0, 0.0, 0.0),
                    origin_rpy=(0.0, 0.0, 0.0),
                    material_name: str | None = None) -> None:
    visual = ET.SubElement(parent_link, "visual")
    _set_origin(visual, origin_xyz, origin_rpy)
    geom = ET.SubElement(visual, "geometry")
    ET.SubElement(geom, "mesh", attrib={"filename": mesh_filename, "scale": "1 1 1"})
    if material_name is not None:
        ET.SubElement(visual, "material", attrib={"name": material_name})


def add_mesh_collision(parent_link: ET.Element,
                       mesh_filename: str,
                       origin_xyz=(0.0, 0.0, 0.0),
                       origin_rpy=(0.0, 0.0, 0.0)) -> None:
    coll = ET.SubElement(parent_link, "collision")
    _set_origin(coll, origin_xyz, origin_rpy)
    geom = ET.SubElement(coll, "geometry")
    ET.SubElement(geom, "mesh", attrib={"filename": mesh_filename, "scale": "1 1 1"})


def add_sdf_collision(parent_link: ET.Element,
                      mesh_filename: str,
                      resolution: int = 512,
                      origin_xyz=(0.0, 0.0, 0.0),
                      origin_rpy=(0.0, 0.0, 0.0)) -> None:
    """Append `<collision>...<mesh/><sdf resolution=N/></collision>`.

    The `<sdf>` element is what makes Isaac Gym cook a per-shape signed
    distance field at load time instead of using triangle-mesh contact.
    """
    coll = ET.SubElement(parent_link, "collision")
    _set_origin(coll, origin_xyz, origin_rpy)
    geom = ET.SubElement(coll, "geometry")
    ET.SubElement(geom, "mesh", attrib={"filename": mesh_filename, "scale": "1 1 1"})
    ET.SubElement(coll, "sdf", attrib={"resolution": str(resolution)})


def add_box_collision_visual(parent_link: ET.Element,
                              size: Tuple[float, float, float],
                              origin_xyz: Tuple[float, float, float],
                              origin_rpy=(0.0, 0.0, 0.0),
                              material_name: str | None = None) -> None:
    """Convenience: append a paired `<visual>` + `<collision>` of identical
    `<box>` geometry (used for the bulk frame around the active hole)."""
    for tag in ("visual", "collision"):
        elem = ET.SubElement(parent_link, tag)
        _set_origin(elem, origin_xyz, origin_rpy)
        geom = ET.SubElement(elem, "geometry")
        ET.SubElement(geom, "box", attrib={
            "size": f"{size[0]:.6f} {size[1]:.6f} {size[2]:.6f}",
        })
        if tag == "visual" and material_name is not None:
            ET.SubElement(elem, "material", attrib={"name": material_name})
