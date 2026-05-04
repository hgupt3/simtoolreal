#!/usr/bin/env python3
"""Generate solid-color textured OBJs for the real-world object tracker.

For each fabrica assembly part we load the canonical mesh, UV-unwrap it with
xatlas, attach a solid-color texture, and export an OBJ with `TextureVisuals`
(required by FoundationPose-style trackers). The shared MTL and PNG live at
the assembly root; the colored OBJ sits next to each canonical part OBJ with a
`_colored` suffix and references the MTL via `mtllib ../material.mtl`.

Layout produced per assembly:

    assets/urdf/fabrica/<assembly>/
      material.mtl
      material_0.png
      <pid>/<pid>_colored.obj   (next to the existing <pid>.obj)

Usage:
    python fabrica/generate_colored_objs.py                      # all, red
    python fabrica/generate_colored_objs.py --assembly beam
    python fabrica/generate_colored_objs.py --color 31 119 209   # custom RGB
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

import numpy as np
import trimesh
import xatlas
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"

ASSEMBLIES = [
    "beam",
    "beam_2x",
    "car",
    "cooling_manifold",
    "duct",
    "gamepad",
    "plumbers_block",
    "stool_circular",
]

DEFAULT_COLOR = (209, 31, 31)

MTL_NAME = "material.mtl"
PNG_NAME = "material_0.png"
COLORED_SUFFIX = "_colored.obj"


def find_parts(assembly_dir: Path):
    parts = []
    for d in sorted(assembly_dir.iterdir()):
        if d.is_dir() and (d / f"{d.name}.obj").exists():
            parts.append(d.name)
    return parts


def load_canonical_mesh(assembly_dir: Path, part_id: str) -> trimesh.Trimesh:
    canonical = assembly_dir / part_id / f"{part_id}_canonical.obj"
    regular = assembly_dir / part_id / f"{part_id}.obj"
    path = canonical if canonical.exists() else regular
    return trimesh.load_mesh(str(path), process=False)


def build_textured(mesh: trimesh.Trimesh, color) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)

    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)
    atlas.generate()
    vmapping, new_faces, uvs = atlas[0]

    out = trimesh.Trimesh(vertices=vertices[vmapping], faces=new_faces, process=False)
    texture = Image.new("RGB", (8, 8), color)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture,
        baseColorFactor=np.array([*color, 255], dtype=np.uint8),
    )
    out.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    return out


def export_part(mesh: trimesh.Trimesh, tmpdir: Path) -> tuple[Path, Path, Path]:
    """Export OBJ+MTL+PNG into tmpdir; return their paths."""
    obj_path = tmpdir / "part.obj"
    mesh.export(str(obj_path), file_type="obj")
    mtl_path = tmpdir / MTL_NAME
    png_path = tmpdir / PNG_NAME
    assert obj_path.exists(), f"OBJ not exported: {obj_path}"
    assert mtl_path.exists(), f"MTL not exported: {mtl_path}"
    assert png_path.exists(), f"PNG not exported: {png_path}"
    return obj_path, mtl_path, png_path


def rewrite_mtllib(obj_src: Path, obj_dst: Path, new_mtllib: str) -> None:
    """Copy obj_src to obj_dst, rewriting the mtllib line."""
    text = obj_src.read_text()
    lines = text.splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith("mtllib "):
            lines[i] = f"mtllib {new_mtllib}"
            replaced = True
            break
    assert replaced, f"no mtllib line found in {obj_src}"
    obj_dst.write_text("\n".join(lines) + "\n")


def rewrite_mtl_texture_path(mtl_src: Path, mtl_dst: Path, new_texture_rel: str) -> None:
    """Copy MTL from tmp to shared location, rewriting map_Kd so the texture
    path is resolved correctly relative to the *OBJ* (trimesh and most OBJ
    loaders look for textures relative to the OBJ, not the MTL).
    """
    text = mtl_src.read_text()
    lines = text.splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith("map_Kd "):
            lines[i] = f"map_Kd {new_texture_rel}"
            replaced = True
            break
    assert replaced, f"no map_Kd line found in {mtl_src}"
    mtl_dst.write_text("\n".join(lines) + "\n")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def verify_colored_obj(obj_path: Path, expected_color) -> tuple:
    mesh = trimesh.load(str(obj_path), force="mesh", process=False)
    assert isinstance(mesh.visual, trimesh.visual.TextureVisuals), (
        f"{obj_path}: expected TextureVisuals, got {type(mesh.visual).__name__}"
    )
    img = mesh.visual.material.image
    assert img is not None, f"{obj_path}: material.image is None"
    arr = np.array(img.convert("RGB"))
    assert arr.size > 0, f"{obj_path}: empty texture image"
    # Every pixel must equal the requested color (8x8 solid).
    sample = tuple(int(c) for c in arr.reshape(-1, 3)[0])
    assert sample == tuple(expected_color), (
        f"{obj_path}: texture color {sample} != expected {tuple(expected_color)}"
    )
    assert mesh.visual.uv is not None and len(mesh.visual.uv) > 0, (
        f"{obj_path}: no UV coordinates"
    )
    return arr.shape, mesh.visual.uv.shape


def process_assembly(assembly: str, color) -> None:
    assembly_dir = ASSETS_DIR / assembly
    if not assembly_dir.exists():
        print(f"WARNING: {assembly_dir} does not exist, skipping")
        return
    part_ids = find_parts(assembly_dir)
    if not part_ids:
        print(f"WARNING: no parts found in {assembly_dir}, skipping")
        return

    print(f"\n{'=' * 60}")
    print(f"Assembly: {assembly} ({len(part_ids)} parts)")
    print(f"{'=' * 60}")

    shared_mtl_dst = assembly_dir / MTL_NAME
    shared_png_dst = assembly_dir / PNG_NAME
    shared_mtl_hash = None
    shared_png_hash = None

    for pid in part_ids:
        canonical = load_canonical_mesh(assembly_dir, pid)
        textured = build_textured(canonical, color)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            obj_tmp, mtl_tmp, png_tmp = export_part(textured, tmp)

            # Place the shared MTL/PNG from the first part; verify byte-identical
            # for later parts so we know the dedup is safe.
            if shared_mtl_hash is None:
                rewrite_mtl_texture_path(
                    mtl_tmp, shared_mtl_dst, f"../{PNG_NAME}"
                )
                shutil.copy2(png_tmp, shared_png_dst)
                shared_mtl_hash = file_hash(mtl_tmp)
                shared_png_hash = file_hash(shared_png_dst)
                print(f"  Wrote {shared_mtl_dst.relative_to(REPO_ROOT)}")
                print(f"  Wrote {shared_png_dst.relative_to(REPO_ROOT)}")
            else:
                # Hash the trimesh-exported MTL/PNG (before rewrite) so we
                # confirm the texture pipeline is deterministic across parts.
                assert file_hash(mtl_tmp) == shared_mtl_hash, (
                    f"{assembly}/{pid}: generated MTL differs from first part "
                    f"— texture pipeline not deterministic"
                )
                assert file_hash(png_tmp) == shared_png_hash, (
                    f"{assembly}/{pid}: generated PNG differs from first part "
                    f"— texture pipeline not deterministic"
                )

            obj_dst = assembly_dir / pid / f"{pid}{COLORED_SUFFIX}"
            rewrite_mtllib(obj_tmp, obj_dst, f"../{MTL_NAME}")

        img_shape, uv_shape = verify_colored_obj(obj_dst, color)
        print(
            f"  {pid}: wrote {obj_dst.relative_to(REPO_ROOT)} "
            f"(image={img_shape}, uv={uv_shape})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate solid-color textured OBJs for the fabrica tracker"
    )
    parser.add_argument(
        "--assembly",
        type=str,
        default=None,
        help="Process only this assembly (default: all)",
    )
    parser.add_argument(
        "--color",
        type=int,
        nargs=3,
        default=list(DEFAULT_COLOR),
        metavar=("R", "G", "B"),
        help=f"Texture RGB color in 0-255 (default: {' '.join(map(str, DEFAULT_COLOR))})",
    )
    args = parser.parse_args()

    for c in args.color:
        assert 0 <= c <= 255, f"color component out of 0-255: {c}"
    color = tuple(args.color)

    assemblies = [args.assembly] if args.assembly else ASSEMBLIES

    print(f"Color: RGB{color}")
    print(f"Target: {ASSETS_DIR}")

    for assembly in assemblies:
        process_assembly(assembly, color)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
