#!/usr/bin/env python3
"""Generate solid-red textured canonical OBJs for FMB peg tracking.

This mirrors ``peg_in_hole_dynamic/fabrica/generate_colored_objs.py`` for the
FMB long pegs.  The pegs root acts like the Fabrica assembly root: one shared
MTL/PNG lives there, and each peg directory gets a colored copy of its
canonical mesh.

Layout produced:

    assets/urdf/fmb/pegs/
      material.mtl
      material_0.png
      peg_46/peg_46_canonical_colored.obj

Usage:
    python -m peg_in_hole_dynamic.fmb.generate_colored_peg_objs
    python -m peg_in_hole_dynamic.fmb.generate_colored_peg_objs --peg peg_46
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


REPO_ROOT = Path(__file__).resolve().parents[2]
PEGS_DIR = REPO_ROOT / "assets" / "urdf" / "fmb" / "pegs"

DEFAULT_COLOR = (209, 31, 31)

MTL_NAME = "material.mtl"
PNG_NAME = "material_0.png"
COLORED_SUFFIX = "_canonical_colored.obj"


def _peg_sort_key(path: Path) -> tuple[int, str]:
    prefix, _, suffix = path.name.partition("_")
    if prefix == "peg" and suffix.isdigit():
        return int(suffix), path.name
    return 10_000, path.name


def find_pegs() -> list[str]:
    pegs = []
    for peg_dir in sorted(PEGS_DIR.iterdir(), key=_peg_sort_key):
        if not peg_dir.is_dir():
            continue
        canonical = peg_dir / f"{peg_dir.name}_canonical.obj"
        if canonical.is_file():
            pegs.append(peg_dir.name)
    return pegs


def load_canonical_mesh(peg_name: str) -> trimesh.Trimesh:
    path = PEGS_DIR / peg_name / f"{peg_name}_canonical.obj"
    return trimesh.load_mesh(str(path), process=False)


def build_textured(mesh: trimesh.Trimesh, color: tuple[int, int, int]) -> trimesh.Trimesh:
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
    obj_path = tmpdir / "part.obj"
    mesh.export(str(obj_path), file_type="obj")
    mtl_path = tmpdir / MTL_NAME
    png_path = tmpdir / PNG_NAME
    assert obj_path.exists(), f"OBJ not exported: {obj_path}"
    assert mtl_path.exists(), f"MTL not exported: {mtl_path}"
    assert png_path.exists(), f"PNG not exported: {png_path}"
    return obj_path, mtl_path, png_path


def rewrite_mtllib(obj_src: Path, obj_dst: Path, new_mtllib: str) -> None:
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


def verify_colored_obj(obj_path: Path, expected_color: tuple[int, int, int]) -> tuple:
    mesh = trimesh.load(str(obj_path), force="mesh", process=False)
    assert isinstance(mesh.visual, trimesh.visual.TextureVisuals), (
        f"{obj_path}: expected TextureVisuals, got {type(mesh.visual).__name__}"
    )
    img = mesh.visual.material.image
    assert img is not None, f"{obj_path}: material.image is None"
    arr = np.array(img.convert("RGB"))
    assert arr.size > 0, f"{obj_path}: empty texture image"
    sample = tuple(int(c) for c in arr.reshape(-1, 3)[0])
    assert sample == tuple(expected_color), (
        f"{obj_path}: texture color {sample} != expected {tuple(expected_color)}"
    )
    assert mesh.visual.uv is not None and len(mesh.visual.uv) > 0, (
        f"{obj_path}: no UV coordinates"
    )
    return arr.shape, mesh.visual.uv.shape


def process_pegs(peg_names: list[str], color: tuple[int, int, int]) -> None:
    if not PEGS_DIR.is_dir():
        raise SystemExit(f"Missing FMB pegs directory: {PEGS_DIR}")
    if not peg_names:
        raise SystemExit(f"No canonical peg OBJ files found in {PEGS_DIR}")

    shared_mtl_dst = PEGS_DIR / MTL_NAME
    shared_png_dst = PEGS_DIR / PNG_NAME
    shared_mtl_hash = None
    shared_png_hash = None

    print(f"Color: RGB{color}")
    print(f"Target: {PEGS_DIR.relative_to(REPO_ROOT)}")
    print(f"Pegs: {', '.join(peg_names)}")

    for peg_name in peg_names:
        canonical = load_canonical_mesh(peg_name)
        textured = build_textured(canonical, color)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            obj_tmp, mtl_tmp, png_tmp = export_part(textured, tmp)

            if shared_mtl_hash is None:
                rewrite_mtl_texture_path(mtl_tmp, shared_mtl_dst, f"../{PNG_NAME}")
                shutil.copy2(png_tmp, shared_png_dst)
                shared_mtl_hash = file_hash(mtl_tmp)
                shared_png_hash = file_hash(shared_png_dst)
                print(f"  Wrote {shared_mtl_dst.relative_to(REPO_ROOT)}")
                print(f"  Wrote {shared_png_dst.relative_to(REPO_ROOT)}")
            else:
                assert file_hash(mtl_tmp) == shared_mtl_hash, (
                    f"{peg_name}: generated MTL differs from first peg"
                )
                assert file_hash(png_tmp) == shared_png_hash, (
                    f"{peg_name}: generated PNG differs from first peg"
                )

            obj_dst = PEGS_DIR / peg_name / f"{peg_name}{COLORED_SUFFIX}"
            rewrite_mtllib(obj_tmp, obj_dst, f"../{MTL_NAME}")

        img_shape, uv_shape = verify_colored_obj(obj_dst, color)
        print(
            f"  {peg_name}: wrote {obj_dst.relative_to(REPO_ROOT)} "
            f"(image={img_shape}, uv={uv_shape})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate solid-red textured canonical OBJ copies for FMB pegs"
    )
    parser.add_argument(
        "--peg",
        action="append",
        default=None,
        help="Process only this peg name, e.g. peg_46. May be passed multiple times.",
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

    available = find_pegs()
    if args.peg:
        missing = sorted(set(args.peg) - set(available))
        if missing:
            raise SystemExit(f"Requested pegs have no canonical OBJ: {', '.join(missing)}")
        peg_names = [p for p in available if p in set(args.peg)]
    else:
        peg_names = available

    process_pegs(peg_names, color)
    print("\nDone.")


if __name__ == "__main__":
    main()
