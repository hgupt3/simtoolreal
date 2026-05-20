#!/usr/bin/env python3
"""Produce a scaled clone of the FurnitureBench one_leg task assets.

Mirrors ``fabrica/scene_generation/scale_assembly.py`` but tailored to the
``furniture_bench/square_table`` layout. Only the assets touched by the
``furniture_bench.one_leg`` task are cloned:

  square_table/square_table_leg4/        -> <out>/square_table_leg4_2x/
  square_table/square_table_top/         -> <out>/square_table_top_2x/
  square_table/hole_patches/             -> <out>/hole_patches/  (OBJs scaled)
  square_table/insertion_fixtures/       -> <out>/insertion_fixtures/
  square_table/one_leg_setup.json        -> <out>/one_leg_setup.json

For each file:
  - .obj           : vertex lines (``v X Y Z [W]``) multiplied by ``--scale``.
  - .urdf          : ``<mesh filename=…>`` paths rewritten to the renamed
                     assets; ``<box size=…>`` and ``<origin xyz=…>`` values
                     scaled by ``--scale``. Robot/link/material names that
                     reference the renamed parts are rewritten.
  - canonical_meta.json: ``R_storage_to_canonical`` is a rotation matrix —
                     left untouched. (Sizes/positions are inferred from
                     the scaled mesh, no metadata needed here.)
  - one_leg_setup.json: all length-scale fields multiplied by ``--scale``;
                     ``parent_part`` / ``child_part`` / ``receptive_urdf``
                     paths rewritten for the new piece directory.
  - .mtl, .png     : copied verbatim.

Usage:
    python -m peg_in_hole_dynamic.furniture_bench.scene_generation.scale_one_leg \\
        --scale 2.0 --out square_table_2x [--force]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FB_ASSETS = REPO_ROOT / "assets" / "urdf" / "furniture_bench"
SRC_PIECE = "square_table"
SRC_PIECE_DIR = FB_ASSETS / SRC_PIECE

# Parts touched by the one_leg task — the only ones we clone.
PARTS_TO_RENAME = ("square_table_leg4", "square_table_top")
# Directories under square_table/ that we copy verbatim (with OBJs scaled
# but no rename). hole_patches is referenced by absolute path inside the
# insertion fixture URDFs.
LITERAL_DIRS = ("hole_patches", "insertion_fixtures")
# Top-level files at the piece root we want carried over (scaled if JSON,
# verbatim otherwise).
PIECE_LEVEL_FILES = ("one_leg_setup.json", "assembly.json")


def _scale_obj_file(src: Path, dst: Path, scale: float) -> int:
    n_vertices = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r") as fin, dst.open("w") as fout:
        for line in fin:
            if line.startswith("v "):
                parts = line.split()
                coords = [float(x) * scale for x in parts[1:]]
                fout.write("v " + " ".join(f"{c:.8f}" for c in coords) + "\n")
                n_vertices += 1
            else:
                fout.write(line)
    return n_vertices


def _scale_xyz_in_urdf(text: str, scale: float) -> str:
    """Multiply every ``xyz="X Y Z"`` attribute value by ``scale``."""

    def _rep(match: re.Match) -> str:
        vals = [float(v) * scale for v in match.group(1).split()]
        return 'xyz="' + " ".join(f"{v:.6f}" for v in vals) + '"'

    return re.sub(r'xyz="([^"]*)"', _rep, text)


def _scale_size_in_urdf(text: str, scale: float) -> str:
    """Multiply every ``<box size="X Y Z"/>`` by ``scale``. Cylinder ``length``
    and ``radius`` and sphere ``radius`` follow the same rule if present.
    """

    def _rep_size(match: re.Match) -> str:
        vals = [float(v) * scale for v in match.group(1).split()]
        return 'size="' + " ".join(f"{v:.6f}" for v in vals) + '"'

    text = re.sub(r'size="([^"]*)"', _rep_size, text)
    text = re.sub(
        r'radius="([^"]+)"',
        lambda m: f'radius="{float(m.group(1)) * scale:.6f}"',
        text,
    )
    text = re.sub(
        r'length="([^"]+)"',
        lambda m: f'length="{float(m.group(1)) * scale:.6f}"',
        text,
    )
    return text


def _rewrite_mesh_paths(text: str, replacements: dict[str, str]) -> str:
    """Apply each ``old -> new`` substring rewrite once over the URDF body.

    ``replacements`` is intentionally tried longest-first so a substring
    like ``square_table_leg4_thread.obj`` is rewritten before its prefix.
    """
    for old in sorted(replacements, key=len, reverse=True):
        new = replacements[old]
        text = text.replace(old, new)
    return text


def _scale_canonical_meta(src: Path, dst: Path) -> None:
    """``canonical_meta.json`` carries only rotation matrices (no lengths) so
    we just copy it verbatim. Kept as a function for symmetry with other
    JSON handlers if richer metadata gets added.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def _scale_setup_json(
    src: Path,
    dst: Path,
    scale: float,
    out_piece: str,
    part_rename: dict[str, str],
) -> None:
    data = json.loads(src.read_text())
    # Length-scale fields. ``hole_xy_canonical`` and ``active_corner_pos_yup_obj``
    # are positions; ``hole_xy_size_canonical`` is a size; ``leg_centroid_in_src``
    # and ``pos_canonical_centroid`` are positions.
    for key in (
        "hole_xy_canonical",
        "hole_xy_size_canonical",
        "active_corner_pos_yup_obj",
        "leg_centroid_in_src",
        "pos_canonical_centroid",
    ):
        if key in data:
            data[key] = [float(v) * scale for v in data[key]]
    if "parent_part" in data and data["parent_part"] in part_rename:
        data["parent_part"] = part_rename[data["parent_part"]]
    if "child_part" in data and data["child_part"] in part_rename:
        data["child_part"] = part_rename[data["child_part"]]
    if "receptive_urdf" in data:
        # path like "urdf/furniture_bench/square_table/insertion_fixtures/one_leg.urdf"
        data["receptive_urdf"] = data["receptive_urdf"].replace(
            f"/{SRC_PIECE}/", f"/{out_piece}/", 1,
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, indent=2))


def _scale_assembly_json(
    src: Path,
    dst: Path,
    scale: float,
    part_rename: dict[str, str],
) -> None:
    """Scale length fields (``pos``) and rename parts in upstream
    ``assembly.json``. ``rpy_xyz`` is rotational — left untouched.
    """
    data = json.loads(src.read_text())
    if "parts" in data:
        for entry in data["parts"]:
            n = entry.get("name")
            if n in part_rename:
                entry["name"] = part_rename[n]
    if "assembled_rel_poses_yup" in data:
        for key, poses in data["assembled_rel_poses_yup"].items():
            for pose in poses:
                if "pos" in pose:
                    pose["pos"] = [float(v) * scale for v in pose["pos"]]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, indent=2))


def _copy_and_process_part(
    src_part_dir: Path,
    dst_part_dir: Path,
    scale: float,
    part_rename: dict[str, str],
) -> None:
    """Clone one part dir (leg4 or top) with OBJ scaling + filename renames."""
    src_part = src_part_dir.name
    dst_part = dst_part_dir.name

    # Filename rewrites applied to URDF bodies (mesh refs + robot/link names).
    name_replacements = {src_part: dst_part}
    # Also rewrite any cross-piece part name (e.g. URDFs that mention the
    # other part by full canonical name).
    for old, new in part_rename.items():
        if old != src_part:
            name_replacements[old] = new

    for src_file in sorted(src_part_dir.rglob("*")):
        if src_file.is_dir():
            continue
        rel = src_file.relative_to(src_part_dir)
        # Rename any file whose stem starts with ``<src_part>`` so it tracks
        # the new part name. Subdir names (``coacd/``, ``sdf_hybrid/``) are
        # left alone.
        new_parts: list[str] = []
        for component in rel.parts:
            renamed = component
            for old, new in name_replacements.items():
                if old != new and old in renamed:
                    renamed = renamed.replace(old, new)
            new_parts.append(renamed)
        dst_file = dst_part_dir.joinpath(*new_parts)
        suffix = src_file.suffix.lower()

        if suffix == ".obj":
            n = _scale_obj_file(src_file, dst_file, scale)
            print(f"  obj  {rel}  ({n} verts × {scale})")
        elif suffix == ".urdf":
            text = src_file.read_text()
            text = _scale_xyz_in_urdf(text, scale)
            text = _scale_size_in_urdf(text, scale)
            text = _rewrite_mesh_paths(text, name_replacements)
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            dst_file.write_text(text)
            print(f"  urdf {rel}  →  {dst_file.relative_to(dst_part_dir)}")
        elif suffix == ".json":
            # Per-part canonical_meta.json carries rotation-only metadata.
            _scale_canonical_meta(src_file, dst_file)
            print(f"  json {rel}  (rotation-only, copied verbatim)")
        else:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_file, dst_file)
            print(f"  misc {rel}  (verbatim)")


def _copy_and_process_literal_dir(
    src_dir: Path,
    dst_dir: Path,
    scale: float,
    name_replacements: dict[str, str],
) -> None:
    """Clone a dir whose name is preserved (``hole_patches``,
    ``insertion_fixtures``). OBJs are scaled, URDFs get path/size/origin
    rewrites with ``name_replacements`` applied (so they point at the
    renamed sibling part dirs).
    """
    for src_file in sorted(src_dir.rglob("*")):
        if src_file.is_dir():
            continue
        rel = src_file.relative_to(src_dir)
        dst_file = dst_dir / rel
        suffix = src_file.suffix.lower()

        if suffix == ".obj":
            n = _scale_obj_file(src_file, dst_file, scale)
            print(f"  obj  {rel}  ({n} verts × {scale})")
        elif suffix == ".urdf":
            text = src_file.read_text()
            text = _scale_xyz_in_urdf(text, scale)
            text = _scale_size_in_urdf(text, scale)
            text = _rewrite_mesh_paths(text, name_replacements)
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            dst_file.write_text(text)
            print(f"  urdf {rel}  (paths+sizes rewritten)")
        else:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_file, dst_file)
            print(f"  misc {rel}  (verbatim)")


def scale_one_leg(scale: float, out_piece: str, force: bool) -> None:
    dst_piece_dir = FB_ASSETS / out_piece
    if not SRC_PIECE_DIR.is_dir():
        raise FileNotFoundError(f"missing source piece dir: {SRC_PIECE_DIR}")
    if out_piece == SRC_PIECE:
        raise ValueError(f"--out must differ from source piece ({SRC_PIECE!r})")
    if dst_piece_dir.exists():
        if not force:
            raise FileExistsError(
                f"{dst_piece_dir} already exists. Use --force to overwrite."
            )
        print(f"[scale] removing existing {dst_piece_dir}")
        shutil.rmtree(dst_piece_dir)
    dst_piece_dir.mkdir(parents=True)
    print(f"[scale] {SRC_PIECE_DIR}  →  {dst_piece_dir}  (scale={scale})")

    # Per-part renames (dir name + filename stems).
    # NOTE: pull off the trailing scale factor from out_piece, e.g.
    # ``square_table_2x`` -> ``_2x`` suffix to append to part names. Mirrors
    # the convention chosen in the asset registry.
    suffix = out_piece[len(SRC_PIECE):] if out_piece.startswith(SRC_PIECE) else f"_{out_piece}"
    part_rename = {p: f"{p}{suffix}" for p in PARTS_TO_RENAME}

    # --- 1. Per-part clones with renames ---
    for src_part in PARTS_TO_RENAME:
        src_part_dir = SRC_PIECE_DIR / src_part
        if not src_part_dir.is_dir():
            print(f"[skip] part dir missing: {src_part_dir}")
            continue
        dst_part_dir = dst_piece_dir / part_rename[src_part]
        print(f"[part] {src_part_dir.name}  →  {dst_part_dir.name}")
        _copy_and_process_part(src_part_dir, dst_part_dir, scale, part_rename)

    # --- 2. Literal-name dirs (hole_patches, insertion_fixtures) ---
    for literal in LITERAL_DIRS:
        src_lit_dir = SRC_PIECE_DIR / literal
        if not src_lit_dir.is_dir():
            continue
        dst_lit_dir = dst_piece_dir / literal
        print(f"[dir]  {literal}/")
        _copy_and_process_literal_dir(src_lit_dir, dst_lit_dir, scale, part_rename)

    # --- 3. Piece-level files (one_leg_setup.json, assembly.json if any) ---
    for fname in PIECE_LEVEL_FILES:
        src_file = SRC_PIECE_DIR / fname
        if not src_file.is_file():
            continue
        dst_file = dst_piece_dir / fname
        if fname == "assembly.json":
            _scale_assembly_json(src_file, dst_file, scale, part_rename)
            print(f"[json] {fname}  (assembled-pose positions × {scale}; parts renamed)")
        elif fname.endswith(".json"):
            _scale_setup_json(src_file, dst_file, scale, out_piece, part_rename)
            print(f"[json] {fname}  (length fields × {scale}; parts renamed)")
        else:
            shutil.copy(src_file, dst_file)
            print(f"[file] {fname}  (verbatim)")

    print(f"\n[scale] done → {dst_piece_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", type=float, default=2.0, help="uniform scale factor (default 2.0)")
    ap.add_argument("--out", default="square_table_2x", help="output piece name (default square_table_2x)")
    ap.add_argument("--force", action="store_true", help="overwrite existing output dir")
    args = ap.parse_args()
    if args.scale <= 0:
        print(f"--scale must be positive, got {args.scale}", file=sys.stderr)
        sys.exit(2)
    scale_one_leg(args.scale, args.out, args.force)


if __name__ == "__main__":
    main()
