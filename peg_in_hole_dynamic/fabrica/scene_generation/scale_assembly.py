#!/usr/bin/env python3
"""Produce a scaled clone of a fabrica assembly directory.

Why this exists: Isaac Gym does not honor the URDF ``<mesh scale=...>``
attribute reliably for collision shapes. To make beams 2× larger (or any
arbitrary scale), we rewrite the OBJ vertex coordinates directly and update
the metadata that downstream code uses (``canonical_transforms.json`` and
``stable_rests/*.npz``). URDFs are copied verbatim — they reference the
already-scaled OBJs.

Usage:
    python -m fabrica.scene_generation.scale_assembly \\
        --assembly beam --scale 2.0 --out-name beam_2x [--force]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"


def _scale_obj_file(src: Path, dst: Path, scale: float) -> int:
    """Stream-rewrite an OBJ file, multiplying every ``v`` line by ``scale``.

    Returns the number of vertex lines rewritten.
    """
    n_vertices = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r") as fin, dst.open("w") as fout:
        for line in fin:
            # Vertex line: starts with literal "v " (NOT "vn"/"vt"/"vp").
            if line.startswith("v "):
                parts = line.split()
                # parts[0] == "v"; parts[1:] are coordinate floats (3 or 4).
                # OBJ allows an optional w component on v; scale it too — for
                # geometric vertices the w defaults to 1, so scaling by S is
                # only meaningful if w != 1, but we apply it uniformly to be
                # safe with rational/homogeneous-coord OBJs.
                coords = [float(x) * scale for x in parts[1:]]
                fout.write("v " + " ".join(f"{c:.8f}" for c in coords) + "\n")
                n_vertices += 1
            else:
                fout.write(line)
    return n_vertices


def _scale_canonical_transforms(src: Path, dst: Path, scale: float) -> int:
    """Multiply ``original_centroid`` and ``canonical_extents`` by ``scale``.

    ``assembled_to_canonical_wxyz`` is a unit quaternion — left untouched.
    Returns the number of part entries rewritten.
    """
    with src.open("r") as f:
        data = json.load(f)
    for pid, entry in data.items():
        entry["original_centroid"] = [c * scale for c in entry["original_centroid"]]
        entry["canonical_extents"] = [e * scale for e in entry["canonical_extents"]]
    with dst.open("w") as f:
        json.dump(data, f, indent=2)
    return len(data)


def _scale_stable_rests(src: Path, dst: Path, scale: float) -> tuple[int, int]:
    """Rescale the translation block of every 4×4 stable-rest transform.

    Returns (n_rests, n_extra_keys) for logging.
    """
    data = np.load(src)
    if "transforms" not in data.files:
        raise RuntimeError(
            f"{src}: expected key 'transforms' in stable rests npz, got {data.files}"
        )
    transforms = data["transforms"].copy()  # (K, 4, 4)
    transforms[:, :3, 3] *= scale
    out = {"transforms": transforms}
    extras = [k for k in data.files if k != "transforms"]
    for k in extras:
        out[k] = data[k]
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)
    return transforms.shape[0], len(extras)


def scale_assembly(assembly: str, scale: float, out_name: str, force: bool) -> None:
    src_dir = ASSETS_DIR / assembly
    dst_dir = ASSETS_DIR / out_name

    if not src_dir.is_dir():
        raise FileNotFoundError(f"source assembly not found: {src_dir}")
    if out_name == assembly:
        raise ValueError(
            f"--out-name must differ from --assembly (both are {assembly!r})"
        )
    if dst_dir.exists():
        if not force:
            raise FileExistsError(
                f"{dst_dir} already exists. Use --force to overwrite."
            )
        print(f"[scale] removing existing {dst_dir}")
        shutil.rmtree(dst_dir)

    print(f"[scale] {src_dir}  →  {dst_dir}  (scale={scale})")
    dst_dir.mkdir(parents=True)

    # --- 1. Walk part dirs and rewrite OBJs / copy-and-rename URDFs ---
    # URDF filenames are prefixed by the source assembly name (e.g.
    # ``beam_0.urdf``); rename them to ``{out_name}_{pid}...`` so the
    # fabrica.objects registry picks them up as {out_name}_{pid}_coacd etc.
    part_dirs = sorted([p for p in src_dir.iterdir() if p.is_dir()])
    urdf_prefix_from = f"{assembly}_"
    urdf_prefix_to = f"{out_name}_"
    for part_dir in part_dirs:
        if part_dir.name in ("stable_rests", "environments", "trajectories"):
            continue
        for src_file in part_dir.rglob("*"):
            if src_file.is_dir():
                continue
            rel_path = src_file.relative_to(src_dir)
            # Rename URDF files: swap the assembly prefix.
            if src_file.suffix.lower() == ".urdf" and src_file.name.startswith(urdf_prefix_from):
                new_name = urdf_prefix_to + src_file.name[len(urdf_prefix_from):]
                dst_rel = rel_path.parent / new_name
            else:
                dst_rel = rel_path
            dst_file = dst_dir / dst_rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            suffix = src_file.suffix.lower()
            if suffix == ".obj":
                n = _scale_obj_file(src_file, dst_file, scale)
                print(f"[scale]   obj  {rel_path}  ({n} verts × {scale})")
            elif suffix == ".urdf":
                shutil.copy(src_file, dst_file)
                print(f"[scale]   urdf {rel_path}  →  {dst_rel}")
            elif suffix in (".mtl", ".png", ".jpg", ".jpeg"):
                shutil.copy(src_file, dst_file)
                print(f"[scale]   asset {rel_path}  (verbatim)")
            else:
                shutil.copy(src_file, dst_file)
                print(f"[scale]   misc {rel_path}  (verbatim)")

    # --- 2. canonical_transforms.json ---
    src_ct = src_dir / "canonical_transforms.json"
    if src_ct.exists():
        n = _scale_canonical_transforms(src_ct, dst_dir / "canonical_transforms.json", scale)
        print(f"[scale] canonical_transforms.json: {n} parts (centroids+extents × {scale})")

    # --- 3. assembly_order.json ---
    src_ao = src_dir / "assembly_order.json"
    if src_ao.exists():
        shutil.copy(src_ao, dst_dir / "assembly_order.json")
        print(f"[scale] assembly_order.json (verbatim)")

    # --- 4. stable_rests/*.npz ---
    src_rests = src_dir / "stable_rests"
    if src_rests.is_dir():
        dst_rests = dst_dir / "stable_rests"
        for src_npz in sorted(src_rests.glob("*.npz")):
            n_rests, n_extras = _scale_stable_rests(
                src_npz, dst_rests / src_npz.name, scale
            )
            print(
                f"[scale] stable_rests/{src_npz.name}: {n_rests} rests "
                f"(translations × {scale}, kept {n_extras} extra keys)"
            )

    print(f"\n[scale] done → {dst_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assembly", required=True, help="source assembly name (e.g., beam)")
    ap.add_argument("--scale", type=float, required=True, help="uniform scale factor (e.g., 2.0)")
    ap.add_argument("--out-name", required=True, help="output assembly name (e.g., beam_2x)")
    ap.add_argument("--force", action="store_true", help="overwrite existing output dir")
    args = ap.parse_args()

    if args.scale <= 0:
        print(f"--scale must be positive, got {args.scale}", file=sys.stderr)
        sys.exit(2)

    scale_assembly(args.assembly, args.scale, args.out_name, args.force)


if __name__ == "__main__":
    main()
