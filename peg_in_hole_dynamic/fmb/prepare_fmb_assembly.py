#!/usr/bin/env python3
"""Restructure extracted FMB multi-object board meshes into Fabrica-compatible layout.

Copies OBJ+STL from assets/urdf/fmb/multi_object_boards/, centers all meshes
so the base plate's XY center is at the origin, and writes assembly_order.json.

Usage:
    python -m fmb.prepare_fmb_assembly
    python -m fmb.prepare_fmb_assembly --board 1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
FMB_DIR = REPO_ROOT / "assets" / "urdf" / "fmb"
SOURCE_DIR = FMB_DIR / "multi_object_boards"

BOARDS = [1, 2, 3]
PARTS_PER_BOARD = 5


def prepare_board(board_num: int, dry_run: bool = False) -> None:
    assembly_name = f"fmb_board_{board_num}"
    assembly_dir = FMB_DIR / assembly_name
    base_part = f"board_{board_num}_0"
    insert_parts = [f"board_{board_num}_{i}" for i in range(1, PARTS_PER_BOARD)]
    all_parts = [base_part] + insert_parts

    print(f"\n{'='*60}")
    print(f"Board {board_num}: {assembly_name}")
    print(f"  Base: {base_part}")
    print(f"  Inserts: {insert_parts}")

    # Load the base mesh to compute XY center for recentering.
    base_src = SOURCE_DIR / base_part / f"{base_part}.obj"
    base_mesh = trimesh.load(str(base_src), force="mesh", process=False)
    base_center_xy = (base_mesh.bounds[0][:2] + base_mesh.bounds[1][:2]) / 2.0
    print(f"  Base XY center: [{base_center_xy[0]:.4f}, {base_center_xy[1]:.4f}]")
    print(f"  Shifting all parts by [{-base_center_xy[0]:.4f}, {-base_center_xy[1]:.4f}, 0]")

    if dry_run:
        print("  (dry-run — no files written)")
        return

    assembly_dir.mkdir(parents=True, exist_ok=True)
    shift = np.array([-base_center_xy[0], -base_center_xy[1], 0.0])

    for part_id in all_parts:
        dst_dir = assembly_dir / part_id
        dst_dir.mkdir(parents=True, exist_ok=True)

        for ext in [".obj", ".stl"]:
            src = SOURCE_DIR / part_id / f"{part_id}{ext}"
            dst = dst_dir / f"{part_id}{ext}"
            if not src.exists():
                print(f"  WARNING: {src} not found")
                continue
            mesh = trimesh.load(str(src), force="mesh", process=False)
            mesh.vertices += shift
            mesh.export(str(dst))
            print(f"  Wrote: {dst.relative_to(REPO_ROOT)} (centered)")

    assembly_order = {
        "steps": all_parts,
        "inserts_into": {pid: base_part for pid in insert_parts},
    }
    order_path = assembly_dir / "assembly_order.json"
    with open(order_path, "w") as f:
        json.dump(assembly_order, f, indent=2)
    print(f"  Wrote: {order_path.relative_to(REPO_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare FMB board assemblies")
    parser.add_argument("--board", type=int, default=None,
                        help="Single board number (1/2/3). Default: all.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    boards = [args.board] if args.board else BOARDS
    for b in boards:
        assert b in BOARDS, f"Board must be one of {BOARDS}, got {b}"
        prepare_board(b, dry_run=args.dry_run)

    print(f"\nDone! Prepared {len(boards)} board assembly(s).")


if __name__ == "__main__":
    main()
