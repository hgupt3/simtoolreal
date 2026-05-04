"""Visualize extracted FMB meshes in a viser web viewer.

Shows pegs, peg boards (with pegs above in assembled layout), and
multi-object assemblies (assembled vs exploded views).

Run: python -m fmb.visualize_fmb
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import trimesh
import tyro
import viser

REPO_ROOT = Path(__file__).resolve().parent.parent
FMB_DIR = REPO_ROOT / "assets" / "urdf" / "fmb"

COLORS_9 = [
    (230, 51, 51),
    (51, 179, 51),
    (51, 77, 230),
    (230, 179, 26),
    (204, 77, 204),
    (26, 204, 204),
    (230, 128, 26),
    (128, 128, 128),
    (153, 51, 153),
]

BOARD_COLOR = (180, 170, 150)
FIXTURE_COLOR = (140, 140, 160)


def load_obj(path: Path) -> trimesh.Trimesh:
    return trimesh.load(str(path), force="mesh", process=False)


def discover_objects(fmb_dir: Path) -> Dict[str, List[Tuple[str, Path]]]:
    """Discover all OBJ files grouped by subdirectory."""
    groups = {}
    for subdir_name in ["pegs", "boards", "fixtures", "multi_object_boards"]:
        subdir = fmb_dir / subdir_name
        if not subdir.exists():
            continue
        items = []
        for obj_dir in sorted(subdir.iterdir()):
            if not obj_dir.is_dir():
                continue
            obj_files = list(obj_dir.glob("*.obj"))
            if obj_files:
                items.append((obj_dir.name, obj_files[0]))
        if items:
            groups[subdir_name] = items
    return groups


def add_mesh(server, name: str, mesh: trimesh.Trimesh, offset: np.ndarray, color, visible: bool = True):
    vertices = (mesh.vertices + offset).astype(np.float32)
    handle = server.scene.add_mesh_simple(
        name,
        vertices=vertices,
        faces=mesh.faces.astype(np.int32),
        color=color,
        flat_shading=True,
        visible=visible,
    )
    return handle


@dataclass
class VizArgs:
    """Visualize FMB meshes in a viser web viewer."""

    fmb_dir: Path = FMB_DIR
    port: int = 8082
    peg_cols: int = 9
    """Columns in peg grid (9 = one per shape)."""
    peg_spacing: float = 0.10
    """Grid spacing for pegs in meters."""
    explode_factor: float = 2.0
    """How far apart to spread parts in exploded view."""


def main() -> None:
    args: VizArgs = tyro.cli(VizArgs)

    groups = discover_objects(args.fmb_dir)
    if not groups:
        print(f"No meshes found in {args.fmb_dir}. Run convert_step_to_mesh.py first.")
        return

    extents_path = args.fmb_dir / "fmb_extents.json"
    extents_data = {}
    if extents_path.exists():
        with open(extents_path) as f:
            extents_data = json.load(f)

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = True

    # ── Section 1: Pegs in a grid ───────────────────────────────────────
    section_x = 0.0
    if "pegs" in groups:
        pegs = groups["pegs"]
        short_pegs = [
            (n, p) for n, p in pegs
            if extents_data.get(n, {}).get("extents_m", [0, 0, 0])[2] <= 0.11
        ]
        long_pegs = [
            (n, p) for n, p in pegs
            if extents_data.get(n, {}).get("extents_m", [0, 0, 0])[2] > 0.11
        ]

        server.gui.add_folder("Pegs")

        y_offset = 0.0
        for group_label, group_pegs in [
            ("Short Pegs (100mm)", short_pegs),
            ("Long Pegs (150mm)", long_pegs),
        ]:
            server.scene.add_label(
                f"/labels/{group_label}",
                group_label,
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                position=np.array([section_x - 0.05, y_offset - 0.04, 0.12]),
            )

            for idx, (name, obj_path) in enumerate(group_pegs):
                mesh = load_obj(obj_path)
                col = idx % args.peg_cols
                row = idx // args.peg_cols
                offset = np.array([
                    section_x + col * args.peg_spacing,
                    y_offset + row * args.peg_spacing,
                    0.0,
                ]) - mesh.centroid
                color = COLORS_9[idx % len(COLORS_9)]
                add_mesh(server, f"/pegs/{name}", mesh, offset, color)

            rows_used = (len(group_pegs) - 1) // args.peg_cols + 1
            y_offset += rows_used * args.peg_spacing + 0.08

    # ── Section 2: Peg boards with pegs shown assembled above ───────────
    section_x = 1.2
    if "boards" in groups and "pegs" in groups:
        boards = groups["boards"]
        pegs = groups["pegs"]
        short_pegs = [
            (n, p) for n, p in pegs
            if extents_data.get(n, {}).get("extents_m", [0, 0, 0])[2] <= 0.11
        ]

        # 3 boards × 9 shapes × 2 lengths = 54, but we don't know exact mapping.
        # Group short pegs into 3 groups of 9 by size (sorted by max XY extent).
        short_sorted = sorted(
            short_pegs,
            key=lambda np_: max(
                extents_data.get(np_[0], {}).get("extents_m", [0, 0, 0])[:2]
            ),
            reverse=True,
        )
        peg_groups = [short_sorted[i:i + 9] for i in range(0, len(short_sorted), 9)]

        server.scene.add_label(
            "/labels/Peg Boards (assembled)",
            "Peg Boards — short pegs grouped by size above each board",
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            position=np.array([section_x - 0.05, -0.04, 0.12]),
        )

        for board_idx, (board_name, board_path) in enumerate(boards):
            board_mesh = load_obj(board_path)
            board_center = board_mesh.centroid.copy()

            bx = section_x + board_idx * 0.30
            board_offset = np.array([bx, 0.0, 0.0]) - board_center
            add_mesh(server, f"/peg_assembly/{board_name}", board_mesh, board_offset, BOARD_COLOR)

            # Place pegs in 3×3 grid above board
            if board_idx < len(peg_groups):
                for peg_idx, (peg_name, peg_path) in enumerate(peg_groups[board_idx]):
                    peg_mesh = load_obj(peg_path)
                    pcol = peg_idx % 3
                    prow = peg_idx // 3
                    peg_offset = np.array([
                        bx - 0.06 + pcol * 0.06,
                        -0.06 + prow * 0.06,
                        0.08,  # float above board
                    ]) - peg_mesh.centroid
                    color = COLORS_9[peg_idx % len(COLORS_9)]
                    add_mesh(server, f"/peg_assembly/pegs_{board_name}/{peg_name}", peg_mesh, peg_offset, color)

    # ── Section 3: Multi-object boards — assembled vs exploded ──────────
    section_x = 2.4
    if "multi_object_boards" in groups:
        mo_items = groups["multi_object_boards"]

        # Group by board number: board_1_0..board_1_4, board_2_0..board_2_4, etc.
        board_groups: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
        for name, path in mo_items:
            # "board_1_0" -> "board_1"
            parts = name.rsplit("_", 1)
            board_key = parts[0]
            board_groups[board_key].append((name, path))

        server.scene.add_label(
            "/labels/Multi-Object Assembled",
            "Multi-Object Boards — Assembled (left) vs Exploded (right)",
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            position=np.array([section_x - 0.05, -0.04, 0.15]),
        )

        for board_num, (board_key, parts) in enumerate(sorted(board_groups.items())):
            by = board_num * 0.40

            # Load all part meshes
            part_meshes = []
            for name, path in parts:
                mesh = load_obj(path)
                part_meshes.append((name, mesh))

            # Compute the overall assembly center (from all parts combined)
            all_verts = np.concatenate([m.vertices for _, m in part_meshes])
            assembly_center = (all_verts.min(axis=0) + all_verts.max(axis=0)) / 2.0

            # --- Assembled view (left) ---
            for pidx, (name, mesh) in enumerate(part_meshes):
                offset = np.array([section_x, by, 0.0]) - assembly_center
                color = BOARD_COLOR if pidx == 0 else COLORS_9[pidx % len(COLORS_9)]
                add_mesh(server, f"/multi_assembled/{board_key}/{name}", mesh, offset, color)

            # --- Exploded view (right) ---
            explode_x = section_x + 0.40
            for pidx, (name, mesh) in enumerate(part_meshes):
                part_center = mesh.centroid
                direction = part_center - assembly_center
                direction[2] = max(direction[2], 0.01)  # bias upward
                explode_offset = direction * args.explode_factor
                offset = np.array([explode_x, by, 0.0]) - assembly_center + explode_offset
                color = BOARD_COLOR if pidx == 0 else COLORS_9[pidx % len(COLORS_9)]
                add_mesh(server, f"/multi_exploded/{board_key}/{name}", mesh, offset, color)

            # Labels
            server.scene.add_label(
                f"/labels/{board_key}_assembled",
                f"{board_key} (assembled)",
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                position=np.array([section_x, by - 0.05, 0.10]),
            )
            server.scene.add_label(
                f"/labels/{board_key}_exploded",
                f"{board_key} (exploded)",
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                position=np.array([explode_x, by - 0.05, 0.10]),
            )

    # ── Section 4: Fixtures ─────────────────────────────────────────────
    section_x = 3.6
    if "fixtures" in groups:
        fixtures = groups["fixtures"]
        server.scene.add_label(
            "/labels/Fixtures",
            "Fixtures",
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            position=np.array([section_x - 0.05, -0.04, 0.10]),
        )
        for idx, (name, path) in enumerate(fixtures):
            mesh = load_obj(path)
            offset = np.array([section_x + idx * 0.20, 0.0, 0.0]) - mesh.centroid
            add_mesh(server, f"/fixtures/{name}", mesh, offset, FIXTURE_COLOR)
            server.scene.add_label(
                f"/labels/fixture_{name}",
                name,
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                position=np.array([section_x + idx * 0.20, -0.03, 0.06]),
            )

    print(f"\nFMB viewer running at http://localhost:{args.port}")
    print("Layout: Pegs | Peg boards + pegs | Multi-object assembled/exploded | Fixtures")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
