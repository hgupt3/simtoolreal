"""Convert FMB STEP files to individual OBJ meshes.

Extracts each solid body from the STEP assemblies, tessellates it,
converts mm to meters, and saves as OBJ files.
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh
import tyro

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_STEP_DIR = REPO_ROOT / "assets" / "urdf" / "fmb" / "raw_step"
OUTPUT_DIR = REPO_ROOT / "assets" / "urdf" / "fmb"

MM_TO_M = 0.001


def _import_cadquery():
    import cadquery as cq
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.BRep import BRep_Tool
    from OCP.StlAPI import StlAPI_Writer
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    return cq, TopExp_Explorer, TopAbs_SOLID, BRep_Tool, StlAPI_Writer, BRepMesh_IncrementalMesh


def extract_solids_from_step(step_path: Path) -> List:
    """Load a STEP file and return individual OCC solid shapes."""
    cq, TopExp_Explorer, TopAbs_SOLID, _, _, _ = _import_cadquery()

    result = cq.importers.importStep(str(step_path))
    compound = result.val().wrapped

    explorer = TopExp_Explorer(compound, TopAbs_SOLID)
    solids = []
    while explorer.More():
        solids.append(explorer.Current())
        explorer.Next()

    return solids


def solid_to_trimesh(solid, linear_deflection: float = 0.1, angular_deflection: float = 0.5) -> trimesh.Trimesh:
    """Tessellate an OCC solid and return a trimesh object (in mm)."""
    _, _, _, BRep_Tool, StlAPI_Writer, BRepMesh_IncrementalMesh = _import_cadquery()

    BRepMesh_IncrementalMesh(solid, linear_deflection, False, angular_deflection, True)

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = tmp.name

    writer = StlAPI_Writer()
    writer.Write(solid, tmp_path)

    mesh = trimesh.load(tmp_path, force="mesh")
    Path(tmp_path).unlink(missing_ok=True)
    return mesh


def process_step_file(
    step_path: Path,
    output_base: Path,
    prefix: str,
    scale: float = MM_TO_M,
) -> List[Tuple[str, dict]]:
    """Extract all solids from a STEP file, save as OBJ, return metadata."""
    print(f"\nProcessing {step_path.name}...")
    solids = extract_solids_from_step(step_path)
    print(f"  Found {len(solids)} solid(s)")

    results = []
    for i, solid in enumerate(solids):
        if len(solids) == 1:
            name = prefix
        else:
            name = f"{prefix}_{i}"

        mesh = solid_to_trimesh(solid)
        mesh.apply_scale(scale)

        obj_dir = output_base / name
        obj_dir.mkdir(parents=True, exist_ok=True)
        obj_path = obj_dir / f"{name}.obj"
        stl_path = obj_dir / f"{name}.stl"

        mesh.export(str(obj_path))
        mesh.export(str(stl_path))

        extents = mesh.bounding_box.extents.tolist()
        centroid = mesh.centroid.tolist()
        volume = float(mesh.volume) if mesh.is_volume else 0.0

        meta = {
            "name": name,
            "extents_m": extents,
            "centroid_m": centroid,
            "volume_m3": volume,
            "num_vertices": len(mesh.vertices),
            "num_faces": len(mesh.faces),
            "source_file": step_path.name,
            "solid_index": i,
        }
        results.append((name, meta))
        print(f"  [{i:2d}] {name}: extents {[f'{e:.4f}' for e in extents]} m, {len(mesh.faces)} faces")

    return results


@dataclass
class ConvertArgs:
    """Convert FMB STEP files to OBJ meshes."""

    raw_step_dir: Path = RAW_STEP_DIR
    output_dir: Path = OUTPUT_DIR
    scale: float = MM_TO_M
    """Scale factor applied to meshes (default: mm to meters)."""


def main() -> None:
    args: ConvertArgs = tyro.cli(ConvertArgs)

    all_metadata = {}

    # Pegs: peg.step contains all 54 peg solids
    peg_step = args.raw_step_dir / "peg.step"
    if peg_step.exists():
        results = process_step_file(
            peg_step, args.output_dir / "pegs", "peg", scale=args.scale
        )
        for name, meta in results:
            all_metadata[name] = meta
    else:
        print(f"WARNING: {peg_step} not found, skipping pegs")

    # Peg boards: peg_board.step contains 3 board solids
    board_step = args.raw_step_dir / "peg_board.step"
    if board_step.exists():
        results = process_step_file(
            board_step, args.output_dir / "boards", "peg_board", scale=args.scale
        )
        for name, meta in results:
            all_metadata[name] = meta
    else:
        print(f"WARNING: {board_step} not found, skipping peg boards")

    # Peg fixture
    fixture_step = args.raw_step_dir / "peg_fixture.step"
    if fixture_step.exists():
        results = process_step_file(
            fixture_step, args.output_dir / "fixtures", "peg_fixture", scale=args.scale
        )
        for name, meta in results:
            all_metadata[name] = meta

    # Multi-object boards (Board 1, 2, 3)
    for board_num in [1, 2, 3]:
        board_path = args.raw_step_dir / f"board_{board_num}.step"
        if board_path.exists():
            results = process_step_file(
                board_path,
                args.output_dir / "multi_object_boards",
                f"board_{board_num}",
                scale=args.scale,
            )
            for name, meta in results:
                all_metadata[name] = meta

    # Board fixture
    bf_step = args.raw_step_dir / "board_fixture.step"
    if bf_step.exists():
        results = process_step_file(
            bf_step, args.output_dir / "fixtures", "board_fixture", scale=args.scale
        )
        for name, meta in results:
            all_metadata[name] = meta

    # Save metadata
    extents_path = args.output_dir / "fmb_extents.json"
    with open(extents_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"\nSaved metadata for {len(all_metadata)} objects to {extents_path}")


if __name__ == "__main__":
    main()
