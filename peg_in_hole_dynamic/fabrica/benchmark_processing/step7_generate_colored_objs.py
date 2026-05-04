#!/usr/bin/env python3
"""Generate colored OBJ meshes with texture for FoundationPose.

For each assembly part, loads the canonical mesh, UV-unwraps it with xatlas,
assigns a solid-color texture image, and exports OBJ+MTL+PNG. The resulting
meshes have proper TextureVisuals that FoundationPose requires.

Usage:
    python fabrica/benchmark_processing/step7_generate_colored_objs.py
    python fabrica/benchmark_processing/step7_generate_colored_objs.py --assembly beam
    python fabrica/benchmark_processing/step7_generate_colored_objs.py --color 31 119 209
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import trimesh
import xatlas
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"
OUTPUT_BASE = ASSETS_DIR / "colored_obj_red"

ASSEMBLIES = [
    "beam", "car", "cooling_manifold", "duct",
    "gamepad", "plumbers_block", "stool_circular",
]

DEFAULT_COLOR = (209, 31, 31)  # red


def find_parts(assembly_dir):
    """Find all part subdirectories that contain an OBJ mesh."""
    parts = []
    for d in sorted(assembly_dir.iterdir()):
        if d.is_dir() and (d / f"{d.name}.obj").exists():
            parts.append(d.name)
    return parts


def load_canonical_mesh(assembly_dir, part_id):
    """Load canonical OBJ if it exists, otherwise fall back to regular OBJ."""
    canonical = assembly_dir / part_id / f"{part_id}_canonical.obj"
    regular = assembly_dir / part_id / f"{part_id}.obj"
    path = canonical if canonical.exists() else regular
    mesh = trimesh.load_mesh(str(path), process=False)
    print(f"  Loaded {path.relative_to(ASSETS_DIR)}")
    return mesh


def add_colored_texture(mesh, color):
    """UV-unwrap mesh with xatlas and assign a solid-color texture image."""
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32)

    # UV-unwrap with xatlas
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)
    atlas.generate()
    vmapping, new_faces, uvs = atlas[0]

    # Rebuild mesh with new topology from xatlas
    new_vertices = vertices[vmapping]
    mesh_out = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

    # Create solid-color texture image (8x8)
    texture = Image.new("RGB", (8, 8), color)

    # Assign texture visuals
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture,
        baseColorFactor=np.array([*color, 255], dtype=np.uint8),
    )
    mesh_out.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)

    return mesh_out


def verify_mesh(obj_path):
    """Verify exported mesh has proper TextureVisuals with a non-None image."""
    mesh = trimesh.load(str(obj_path), force="mesh", process=False)
    assert isinstance(mesh.visual, trimesh.visual.TextureVisuals), \
        f"{obj_path}: expected TextureVisuals, got {type(mesh.visual).__name__}"
    img = mesh.visual.material.image
    assert img is not None, f"{obj_path}: material.image is None"
    arr = np.array(img.convert("RGB"))
    assert arr.shape[0] > 0 and arr.shape[1] > 0, f"{obj_path}: image has zero size"
    assert mesh.visual.uv is not None and len(mesh.visual.uv) > 0, \
        f"{obj_path}: no UV coordinates"
    return arr.shape, mesh.visual.uv.shape


def process_assembly(assembly, color):
    """Process all parts of an assembly."""
    assembly_dir = ASSETS_DIR / assembly
    if not assembly_dir.exists():
        print(f"WARNING: {assembly_dir} does not exist, skipping")
        return

    part_ids = find_parts(assembly_dir)
    if not part_ids:
        print(f"WARNING: no parts found in {assembly_dir}, skipping")
        return

    out_dir = OUTPUT_BASE / assembly
    # Clean and recreate output directory
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Assembly: {assembly} ({len(part_ids)} parts)")
    print(f"{'='*60}")

    for pid in part_ids:
        mesh = load_canonical_mesh(assembly_dir, pid)
        textured = add_colored_texture(mesh, color)

        obj_path = out_dir / f"{pid}.obj"
        textured.export(str(obj_path), file_type="obj")
        print(f"  Exported {obj_path.relative_to(REPO_ROOT)}")

        # Verify
        img_shape, uv_shape = verify_mesh(obj_path)
        print(f"  Verified: image={img_shape}, uv={uv_shape}")


def cleanup_old_files():
    """Remove old red_plastic.mtl files and stale README."""
    for mtl in OUTPUT_BASE.rglob("red_plastic.mtl"):
        mtl.unlink()
        print(f"  Removed old {mtl.relative_to(REPO_ROOT)}")
    readme = OUTPUT_BASE / "README.md"
    if readme.exists():
        readme.unlink()
        print(f"  Removed old {readme.relative_to(REPO_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Generate colored OBJ meshes for FoundationPose")
    parser.add_argument("--assembly", type=str, default=None,
                        help="Process only this assembly (default: all)")
    parser.add_argument("--color", type=int, nargs=3, default=list(DEFAULT_COLOR),
                        metavar=("R", "G", "B"), help="Texture color (default: 209 31 31)")
    args = parser.parse_args()

    color = tuple(args.color)
    assemblies = [args.assembly] if args.assembly else ASSEMBLIES

    print(f"Color: RGB{color}")
    print(f"Output: {OUTPUT_BASE}")

    cleanup_old_files()

    for assembly in assemblies:
        process_assembly(assembly, color)

    print(f"\nDone! All meshes written to {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
