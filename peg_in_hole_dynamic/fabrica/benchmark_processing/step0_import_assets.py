#!/usr/bin/env python3
"""Import Fabrica OBJ assets: scale vertices cm->m.

Usage:
    python fabrica/benchmark_processing/step0_import_assets.py --assemblies beam

    # Inspect imported assemblies in viser:
    python fabrica/benchmark_processing/step0_import_assets.py --viz --assemblies beam
"""

import argparse
import time
from pathlib import Path

import trimesh

ALL_ASSEMBLIES = [
    "beam", "car", "cooling_manifold", "duct",
    "gamepad", "plumbers_block", "stool_circular",
]


def import_assembly(fabrica_dir: Path, output_dir: Path, assembly: str):
    src_dir = fabrica_dir / assembly
    obj_files = sorted(src_dir.glob("*.obj"))
    if not obj_files:
        print(f"  WARNING: No OBJ files found in {src_dir}")
        return 0

    count = 0
    for obj_path in obj_files:
        part_id = obj_path.stem
        part_dir = output_dir / assembly / part_id
        out_obj = part_dir / f"{part_id}.obj"

        mesh = trimesh.load_mesh(str(obj_path), process=False, maintain_order=True)
        mesh.vertices *= 0.01  # cm -> m

        part_dir.mkdir(parents=True, exist_ok=True)
        mesh.export(str(out_obj))
        count += 1
        print(f"  {assembly}/{part_id}: OBJ ({len(mesh.vertices)} verts)")

    return count


def run_viz(assemblies, output_dir, port):
    """Viser viewer: all parts in assembled frame with explode/assemble animation."""
    import numpy as np
    import viser
    from fabrica.viser_utils import COLORS, compute_explode_offsets, load_all_assemblies

    print("Loading meshes...")
    all_parts = load_all_assemblies(output_dir)
    parts = {n: p for n, p in all_parts.items() if n in assemblies}
    if not parts:
        print(f"No assemblies found in {output_dir} matching {assemblies}")
        return

    offsets = {n: compute_explode_offsets(p, spread=0.10) for n, p in parts.items()}

    # Grid layout
    grid = {}
    for idx, name in enumerate(parts):
        grid[name] = ((idx % 4) * 0.35, (idx // 4) * 0.35)

    server = viser.ViserServer(host="0.0.0.0", port=port)
    is_animating = False
    frames = {}

    for name, part_list in parts.items():
        gx, gy = grid[name]
        server.scene.add_label(f"/{name}/label", text=name, wxyz=(1, 0, 0, 0), position=(gx, gy, -0.05))
        for i, (pid, mesh) in enumerate(part_list):
            f = server.scene.add_frame(f"/{name}/frame_{pid}", wxyz=(1, 0, 0, 0), position=(gx, gy, 0), show_axes=False)
            frames[(name, i)] = f
            server.scene.add_mesh_simple(
                f"/{name}/frame_{pid}/mesh",
                vertices=np.array(mesh.vertices, dtype=np.float32),
                faces=np.array(mesh.faces, dtype=np.uint32),
                color=COLORS[i % len(COLORS)],
            )

    server.scene.add_frame("/origin", axes_length=0.05, axes_radius=0.002)

    def set_positions(t):
        a = t * t * (3.0 - 2.0 * t)  # ease in-out
        for name, part_list in parts.items():
            gx, gy = grid[name]
            for i, off in enumerate(offsets[name]):
                d = off * (1.0 - a)
                frames[(name, i)].position = (gx + d[0], gy + d[1], d[2])

    def animate(forward=True):
        nonlocal is_animating
        if is_animating:
            return
        is_animating = True
        n = 60  # 2s at 30fps
        for fi in range(n + 1):
            t = fi / n
            set_positions(t if forward else 1.0 - t)
            time.sleep(1.0 / 30)
        is_animating = False

    server.gui.add_button("Assemble").on_click(lambda _: animate(True))
    server.gui.add_button("Explode").on_click(lambda _: animate(False))
    set_positions(0.0)

    print(f"\nOpen http://localhost:{port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(description="Import Fabrica OBJ assets")
    parser.add_argument("--fabrica-dir", type=Path,
                        default=Path("/share/portal/kk837/Fabrica/assets/fabrica"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent / "assets" / "urdf" / "fabrica")
    parser.add_argument("--assemblies", nargs="+", default=ALL_ASSEMBLIES, choices=ALL_ASSEMBLIES)
    parser.add_argument("--viz", action="store_true", help="Launch viser viewer instead of importing")
    parser.add_argument("--port", type=int, default=8082, help="Viser port for --viz")
    args = parser.parse_args()

    if args.viz:
        run_viz(args.assemblies, args.output_dir, args.port)
        return

    total = 0
    for assembly in args.assemblies:
        print(f"Importing {assembly}...")
        total += import_assembly(args.fabrica_dir, args.output_dir, assembly)
    print(f"\nDone. Imported {total} parts.")


if __name__ == "__main__":
    main()
