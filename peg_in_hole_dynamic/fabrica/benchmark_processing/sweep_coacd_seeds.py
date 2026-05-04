#!/usr/bin/env python3
"""Sweep CoACD seeds until all insertions pass validation.

For each seed, runs step4 (CoACD) on all parts in parallel, regenerates scene
URDFs (step5), and validates all non-base parts in parallel (step6). Stops on
the first seed where every part passes.

Usage:
    python sweep_coacd_seeds.py --assembly car
    python sweep_coacd_seeds.py --assembly beam --max-seeds 100
    python sweep_coacd_seeds.py --assembly car --coacd-args "--preprocess-mode off --threshold 0.05"
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ASSETS_DIR = Path("assets/urdf/fabrica")

MAX_POS_MM = 3.0
MAX_KP_MM = 5.0


def load_assembly_info(assembly):
    # type: (str) -> Tuple[List[str], List[str]]
    """Load assembly_order.json and return steps and parts to validate."""
    order_path = ASSETS_DIR / assembly / "assembly_order.json"
    with open(order_path) as f:
        order = json.load(f)

    steps = order["steps"]

    # Parts to validate = all non-base parts (everything after the first step)
    base_part = steps[0]
    parts_to_validate = [p for p in steps if p != base_part]

    return steps, parts_to_validate


def _run_step4(assembly, part, seed, extra_args):
    # type: (str, str, int, List[str]) -> Tuple[str, str]
    """Run CoACD decomposition for a single part. Returns (part, summary)."""
    cmd = [
        sys.executable, "fabrica/benchmark_processing/step4_run_coacd.py",
        "--assembly", assembly, "--part", part, "--seed", str(seed),
    ] + extra_args
    r = subprocess.run(cmd, capture_output=True, text=True)
    summary_lines = []
    for line in r.stdout.split("\n"):
        if "Convex Hulls" in line or "overshoot" in line:
            summary_lines.append(line.strip())
    return part, "\n".join(summary_lines)


def run_step4_parallel(assembly, parts, seed, extra_args):
    # type: (str, List[str], int, List[str]) -> None
    """Run CoACD for all parts in parallel."""
    with ProcessPoolExecutor() as pool:
        futures = {
            pool.submit(_run_step4, assembly, part, seed, extra_args): part
            for part in parts
        }
        for future in as_completed(futures):
            part, summary = future.result()
            print(f"  CoACD part {part}:")
            if summary:
                for line in summary.split("\n"):
                    print(f"    {line}")


def run_step5(assembly):
    # type: (str) -> None
    """Regenerate scene URDFs."""
    subprocess.run([
        sys.executable, "fabrica/benchmark_processing/step5_generate_table_urdfs.py",
        "--assembly", assembly,
    ], capture_output=True, text=True)


def _run_step6(assembly, part, timestamp):
    # type: (str, str, str) -> Tuple[str, Optional[Tuple[float, float]]]
    """Run insertion validation. Returns (part, (pos_mm, kp_mm)) or (part, None)."""
    r = subprocess.run([
        sys.executable, "fabrica/benchmark_processing/step6_validate_insertions.py",
        "--assembly", assembly, "--part", part,
        "--method", "coacd", "--timestamp", timestamp,
    ], capture_output=True, text=True)

    final_pos = final_kp = None
    for line in r.stdout.split("\n"):
        if "Final:" in line:
            val = float(line.split("Final:")[1].strip().split()[0])
            if final_pos is None:
                final_pos = val
            else:
                final_kp = val

    if final_pos is None or final_kp is None:
        err = ""
        if r.stdout:
            err += r.stdout[-500:]
        if r.stderr:
            err += r.stderr[-500:]
        return part, None, err

    return part, (final_pos, final_kp), ""


def run_step6_parallel(assembly, parts, timestamp):
    # type: (str, List[str], str) -> Dict[str, Optional[Tuple[float, float]]]
    """Run validation for all parts in parallel. Returns {part: (pos, kp) or None}."""
    results = {}
    with ProcessPoolExecutor() as pool:
        futures = {
            pool.submit(_run_step6, assembly, part, timestamp): part
            for part in parts
        }
        for future in as_completed(futures):
            part, metrics, err = future.result()
            results[part] = metrics
            if metrics is None:
                print(f"    Part {part}: PARSE ERROR")
                if err:
                    print(err)
    return results


def main():
    parser = argparse.ArgumentParser(description="Sweep CoACD seeds until insertion validation passes")
    parser.add_argument("--assembly", required=True, help="Assembly name (e.g., beam, car)")
    parser.add_argument("--max-seeds", type=int, default=50, help="Max seeds to try (default: 50)")
    parser.add_argument("--max-pos", type=float, default=MAX_POS_MM, help="Position error threshold in mm")
    parser.add_argument("--max-kp", type=float, default=MAX_KP_MM, help="Keypoint error threshold in mm")
    parser.add_argument("--coacd-args", default="", help="Extra args to pass to step4 (e.g., '--preprocess-mode off')")
    args = parser.parse_args()

    extra_args = args.coacd_args.split() if args.coacd_args else []

    steps, parts_to_validate = load_assembly_info(args.assembly)
    print(f"Assembly: {args.assembly}")
    print(f"  Parts to decompose: {steps}")
    print(f"  Parts to validate: {parts_to_validate}")

    for seed in range(args.max_seeds):
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"SEED {seed}")
        print(sep)

        # Step 4: CoACD for all parts (parallel)
        run_step4_parallel(args.assembly, steps, seed, extra_args)

        # Step 5: Regenerate scene URDFs
        run_step5(args.assembly)

        # Step 6: Validate all non-base parts (parallel)
        ts = f"seed_{seed}"
        results = run_step6_parallel(args.assembly, parts_to_validate, ts)

        # Print results in assembly order and check pass/fail
        all_pass = True
        for part in parts_to_validate:
            metrics = results.get(part)
            if metrics is None:
                all_pass = False
                continue
            pos, kp = metrics
            passed = pos < args.max_pos and kp < args.max_kp
            status = "PASS" if passed else "FAIL"
            print(f"  Part {part}: pos={pos:.1f}mm kp={kp:.1f}mm {status}")
            if not passed:
                all_pass = False

        if all_pass:
            print(f"\n*** ALL PARTS PASS with seed {seed}! ***")
            break
    else:
        print(f"\nNo seed in range 0-{args.max_seeds - 1} passed all parts.")


if __name__ == "__main__":
    main()
