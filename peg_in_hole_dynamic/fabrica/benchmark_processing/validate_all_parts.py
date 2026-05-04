#!/usr/bin/env python3
"""Validate insertion for every part in an assembly (base included).

Runs step6_validate_insertions.py in parallel for each entry in
`assembly_order.json["steps"]` and reports per-part final position + keypoint
error. Exits 0 if all parts pass the thresholds, 1 otherwise.

Usage:
    python fabrica/benchmark_processing/validate_all_parts.py --assembly stool_circular
    python fabrica/benchmark_processing/validate_all_parts.py --assembly beam --max-pos 3 --max-kp 5
"""

import argparse
import datetime
import sys

from sweep_coacd_seeds import (
    MAX_POS_MM,
    MAX_KP_MM,
    load_assembly_info,
    run_step6_parallel,
)


def main():
    parser = argparse.ArgumentParser(description="Validate insertion for every part in an assembly")
    parser.add_argument("--assembly", required=True, help="Assembly name (e.g., stool_circular, beam)")
    parser.add_argument("--max-pos", type=float, default=MAX_POS_MM,
                        help=f"Final position error threshold in mm (default: {MAX_POS_MM})")
    parser.add_argument("--max-kp", type=float, default=MAX_KP_MM,
                        help=f"Final keypoint error threshold in mm (default: {MAX_KP_MM})")
    args = parser.parse_args()

    steps, _ = load_assembly_info(args.assembly)
    base = steps[0]

    timestamp = "validate_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"Assembly: {args.assembly}")
    print(f"Parts to validate: {', '.join(steps)}  (base={base}, {len(steps)} total)")
    print(f"Thresholds: pos < {args.max_pos:.2f} mm, kp < {args.max_kp:.2f} mm")
    print(f"Timestamp: {timestamp}\n")

    results = run_step6_parallel(args.assembly, steps, timestamp)

    print(f"\n{'Part':<6}{'final_pos_mm':>14}{'final_kp_mm':>14}  verdict")
    print("-" * 50)
    num_pass = 0
    for part in steps:
        metrics = results.get(part)
        tag = "  (base)" if part == base else ""
        if metrics is None:
            print(f"{part:<6}{'—':>14}{'—':>14}  PARSE ERROR{tag}")
            continue
        pos_mm, kp_mm = metrics
        passed = (pos_mm < args.max_pos) and (kp_mm < args.max_kp)
        verdict = "PASS" if passed else "FAIL"
        if passed:
            num_pass += 1
        print(f"{part:<6}{pos_mm:>14.3f}{kp_mm:>14.3f}  {verdict}{tag}")

    print(f"\nSummary: {num_pass}/{len(steps)} PASS")
    print(f"Videos: fabrica/debug_output/insertion/{args.assembly}/{timestamp}/")

    sys.exit(0 if num_pass == len(steps) else 1)


if __name__ == "__main__":
    main()
