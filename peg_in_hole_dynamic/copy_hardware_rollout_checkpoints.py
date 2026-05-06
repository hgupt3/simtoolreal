#!/usr/bin/env python3
"""Copy selected PegInHoleDynamic checkpoints for hardware rollouts.

Copies the latest ``last/model.pth`` from the six kept training runs into a
dated folder under ``hardware_rollouts``. Each copied checkpoint gets a small
metadata file plus the saved training config/cmd when available.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATE = datetime.now().strftime("%Y-%m-%d")
DEFAULT_DEST_NAME = f"{DEFAULT_DATE}_peg_in_hole_dynamic_checkpoints"


@dataclass(frozen=True)
class RunSpec:
    slug: str
    job_id: int
    problem: str
    run_dir: str


RUNS = (
    RunSpec(
        slug="peg_tol0p5mm",
        job_id=597842,
        problem="peg.tol0p5mm",
        run_dir=(
            "train_dir/PEG_IN_HOLE_DYNAMIC/peg_tol0p5mm_final/"
            "finetune_rgf10_2026-05-05_23-17-49"
        ),
    ),
    RunSpec(
        slug="fmb_peg_board_1_peg_46_sdf_hybrid",
        job_id=608410,
        problem="fmb.peg_board_1.peg_46_sdf_hybrid",
        run_dir=(
            "train_dir/PEG_IN_HOLE_DYNAMIC/fmb_peg_board_1_peg_46_sdf_hybrid_better_physics/"
            "finetune_rgf10_2026-05-06_01-56-32"
        ),
    ),
    RunSpec(
        slug="fabrica_beam_2x_part_2_sdf_hybrid",
        job_id=605707,
        problem="fabrica.beam_2x.part_2_sdf_hybrid",
        run_dir=(
            "train_dir/PEG_IN_HOLE_DYNAMIC/fabrica_beam_2x_part_2_sdf_hybrid_better_physics/"
            "finetune_rgf10_2026-05-06_01-37-50"
        ),
    ),
    RunSpec(
        slug="furniture_bench_one_leg_sdf_hybrid",
        job_id=614837,
        problem="furniture_bench.one_leg_sdf_hybrid",
        run_dir=(
            "train_dir/PEG_IN_HOLE_DYNAMIC/furniture_bench_one_leg_sdf_hybrid_better_physics/"
            "finetune_rgf10_2026-05-06_03-17-45"
        ),
    ),
    RunSpec(
        slug="furniture_bench_one_leg_sdf_hybrid_sparse",
        job_id=657781,
        problem="furniture_bench.one_leg_sdf_hybrid_sparse",
        run_dir=(
            "train_dir/PEG_IN_HOLE_DYNAMIC/furniture_bench_sparse_vs_dense/"
            "sparse_finetune_rgf10_2026-05-06_14-14-29"
        ),
    ),
    RunSpec(
        slug="furniture_bench_one_leg_sdf_hybrid_dense",
        job_id=657782,
        problem="furniture_bench.one_leg_sdf_hybrid_dense",
        run_dir=(
            "train_dir/PEG_IN_HOLE_DYNAMIC/furniture_bench_sparse_vs_dense/"
            "dense_finetune_rgf10_2026-05-06_14-14-35"
        ),
    ),
)

PRETRAINED_POLICY_DIR = REPO_ROOT / "pretrained_policy"


def _latest_last_checkpoint(run_dir: Path) -> Path:
    candidates = sorted(
        run_dir.glob("runs/*/last/model.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No runs/*/last/model.pth under {run_dir}")
    return candidates[0]


def _copy_if_exists(src: Path, dst: Path) -> Optional[Path]:
    if not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _copy_run(spec: RunSpec, dest_root: Path, *, dry_run: bool = False) -> dict:
    run_dir = REPO_ROOT / spec.run_dir
    checkpoint = _latest_last_checkpoint(run_dir)
    run_root = checkpoint.parents[1]
    out_dir = dest_root / spec.slug

    metadata = {
        **asdict(spec),
        "source_run_dir": str(run_dir),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_mtime": datetime.fromtimestamp(
            checkpoint.stat().st_mtime
        ).isoformat(timespec="seconds"),
        "copied_checkpoint": str(out_dir / "model.pth"),
    }

    if dry_run:
        metadata["dry_run"] = True
        return metadata

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, out_dir / "model.pth")
    _copy_if_exists(run_root / "config.yaml", out_dir / "config.yaml")
    _copy_if_exists(run_root / "cmd.txt", out_dir / "cmd.txt")
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def copy_checkpoints(dest_root: Path, specs: Iterable[RunSpec], *, dry_run: bool = False) -> list:
    if not dry_run:
        dest_root.mkdir(parents=True, exist_ok=True)
    copied = [_copy_run(spec, dest_root, dry_run=dry_run) for spec in specs]
    pretrained = _copy_pretrained_policy(dest_root, dry_run=dry_run)
    copied.append(pretrained)
    if not dry_run:
        (dest_root / "manifest.json").write_text(
            json.dumps(copied, indent=2, sort_keys=True) + "\n"
        )
    return copied


def _copy_pretrained_policy(dest_root: Path, *, dry_run: bool = False) -> dict:
    checkpoint = PRETRAINED_POLICY_DIR / "model.pth"
    config = PRETRAINED_POLICY_DIR / "config.yaml"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not config.is_file():
        raise FileNotFoundError(config)

    out_dir = dest_root / "pretrained_policy"
    metadata = {
        "slug": "pretrained_policy",
        "job_id": None,
        "problem": "pretrained_policy",
        "source_run_dir": str(PRETRAINED_POLICY_DIR),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_mtime": datetime.fromtimestamp(
            checkpoint.stat().st_mtime
        ).isoformat(timespec="seconds"),
        "copied_checkpoint": str(out_dir / "model.pth"),
    }

    if dry_run:
        metadata["dry_run"] = True
        return metadata

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, out_dir / "model.pth")
    shutil.copy2(config, out_dir / "config.yaml")
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=REPO_ROOT / "hardware_rollouts" / DEFAULT_DEST_NAME,
        help="Destination folder. Defaults to hardware_rollouts/<today>_peg_in_hole_dynamic_checkpoints.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without writing files.",
    )
    args = parser.parse_args()

    dest = args.dest
    if not dest.is_absolute():
        dest = REPO_ROOT / dest

    copied = copy_checkpoints(dest, RUNS, dry_run=args.dry_run)
    print(f"{'Would copy' if args.dry_run else 'Copied'} {len(copied)} policies to {dest}")
    for item in copied:
        print(f"- {item['slug']}: {item['source_checkpoint']}")


if __name__ == "__main__":
    main()
