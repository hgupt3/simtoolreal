"""Register FMB Problem entries.

Same first-principles convention as fabrica/problems.py — verified by
inspecting board_1 mesh bounds:

  * ``<pid>.obj`` is in the **A frame**: ``min_z = 0`` for the base,
    centroid at ``original_centroid``.
  * ``<pid>_canonical.obj`` and ``coacd/decomp_*.obj`` are in the
    **canonical frame**: axis-aligned, centered, ``canonical_extents``
    matches the bounds.
  * ``assembled_to_canonical_wxyz`` rotates A-frame coords to C-frame;
    its inverse rotates a C-frame mesh into its assembled orientation.

For each ``(insertion_part, receiver)`` pair in
``assembly_order.json["inserts_into"]`` we register one Problem:

  * ``receptive_urdf`` = generated wrapper that, for every fixture part,
    inlines its CoACD hulls (canonical) under one ``<link>`` whose joint
    origin = ``(original_centroid_p, inverse(q_a→c_p))`` — i.e. each
    part's assembled pose.
  * ``insert_pose_rel_receptive`` = the inserter's A-frame pose:
    ``(original_centroid_ins, inverse(q_a→c_ins))``.
  * ``hole_z_offset = 0``: the URDF root sits on the table top by
    construction (the lowest fixture point in A-frame is z=0).

Initial scope: ``fmb_board_1`` insertions.
"""

import json
import logging
from pathlib import Path

from peg_in_hole_dynamic import PROBLEM_REGISTRY, Problem
from peg_in_hole_dynamic.fabrica._pose_utils import write_fixture_urdf
from peg_in_hole_dynamic.fabrica.problems import _assembled_pose

_LOG = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FMB_DIR = _REPO_ROOT / "assets" / "urdf" / "fmb"


def _register_assembly_problems(assembly: str) -> None:
    transforms_path = _FMB_DIR / assembly / "canonical_transforms.json"
    order_path = _FMB_DIR / assembly / "assembly_order.json"
    if not transforms_path.is_file() or not order_path.is_file():
        _LOG.info(
            "fmb.%s: missing canonical_transforms.json or assembly_order.json — skipping",
            assembly,
        )
        return

    transforms = json.loads(transforms_path.read_text())
    order = json.loads(order_path.read_text())
    inserts_into = order.get("inserts_into", {})
    steps = order.get("steps", [])
    if not inserts_into or not steps:
        return

    for inserter_id, _receiver_id in inserts_into.items():
        if inserter_id not in steps:
            _LOG.warning(
                "fmb.%s.%s: %r not in assembly_order.steps — skipping",
                assembly, inserter_id, inserter_id,
            )
            continue
        if inserter_id not in transforms:
            _LOG.warning(
                "fmb.%s.%s: missing canonical_transforms entry — skipping",
                assembly, inserter_id,
            )
            continue

        fixture_pids = steps[: steps.index(inserter_id)]
        if not fixture_pids:
            _LOG.warning(
                "fmb.%s.%s: empty fixture (no earlier steps) — skipping",
                assembly, inserter_id,
            )
            continue

        fixture_entries = []
        skip_problem = False
        for pid in fixture_pids:
            if pid not in transforms:
                _LOG.warning(
                    "fmb.%s.%s: fixture part %s missing transforms — skipping",
                    assembly, inserter_id, pid,
                )
                skip_problem = True
                break
            coacd_dir = _FMB_DIR / assembly / pid / "coacd"
            decomp_files = sorted(coacd_dir.glob("decomp_*.obj"))
            if not decomp_files:
                _LOG.warning(
                    "fmb.%s.%s: no CoACD hulls in %s — skipping",
                    assembly, inserter_id, coacd_dir,
                )
                skip_problem = True
                break
            mesh_rels = [f"../{pid}/coacd/{p.name}" for p in decomp_files]
            fixture_entries.append((pid, _assembled_pose(transforms, pid), mesh_rels))

        if skip_problem:
            continue

        # Strip the redundant assembly prefix from short labels (board_1_1
        # rather than fmb_board_1_board_1_1) so the problem name reads cleanly.
        short = inserter_id
        fixture_urdf = write_fixture_urdf(
            output_path=(
                _FMB_DIR / assembly / "insertion_fixtures" / f"{short}.urdf"
            ),
            parts=fixture_entries,
            robot_name=f"fixture_{assembly}_{short}",
        )
        receptive_rel = fixture_urdf.relative_to(_REPO_ROOT / "assets").as_posix()

        pose_ins = _assembled_pose(transforms, inserter_id)

        name = f"fmb.{assembly}.{short}"
        PROBLEM_REGISTRY[name] = Problem(
            name=name,
            insertion_object_name=f"{assembly}_{inserter_id}_coacd",
            receptive_urdf=receptive_rel,
            insert_pose_rel_receptive=pose_ins,
            hole_z_offset=0.0,
            pre_insert_offset=0.04,
        )


_register_assembly_problems("fmb_board_1")
