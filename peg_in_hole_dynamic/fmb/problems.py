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
_register_assembly_problems("fmb_board_2")
_register_assembly_problems("fmb_board_3")


def _register_peg_board_problems(board_name: str) -> None:
    """Register one Problem per (long peg ↔ hole) pair in the
    `<setup>/<board_name>_assemblies.json` table.

    Convention for the inserter's pose:
      * The peg's URDF references its `<peg>_canonical.obj`, which is
        XYZ-centred at the bbox centroid with the longest XY extent along
        X (a 90° Z rotation may have been baked in by step 2 to enforce
        this; canonical_meta.json records it).
      * `pos.z = peg_height / 2` so applying a 180°X flip in the URDF pose
        lands the peg's tip at A-frame z=0 (board bottom).
      * Quaternion = R_x180 ∘ R_yaw_saved ∘ R_canonical_inverse. The
        saved yaw was authored against the *storage*-frame .obj in step 1;
        right-multiplying by R_canonical_inverse cancels the canonical
        rotation so the visual result matches what the user dialled in.
    """
    from scipy.spatial.transform import Rotation as R

    setup_dir = Path(__file__).resolve().parent / "peg_board_problem_setup"
    json_path = setup_dir / f"{board_name}_assemblies.json"
    if not json_path.is_file():
        _LOG.info("fmb.%s: no %s — skipping", board_name, json_path.name)
        return

    data = json.loads(json_path.read_text())
    pegs_dir = _FMB_DIR / "pegs"

    for hole_id, info in sorted(data.items()):
        peg_name      = info["peg"]
        meta_path     = pegs_dir / peg_name / "canonical_meta.json"
        canonical_obj = pegs_dir / peg_name / f"{peg_name}_canonical.obj"

        # Per-problem receptive URDF: only the active hole's CoACD + a
        # coarse 4-box frame around it (other holes ignored).
        receptive_rel = (
            f"urdf/fmb/boards/{board_name}/insertion_fixtures/{board_name}_{peg_name}.urdf"
        )
        if not (_REPO_ROOT / "assets" / receptive_rel).is_file():
            _LOG.warning("fmb.%s.%s: missing receptive URDF %s — skipping",
                         board_name, hole_id, receptive_rel)
            continue
        if not meta_path.is_file() or not canonical_obj.is_file():
            _LOG.warning("fmb.%s.%s: missing canonical assets for %s — skipping",
                         board_name, hole_id, peg_name)
            continue

        rotated_z90 = bool(json.loads(meta_path.read_text()).get(
            "canonical_rotated_z90", False))

        # Peg height in A-frame = canonical mesh's Z extent (≈ 0.15 for long pegs)
        import trimesh
        canonical = trimesh.load_mesh(str(canonical_obj), process=False)
        peg_h = float(canonical.bounds[1, 2] - canonical.bounds[0, 2])

        cx, cy   = info["hole_xy_A"]
        ndx, ndy = info["nudge_mm"]
        pos = (
            float(cx + ndx * 1e-3),
            float(cy + ndy * 1e-3),
            peg_h / 2.0,
        )

        R_x180 = R.from_euler("x", 180, degrees=True)
        R_yaw  = R.from_euler("z", float(info["yaw_deg"]), degrees=True)
        R_can_inv = (R.from_euler("z", -90, degrees=True) if rotated_z90 else R.identity())
        q = R_x180 * R_yaw * R_can_inv
        qx, qy, qz, qw = (float(v) for v in q.as_quat())

        name = f"fmb.{board_name}.{peg_name}"
        PROBLEM_REGISTRY[name] = Problem(
            name=name,
            insertion_object_name=f"fmb_{peg_name}_coacd",
            receptive_urdf=receptive_rel,
            insert_pose_rel_receptive=(*pos, qx, qy, qz, qw),
            hole_z_offset=0.0,
            pre_insert_offset=0.04,
        )


_register_peg_board_problems("peg_board_1")
