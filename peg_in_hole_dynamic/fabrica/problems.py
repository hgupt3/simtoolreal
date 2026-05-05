"""Register fabrica Problem entries.

For each (insertion_part, receiver) pair in an assembly's
``inserts_into`` map, register a Problem from first principles:

  * ``receptive_urdf`` = a generated wrapper URDF that includes every
    fixture part's **assembled-frame mesh** (``<pid>.obj``) under an empty
    ``root`` link, with **identity joint origins**. The mesh vertices are
    already at A-frame coordinates, so no extra transform is needed.
    The URDF root therefore coincides with the assembly's A-frame origin
    (where scene-generation pinned the lowest fixture point to z=0).
  * ``insert_pose_rel_receptive`` = the inserter's pose in the A frame:
    ``(original_centroid_ins, inverse(q_a→c_ins))``. The inserter actor
    in the env is loaded from the canonical mesh; the inverse rotation
    rotates it into its assembled orientation.
  * ``hole_z_offset = 0``: the URDF root sits on the table top by
    construction.

Initial scope: all beam_2x insertions.
"""

import json
import logging
from pathlib import Path
from typing import Tuple

from scipy.spatial.transform import Rotation as R

from peg_in_hole_dynamic import PROBLEM_REGISTRY, Problem
from peg_in_hole_dynamic.fabrica._pose_utils import write_fixture_urdf

_LOG = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FABRICA_DIR = _REPO_ROOT / "assets" / "urdf" / "fabrica"

_IDENTITY_POSE: Tuple[float, float, float, float, float, float, float] = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
)


def _assembled_pose(transforms: dict, part_id: str
                    ) -> Tuple[float, float, float, float, float, float, float]:
    """Pose of ``part_id`` in the A frame, expressed as
    ``(x, y, z, qx, qy, qz, qw)`` — translation = original_centroid,
    rotation = inverse(q_a→c). When applied to the canonical mesh this
    reproduces the part's assembled position+orientation."""
    cx, cy, cz = transforms[part_id]["original_centroid"]
    qw, qx, qy, qz = transforms[part_id]["assembled_to_canonical_wxyz"]
    rot_xyzw = R.from_quat([qx, qy, qz, qw]).inv().as_quat()
    return (
        float(cx), float(cy), float(cz),
        float(rot_xyzw[0]), float(rot_xyzw[1]), float(rot_xyzw[2]), float(rot_xyzw[3]),
    )


def _register_assembly_problems(assembly: str) -> None:
    transforms_path = _FABRICA_DIR / assembly / "canonical_transforms.json"
    order_path = _FABRICA_DIR / assembly / "assembly_order.json"
    if not transforms_path.is_file() or not order_path.is_file():
        _LOG.info(
            "fabrica.%s: missing canonical_transforms.json or assembly_order.json — skipping",
            assembly,
        )
        return

    transforms = json.loads(transforms_path.read_text())
    order = json.loads(order_path.read_text())
    inserts_into = order.get("inserts_into", {})
    steps = order.get("steps", [])
    if not inserts_into or not steps:
        return

    for inserter_id, receiver_id in inserts_into.items():
        if inserter_id not in steps:
            _LOG.warning(
                "fabrica.%s.part_%s: %r not in assembly_order.steps — skipping",
                assembly, inserter_id, inserter_id,
            )
            continue
        if inserter_id not in transforms or receiver_id not in transforms:
            _LOG.warning(
                "fabrica.%s.part_%s: missing canonical_transforms entry — skipping",
                assembly, inserter_id,
            )
            continue

        # Fixture = every part already placed in the assembly before the
        # inserter step.
        fixture_pids = steps[: steps.index(inserter_id)]
        if not fixture_pids:
            _LOG.warning(
                "fabrica.%s.part_%s: empty fixture (no earlier steps) — skipping",
                assembly, inserter_id,
            )
            continue

        # Each fixture part is decomposed into its CoACD hulls (canonical
        # frame), all sharing the part's assembled pose
        # (original_centroid_p, inverse(q_a→c_p)). They go into one URDF
        # link per part so the visualizer can color each part distinctly.
        fixture_entries = []
        skip_problem = False
        for pid in fixture_pids:
            coacd_dir = _FABRICA_DIR / assembly / pid / "coacd"
            decomp_files = sorted(coacd_dir.glob("decomp_*.obj"))
            if not decomp_files:
                _LOG.warning(
                    "fabrica.%s.part_%s: no CoACD hulls in %s — skipping",
                    assembly, inserter_id, coacd_dir,
                )
                skip_problem = True
                break
            mesh_rels = [f"../{pid}/coacd/{p.name}" for p in decomp_files]
            fixture_entries.append((pid, _assembled_pose(transforms, pid), mesh_rels))

        if skip_problem:
            continue

        fixture_urdf = write_fixture_urdf(
            output_path=(
                _FABRICA_DIR / assembly / "insertion_fixtures" / f"part_{inserter_id}.urdf"
            ),
            parts=fixture_entries,
            robot_name=f"fixture_{assembly}_part_{inserter_id}",
        )
        receptive_rel = fixture_urdf.relative_to(_REPO_ROOT / "assets").as_posix()

        # Inserter pose in A frame.
        pose_ins = _assembled_pose(transforms, inserter_id)

        name = f"fabrica.{assembly}.part_{inserter_id}"
        PROBLEM_REGISTRY[name] = Problem(
            name=name,
            insertion_object_name=f"{assembly}_{inserter_id}_coacd",
            receptive_urdf=receptive_rel,
            insert_pose_rel_receptive=pose_ins,
            hole_z_offset=0.0,
            pre_insert_offset=0.025,
        )


_register_assembly_problems("beam_2x")
