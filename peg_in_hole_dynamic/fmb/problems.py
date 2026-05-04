"""Register FMB Problem entries.

Minimal scope (per plan): one canonical entry, ``fmb.board_1.peg_1`` —
the asset is fmb_board_1's part_id ``board_1_1``, inserted into the board
fixture (board_1_0 + table). Broader discovery is deferred.

Pose math mirrors fabrica.problems:
``assembled_to_canonical_wxyz`` rotates canonical -> assembled, so the
actor's pose in the fixture frame is the inverse of that quaternion,
with translation = the part's ``original_centroid``.
"""

import json
import logging
from pathlib import Path

from peg_in_hole_dynamic import PROBLEM_REGISTRY, Problem

_LOG = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FMB_DIR = _REPO_ROOT / "assets" / "urdf" / "fmb"


def _pose_from_canonical_transforms(transforms: dict, part_id: str):
    entry = transforms[part_id]
    cx, cy, cz = entry["original_centroid"]
    w, x, y, z = entry["assembled_to_canonical_wxyz"]
    qx, qy, qz, qw = -x, -y, -z, w
    return (float(cx), float(cy), float(cz), float(qx), float(qy), float(qz), float(qw))


def _register_board_1_peg_1() -> None:
    assembly = "fmb_board_1"
    part_id = "board_1_1"  # named "peg_1" in the registry per user convention
    transforms_path = _FMB_DIR / assembly / "canonical_transforms.json"
    receptive_rel = f"urdf/fmb/{assembly}/insertion_scenes/{part_id}/scene_0000.urdf"
    receptive_abs = _REPO_ROOT / "assets" / receptive_rel

    if not transforms_path.is_file():
        _LOG.warning("fmb.board_1.peg_1: missing %s — skipping", transforms_path)
        return
    if not receptive_abs.is_file():
        _LOG.warning("fmb.board_1.peg_1: missing %s — skipping", receptive_abs)
        return

    with open(transforms_path) as f:
        transforms = json.load(f)

    pose = _pose_from_canonical_transforms(transforms, part_id)
    name = "fmb.board_1.peg_1"
    PROBLEM_REGISTRY[name] = Problem(
        name=name,
        insertion_object_name=f"{assembly}_{part_id}_coacd",
        receptive_urdf=receptive_rel,
        insert_pose_rel_receptive=pose,
    )


_register_board_1_peg_1()
