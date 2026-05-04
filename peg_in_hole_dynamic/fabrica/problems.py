"""Register fabrica Problem entries.

Minimal scope (per plan): one canonical entry, ``fabrica.beam_2x.part_2``
(beam_2x part 2 inserting into part 6). Broader auto-discovery across
all assemblies is deferred to a later pass.

The receptive URDF is the pre-baked scene fixture from
``insertion_scenes/<part>/scene_0000.urdf`` (already produced by
``scene_generation/generate_scenes.py``). The insert pose, relative to
the fixture origin, is derived from ``canonical_transforms.json``: it
matches the insertion part's assembled centroid + the inverse of its
``assembled_to_canonical`` rotation.
"""

import json
import logging
from pathlib import Path

from peg_in_hole_dynamic import PROBLEM_REGISTRY, Problem

_LOG = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FABRICA_DIR = _REPO_ROOT / "assets" / "urdf" / "fabrica"


def _pose_from_canonical_transforms(transforms: dict, part_id: str):
    """Compute insertion-part pose (xyz + xyzw quat) in the fixture frame.

    ``assembled_to_canonical_wxyz`` rotates a point in the canonical mesh
    frame to the assembled frame. The actor body frame *is* the canonical
    frame, so its orientation in the fixture (assembled) frame is the
    inverse of that quaternion.
    """
    entry = transforms[part_id]
    cx, cy, cz = entry["original_centroid"]
    w, x, y, z = entry["assembled_to_canonical_wxyz"]
    # Inverse of a unit quaternion is its conjugate; convert wxyz -> xyzw
    qx, qy, qz, qw = -x, -y, -z, w
    return (float(cx), float(cy), float(cz), float(qx), float(qy), float(qz), float(qw))


def _register_beam_2x_part_2() -> None:
    assembly = "beam_2x"
    part_id = "2"
    transforms_path = _FABRICA_DIR / assembly / "canonical_transforms.json"
    receptive_rel = f"urdf/fabrica/{assembly}/insertion_scenes/{part_id}/scene_0000.urdf"
    receptive_abs = _REPO_ROOT / "assets" / receptive_rel

    if not transforms_path.is_file():
        _LOG.warning("fabrica.beam_2x.part_2: missing %s — skipping", transforms_path)
        return
    if not receptive_abs.is_file():
        _LOG.warning("fabrica.beam_2x.part_2: missing %s — skipping", receptive_abs)
        return

    with open(transforms_path) as f:
        transforms = json.load(f)

    pose = _pose_from_canonical_transforms(transforms, part_id)
    name = "fabrica.beam_2x.part_2"
    PROBLEM_REGISTRY[name] = Problem(
        name=name,
        insertion_object_name=f"{assembly}_{part_id}_coacd",
        receptive_urdf=receptive_rel,
        insert_pose_rel_receptive=pose,
    )


_register_beam_2x_part_2()
