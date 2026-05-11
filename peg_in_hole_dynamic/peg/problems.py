"""Register peg-in-hole Problem entries.

One Problem per hole tolerance variant under
``assets/urdf/peg_in_hole/holes/hole_tol*/``. All variants share the same
insertion object (``peg``) and the same canonical insert pose; only the
receptive (hole) URDF differs.
"""

from pathlib import Path

from peg_in_hole_dynamic import PROBLEM_REGISTRY, Problem, make_pre_insert_sequence


# Canonical pose: peg handle center 13.6 cm above hole base, body +X
# pointing down (-90 deg about Y, xyzw quaternion).
_PEG_INSERT_POSE = (
    0.0, 0.0, 0.136,
    0.0, -0.70710678, 0.0, 0.70710678,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOLES_DIR = _REPO_ROOT / "assets" / "urdf" / "peg_in_hole" / "holes"


def _register_all() -> None:
    if not _HOLES_DIR.is_dir():
        return
    for hole_dir in sorted(_HOLES_DIR.iterdir()):
        if not hole_dir.is_dir() or not hole_dir.name.startswith("hole_tol"):
            continue
        urdf_rel = f"urdf/peg_in_hole/holes/{hole_dir.name}/{hole_dir.name}.urdf"
        if not (_REPO_ROOT / "assets" / urdf_rel).exists():
            continue
        # tag is the part after "hole_" (e.g. "tol0p5mm")
        tag = hole_dir.name[len("hole_"):]
        for prefix, object_name in (("peg", "peg"), ("Lpeg", "lpeg")):
            name = f"{prefix}.{tag}"
            PROBLEM_REGISTRY[name] = Problem(
                name=name,
                insertion_object_name=object_name,
                receptive_urdf=urdf_rel,
                insert_pose_rel_receptive=make_pre_insert_sequence(_PEG_INSERT_POSE),
            )


_register_all()
