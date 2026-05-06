"""Register FurnitureBench Problem entries.

Currently registers ``furniture_bench.one_leg`` (insert square_table_leg4
into one corner of the square_table_top). The receptive URDF + per-task
setup metadata are produced by
``one_leg_problem_setup/step2_generate_assets.py``.

Pose composition (see step2 for the canonical conventions):
  * Receivers: ``R_storage_to_canonical = R_x(+90°)``. Loaded at world
    origin with identity URDF rotation; top surface at world +Z.
  * Inserters: ``R_storage_to_canonical = R_y(+90°) ∘ R_x(+90°)``.
    Long axis on canonical X, tenon at canonical -X.

For the inserter URDF pose relative to the receiver:
  * pos_canonical = R_x(+90°) ∘ pos_yup
  * q_urdf = R_x(+90°) ∘ R_yup_relative ∘ R_storage_to_canonical_inserter^-1
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from dextoolbench.objects import NAME_TO_OBJECT
from peg_in_hole_dynamic import PROBLEM_REGISTRY, Problem

_LOG = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ASSETS_FB = _REPO_ROOT / "assets" / "urdf" / "furniture_bench"

R_X90 = R.from_euler("x", 90, degrees=True)
ONE_LEG_PRE_INSERT_OFFSET_M = 0.025


def _register_one_leg() -> None:
    setup_path = _ASSETS_FB / "square_table" / "one_leg_setup.json"
    assembly_path = _ASSETS_FB / "square_table" / "assembly.json"
    if not setup_path.is_file() or not assembly_path.is_file():
        _LOG.info("furniture_bench.one_leg: missing setup/assembly JSON — skipping")
        return

    setup = json.loads(setup_path.read_text())
    assembly = json.loads(assembly_path.read_text())

    # Inserter Object key (registered by objects.py).
    child_part = setup["child_part"]
    inserter_key = f"furniture_bench_{child_part}_coacd"
    if inserter_key not in NAME_TO_OBJECT:
        _LOG.warning("furniture_bench.one_leg: %s not in NAME_TO_OBJECT — skipping",
                     inserter_key)
        return

    # Read the inserter's R_storage_to_canonical from canonical_meta.json so
    # we never drift out of sync with what step2 actually applied.
    meta_path = _ASSETS_FB / "square_table" / child_part / "canonical_meta.json"
    if not meta_path.is_file():
        _LOG.warning("furniture_bench.one_leg: missing %s — skipping", meta_path)
        return
    meta = json.loads(meta_path.read_text())
    P_inserter = np.asarray(meta["R_storage_to_canonical"], dtype=float)

    # Receptive URDF path (relative to assets/).
    recv_rel = setup["receptive_urdf"]

    # Step2 already applied the OBJ-origin → bbox-centroid offset
    # correction and rotated to canonical frame, so just use it.
    if "pos_canonical_centroid" not in setup:
        _LOG.warning(
            "furniture_bench.one_leg: setup JSON predates the centroid-offset "
            "fix — re-run step2_generate_assets.py. Skipping.",
        )
        return
    pos_canonical = np.asarray(setup["pos_canonical_centroid"], dtype=float)
    rpy_yup = np.asarray(setup["active_corner_rpy_yup"], dtype=float)

    # q_urdf = R_x(+90°) ∘ R_yup_relative ∘ Pᵀ_inserter.
    R_yup_rel = R.from_euler("xyz", rpy_yup)
    q = R_X90 * R_yup_rel * R.from_matrix(P_inserter.T)
    qx, qy, qz, qw = (float(v) for v in q.as_quat())

    # hole_z_offset: the receiver's bottom face should sit on the table
    # surface in the env. The receiver's canonical mesh is centered, so
    # half its Z extent puts the bottom face at z=0 in world.
    # square_table_top z extent ≈ 31.3 mm; offset = 0.0156 m.
    import trimesh
    top_canonical = trimesh.load(
        str(_ASSETS_FB / "square_table" / setup["parent_part"]
            / f"{setup['parent_part']}_canonical.obj"),
        force="mesh", process=False,
    )
    hole_z_offset = float((top_canonical.bounds[1, 2] - top_canonical.bounds[0, 2]) / 2)

    name = "furniture_bench.one_leg"
    PROBLEM_REGISTRY[name] = Problem(
        name=name,
        insertion_object_name=inserter_key,
        receptive_urdf=recv_rel,
        insert_pose_rel_receptive=(
            float(pos_canonical[0]), float(pos_canonical[1]), float(pos_canonical[2]),
            qx, qy, qz, qw,
        ),
        hole_z_offset=hole_z_offset,
        # One 25 mm thread/tenon length: with insertion_direction=(0,0,-1),
        # the pre-insert pose starts with the threaded tip at the hole entrance.
        pre_insert_offset=ONE_LEG_PRE_INSERT_OFFSET_M,
    )

    hybrid_recv = (
        _ASSETS_FB / "square_table" / "insertion_fixtures"
        / "one_leg_sdf_hybrid.urdf"
    )
    hybrid_key = f"furniture_bench_{child_part}_sdf_hybrid"
    if hybrid_recv.is_file() and hybrid_key in NAME_TO_OBJECT:
        hybrid_name = f"{name}_sdf_hybrid"
        PROBLEM_REGISTRY[hybrid_name] = Problem(
            name=hybrid_name,
            insertion_object_name=hybrid_key,
            receptive_urdf=hybrid_recv.relative_to(_REPO_ROOT / "assets").as_posix(),
            insert_pose_rel_receptive=(
                float(pos_canonical[0]), float(pos_canonical[1]), float(pos_canonical[2]),
                qx, qy, qz, qw,
            ),
            hole_z_offset=hole_z_offset,
            pre_insert_offset=ONE_LEG_PRE_INSERT_OFFSET_M,
        )


_register_one_leg()
