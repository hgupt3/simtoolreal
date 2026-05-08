"""Register FurnitureBench Problem entries.

Currently registers sparse and dense ``furniture_bench.one_leg`` variants
(insert square_table_leg4 into one corner of the square_table_top). The
receptive URDF + per-task setup metadata are produced by
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
ONE_LEG_THREAD_PITCH_M = 0.00937368684342171


def _one_leg_screw_insert_waypoints(final_pose, turns):
    """Return lead-in + screw waypoints at the requested turns + final pose."""
    final_pose = tuple(float(v) for v in final_pose)
    final_pos = np.asarray(final_pose[:3], dtype=float)
    final_quat = np.asarray(final_pose[3:7], dtype=float)
    final_rot = R.from_quat(final_quat)
    insertion_dir = np.asarray((0.0, 0.0, -1.0), dtype=float)

    waypoints = []

    # Lead-in: 25 mm above final with the same orientation. This preserves the
    # existing one-leg pre-insert heuristic while keeping it explicit in the
    # problem's subgoal list.
    lead_pos = final_pos - insertion_dir * ONE_LEG_PRE_INSERT_OFFSET_M
    waypoints.append((*lead_pos.tolist(), *final_quat.tolist()))

    # Screw guidance. Full-turn waypoints are intentionally the same SO(3)
    # orientation as final; their ordering/z offset carries phase.
    for turn in turns:
        backoff = float(turn) * ONE_LEG_THREAD_PITCH_M
        pos = final_pos - insertion_dir * backoff
        quat = (R.from_euler("z", 360.0 * float(turn), degrees=True) * final_rot).as_quat()
        if float(np.dot(quat, final_quat)) < 0.0:
            quat = -quat
        waypoints.append((*pos.tolist(), *quat.tolist()))

    waypoints.append(final_pose)
    return tuple(waypoints)


def _one_leg_dense_insert_waypoints(final_pose):
    """Return lead-in + two-turn screw waypoints + final assembled pose."""
    return _one_leg_screw_insert_waypoints(final_pose, turns=(2.0, 1.5, 1.0, 0.5))


def _one_leg_semi_dense_insert_waypoints(final_pose):
    """Return lead-in + one-turn screw waypoints + final assembled pose."""
    return _one_leg_screw_insert_waypoints(final_pose, turns=(1.0, 0.5))


def _register_problem_variant(
    *,
    name: str,
    insertion_object_name: str,
    receptive_urdf: str,
    insert_poses,
    hole_z_offset: float,
) -> None:
    PROBLEM_REGISTRY[name] = Problem(
        name=name,
        insertion_object_name=insertion_object_name,
        receptive_urdf=receptive_urdf,
        insert_pose_rel_receptive=insert_poses,
        hole_z_offset=hole_z_offset,
        # One 25 mm thread/tenon length: with insertion_direction=(0,0,-1),
        # the pre-insert pose starts with the threaded tip at the hole entrance.
        pre_insert_offset=ONE_LEG_PRE_INSERT_OFFSET_M,
    )


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

    final_pose = (
        float(pos_canonical[0]), float(pos_canonical[1]), float(pos_canonical[2]),
        qx, qy, qz, qw,
    )
    dense_waypoints = _one_leg_dense_insert_waypoints(final_pose)
    semi_dense_waypoints = _one_leg_semi_dense_insert_waypoints(final_pose)
    sparse_waypoints = (dense_waypoints[0], dense_waypoints[-1])

    base_name = "furniture_bench.one_leg"
    for variant_name, insert_poses in (
        (base_name, dense_waypoints),
        (f"{base_name}_dense", dense_waypoints),
        (f"{base_name}_semi_dense", semi_dense_waypoints),
        (f"{base_name}_sparse", sparse_waypoints),
    ):
        _register_problem_variant(
            name=variant_name,
            insertion_object_name=inserter_key,
            receptive_urdf=recv_rel,
            insert_poses=insert_poses,
            hole_z_offset=hole_z_offset,
        )

    hybrid_recv = (
        _ASSETS_FB / "square_table" / "insertion_fixtures"
        / "one_leg_sdf_hybrid.urdf"
    )
    hybrid_key = f"furniture_bench_{child_part}_sdf_hybrid"
    if hybrid_recv.is_file() and hybrid_key in NAME_TO_OBJECT:
        hybrid_base_name = f"{base_name}_sdf_hybrid"
        hybrid_recv_rel = hybrid_recv.relative_to(_REPO_ROOT / "assets").as_posix()
        for variant_name, insert_poses in (
            (hybrid_base_name, dense_waypoints),
            (f"{hybrid_base_name}_dense", dense_waypoints),
            (f"{hybrid_base_name}_semi_dense", semi_dense_waypoints),
            (f"{hybrid_base_name}_sparse", sparse_waypoints),
        ):
            _register_problem_variant(
                name=variant_name,
                insertion_object_name=hybrid_key,
                receptive_urdf=hybrid_recv_rel,
                insert_poses=insert_poses,
                hole_z_offset=hole_z_offset,
            )


_register_one_leg()
