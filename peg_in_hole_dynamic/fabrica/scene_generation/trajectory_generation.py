"""Variable-length crane-style trajectory generation for scene caching.

Mirrors the geometric logic of
``fabrica/benchmark_processing/step3_generate_trajectories.py:generate_trajectory``
but produces a variable number of waypoints driven by a target Euclidean
spacing between consecutive waypoint centers (in meters), and exposes the
lift altitude as a parameter so the scene generator can adapt it for
collision-aware planning.

Step3 itself is intentionally not modified. Helpers are imported from step3
where possible to avoid duplication.
"""

from __future__ import annotations

import numpy as np

from fabrica.benchmark_processing.step3_generate_trajectories import slerp_wxyz


def generate_variable_trajectory(
    start_pos,
    start_quat_wxyz,
    end_pos,
    end_quat_wxyz,
    clearance_z: float,
    insertion_dir=None,
    target_spacing: float = 0.025,
):
    """Generate a crane-style trajectory with variable waypoint count.

    Phases (vertical insertion):  lift → transit → approach (descent).
    Phases (horizontal insertion): lift → transit → descent → horizontal approach.

    Parameters
    ----------
    start_pos, end_pos : (3,) array_like
        World-frame positions in meters.
    start_quat_wxyz, end_quat_wxyz : (4,) array_like
        Orientations as wxyz quaternions (matching step3's internal convention).
    clearance_z : float
        Lift altitude in meters (world frame). The lift phase rises straight up
        from ``start_pos`` to ``(start_pos.x, start_pos.y, clearance_z)``, and
        the transit phase travels to a point above ``approach_start`` at the
        same z.
    insertion_dir : (3,) array_like, optional
        Insertion direction unit vector. Defaults to [0, 0, -1] (top-down).
    target_spacing : float
        Approximate Euclidean spacing in meters between consecutive waypoint
        centers. The number of waypoints in each phase is
        ``max(1, ceil(phase_length / target_spacing))``.

    Returns
    -------
    waypoints : np.ndarray, shape (N, 7), dtype float32
        Waypoint poses in (x, y, z, qx, qy, qz, qw) order, **excluding** the
        start pose (matching the convention of ``pick_place.json``'s "goals").
    lift_transit_end : int
        Index into ``waypoints`` marking the first descent (proximity-allowed)
        waypoint. ``waypoints[:lift_transit_end]`` are the lift+transit phase
        that should be collision-checked; ``waypoints[lift_transit_end:]`` are
        the descent/approach phase that's allowed to be near placed parts.
    """
    if insertion_dir is None:
        insertion_dir = np.array([0.0, 0.0, -1.0])
    else:
        insertion_dir = np.asarray(insertion_dir, dtype=float)
        insertion_dir = insertion_dir / np.linalg.norm(insertion_dir)

    horizontal_insertion = abs(insertion_dir[2]) <= 0.01

    start_pos = np.asarray(start_pos, dtype=float)
    end_pos = np.asarray(end_pos, dtype=float)

    if not horizontal_insertion:
        approach_dist = abs((clearance_z - end_pos[2]) / insertion_dir[2])
    else:
        approach_dist = clearance_z - end_pos[2]
    approach_start = end_pos - insertion_dir * approach_dist

    lift_target = np.array([start_pos[0], start_pos[1], clearance_z])

    # Per-phase path lengths → waypoint counts.
    lift_len = float(np.linalg.norm(lift_target - start_pos))
    if horizontal_insertion:
        transit_target_xy = np.array([approach_start[0], approach_start[1], clearance_z])
        transit_len = float(np.linalg.norm(transit_target_xy - lift_target))
    else:
        transit_len = float(np.linalg.norm(approach_start - lift_target))
    approach_len = float(np.linalg.norm(end_pos - approach_start))

    n_lift = max(1, int(np.ceil(lift_len / target_spacing)))
    n_transit = max(1, int(np.ceil(transit_len / target_spacing)))
    n_approach = max(1, int(np.ceil(approach_len / target_spacing)))

    waypoints = []  # list of (pos: np.ndarray[3], quat_wxyz: list[4])

    # Lift: linear interp from start to lift_target, holding the start orientation.
    for i in range(1, n_lift + 1):
        t = i / n_lift
        pos = start_pos + t * (lift_target - start_pos)
        waypoints.append((pos, list(start_quat_wxyz)))

    if horizontal_insertion:
        # Horizontal: transit at clearance, then descent, then horizontal approach.
        transit_target = np.array([approach_start[0], approach_start[1], clearance_z])
        for i in range(1, n_transit + 1):
            t = i / n_transit
            pos = lift_target + t * (transit_target - lift_target)
            quat = slerp_wxyz(start_quat_wxyz, end_quat_wxyz, t)
            waypoints.append((pos, quat))

        n_descent = max(1, n_approach // 2)
        n_horiz = max(1, n_approach - n_descent)
        for i in range(1, n_descent + 1):
            t = i / n_descent
            pos = transit_target + t * (approach_start - transit_target)
            waypoints.append((pos, list(end_quat_wxyz)))
        for i in range(1, n_horiz + 1):
            t = i / n_horiz
            pos = approach_start + t * (end_pos - approach_start)
            waypoints.append((pos, list(end_quat_wxyz)))
    else:
        # Vertical-ish: transit (with rotation slerp), then straight approach.
        for i in range(1, n_transit + 1):
            t = i / n_transit
            pos = lift_target + t * (approach_start - lift_target)
            quat = slerp_wxyz(start_quat_wxyz, end_quat_wxyz, t)
            waypoints.append((pos, quat))
        for i in range(1, n_approach + 1):
            t = i / n_approach
            pos = approach_start + t * (end_pos - approach_start)
            waypoints.append((pos, list(end_quat_wxyz)))

    lift_transit_end = n_lift + n_transit

    # Convert to [N, 7] xyzw float32 for cache compatibility.
    out = np.zeros((len(waypoints), 7), dtype=np.float32)
    for i, (pos, qwxyz) in enumerate(waypoints):
        out[i, 0:3] = pos
        # wxyz → xyzw
        out[i, 3] = qwxyz[1]
        out[i, 4] = qwxyz[2]
        out[i, 5] = qwxyz[3]
        out[i, 6] = qwxyz[0]
    return out, lift_transit_end
