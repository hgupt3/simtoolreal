# Scene Generation Pipeline

Generates diverse initial states for assembly RL training. Each **scene** is a set of start poses + trajectories (one per part) that can be loaded by the training env to randomize episode initial conditions.

## Pipeline

Run in order:

### 1. `compute_stable_rests_isaacgym.py`
Validates which axis-aligned orientations (6 face-down rotations) are physically stable for each part by dropping them in Isaac Gym. Outputs per-part `.npz` files to `assets/urdf/fabrica/{assembly}/stable_rests/`.

```bash
python fabrica/scene_generation/compute_stable_rests_isaacgym.py --assembly beam
```

### 2. `generate_scenes.py`
Samples N scenes. Per scene, per part: picks a stable rest, random yaw, random (x, y) on the table with rejection sampling against the fixture, placed parts, and other start poses (pairwise non-colliding). Generates a crane-style trajectory per part using `trajectory_generation.py`. Outputs `assets/urdf/fabrica/{assembly}/scenes.npz`.

```bash
python fabrica/scene_generation/generate_scenes.py --assembly beam --num-scenes 100
```

### 3. `trajectory_generation.py`
Module used by `generate_scenes.py`. Generates variable-length crane trajectories (lift, transit, descent) with slerp rotation interpolation and adaptive lift altitude that clears placed parts.

## Visualization

### `visualize_stable_rests.py`
Viser viewer showing all stable resting orientations per part side by side.

```bash
python fabrica/scene_generation/visualize_stable_rests.py --assembly beam --port 8085
```

### `visualize_scenes.py`
Viser viewer with a scene dropdown. Shows all part start poses on the table and animates each trajectory in assembly order.

```bash
python fabrica/scene_generation/visualize_scenes.py --assembly beam --port 8080
```

## Output Format

`scenes.npz` contains:
- `start_poses`: `[num_scenes, num_parts, 7]` float32 (xyz + xyzw quat)
- `goals`: `[num_scenes, num_parts, max_traj_len, 7]` float32 (zero-padded)
- `traj_lengths`: `[num_scenes, num_parts]` int32
