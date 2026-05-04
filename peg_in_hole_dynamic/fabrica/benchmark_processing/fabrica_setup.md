# Fabrica Assembly Setup Pipeline

Raw Fabrica meshes → training-ready assets for RL assembly.

## Key Concepts

**Assembled frame**: All parts share one coordinate system where their vertices define the fully-assembled structure. Part centroids in this frame give exact goal positions.

**Canonical frame**: Each part centered at origin, rotated so longest bbox extent aligns with X axis. This is the frame the RL policy manipulates the object in. Trajectories interpolate rotation from canonical → assembled during insertion.

**Insertion direction**: The approach vector for the final phase of the trajectory (default: top-down `[0,0,-1]`). Set per-part in `assembly_order.json`.

**inserts_into**: Maps each part to the part it physically inserts into. Controls which completed part gets CoACD collision geometry in scene URDFs (tight collision where it matters).

---

## Pipeline

```
step0_import_assets.py         →  per-part OBJ directories
step1_define_assembly_order.py →  assembly_order.json
step2_compute_canonical.py     →  canonical_transforms.json + canonical OBJs + URDFs
step3_generate_trajectories.py →  pick_place.json per part
step4_run_coacd.py             →  coacd/ convex decomposition per part
step5_generate_table_urdfs.py  →  scene URDFs per assembly step
step6_validate_insertions.py   →  IsaacGym teleport-test (GPU required)
```

All scripts support `--viz` (viser viewer on port 8082) except step6 which is always visual.

---

## Step 0: Import Assets

Copies OBJ meshes from `fabrica_copied/` into per-part directories under `assets/urdf/fabrica/`.

- **Input**: `assets/urdf/fabrica_copied/{assembly}/{part_id}.obj` — meshes in meters, assembled frame
- **Output**: `assets/urdf/fabrica/{assembly}/{part_id}/{part_id}.obj`
- **`--viz`**: shows all parts in their assembled positions (explode/assemble animation)

```bash
python fabrica/benchmark_processing/step0_import_assets.py --assemblies beam
```

## Step 1: Define Assembly Order

Interactive viser UI to specify step order, `inserts_into` mapping, and `insertion_directions`. Shows parts on table with robot for spatial context. "Animate Insertions" previews parts sliding in along their insertion directions in step order.

- **Input**: part directories from step 0, optionally existing `assembly_order.json`
- **Output**: `assets/urdf/fabrica/{assembly}/assembly_order.json`

```bash
python fabrica/benchmark_processing/step1_define_assembly_order.py --assembly beam
```

```json
{"steps": ["6","2","0","3","1"], "inserts_into": {"2":"6","0":"2","3":"6","1":"3"}}
```

- `steps`: assembly order, first part is base (placed first, nothing beneath it)
- `inserts_into`: which already-placed part each new part inserts into
- `insertion_directions`: (optional) per-part approach vector, default `[0,0,-1]`

## Step 2: Compute Canonical Transforms

For each part: loads the assembled-frame mesh, computes centroid and bbox, finds rotation that aligns longest extent to X. Generates the canonical mesh (centered + rotated) and per-part URDFs (V-HACD and SDF variants).

- **Input**: `assets/urdf/fabrica_copied/{assembly}/{part_id}/{part_id}.obj`
- **Output**:
  - `canonical_transforms.json` — per-part `assembled_to_canonical_wxyz` quaternion, `original_centroid`, `canonical_extents`
  - `{part_id}_canonical.obj` — mesh in canonical frame
  - `{assembly}_{part_id}.urdf`, `{assembly}_{part_id}_sdf.urdf` — per-part collision URDFs
- **`--viz`**: two rows of individual parts with axes — top row in assembled frame, bottom row in canonical frame

```bash
python fabrica/benchmark_processing/step2_compute_canonical.py --assembly beam
```

The `original_centroid` is used later to compute goal positions (centroid + table offset = where the part sits in the scene). The quaternion's inverse gives the end rotation for trajectories.

## Step 3: Generate Trajectories

For each part in assembly order, generates a crane-style trajectory with three phases:
1. **Lift**: straight up from start pose to clearance height (0.68m), rotation unchanged
2. **Transit**: horizontal move toward goal + slerp from canonical rotation to assembled rotation
3. **Approach**: translate along insertion direction to final pose, rotation fixed

Start positions are computed by bin-packing parts into rows on the table by their X width, centered around `START_XY`. End positions come from `original_centroid` + table offset.

- **Input**: `canonical_transforms.json`, `assembly_order.json`, canonical OBJs (for bbox-based layout)
- **Output**: `assets/urdf/fabrica/{assembly}/trajectories/{part_id}/pick_place.json`
- **`--viz`**: animated sequential trajectory playback with robot and table

```bash
python fabrica/benchmark_processing/step3_generate_trajectories.py --assembly beam
```

Output format (quaternions are **xyzw**):
```json
{"start_pose": [x,y,z, qx,qy,qz,qw], "goals": [[x,y,z, qx,qy,qz,qw], ...]}
```

## Step 4: Run CoACD

Decomposes each part's canonical mesh into convex hulls using CoACD. Generates per-hull OBJ files and a multi-body URDF. Uses tyro args (dataclass-based CLI).

- **Input**: `{part_id}_canonical.obj`
- **Output**: `assets/urdf/fabrica/{assembly}/{part_id}/coacd/` with `decomp_*.obj` + `{assembly}_{part_id}_coacd.urdf`
- **`--viz`**: original mesh (semi-transparent) with colored hull overlay

```bash
for part in 0 1 2 3 6; do
    python fabrica/benchmark_processing/step4_run_coacd.py --assembly beam --part $part
done
```

Reports max hull overshoot (Hausdorff distance from hull surfaces to original mesh) — this directly predicts insertion collision errors.

**Tuning**: The insertion collision error for a part is primarily determined by the CoACD quality of its **insertion target** (the part it `inserts_into`), not its own decomposition. Tune the target part first.

| Param | Default | Tuned | Effect |
|-------|---------|-------|--------|
| `--preprocess-resolution` | 50 | 150 | Preserves narrow channels in manifold repair |
| `--resolution` | 3000 | 16000 | Tighter Hausdorff fit per hull |
| `--threshold` | 0.03 | 0.03 | Concavity threshold (lower = more hulls) |

Example for beam part 6 (insertion target for parts 2 and 3):
```bash
python fabrica/benchmark_processing/step4_run_coacd.py \
    --assembly beam --part 6 --preprocess-resolution 150 --resolution 16000
```

See `collision_comparison_summary.md` for detailed parameter comparison.

## Step 5: Generate Scene URDFs

For each assembly step, generates a scene URDF containing the table + all previously-completed parts at their goal poses as fixed collision geometry. Three variants per step:

| File | Method | How completed parts appear |
|------|--------|---------------------------|
| `scene.urdf` | V-HACD | Raw mesh (IsaacGym auto-decomposes) |
| `scene_sdf.urdf` | SDF | Raw mesh + `<sdf>` tag |
| `scene_coacd.urdf` | CoACD | Insertion target uses pre-decomposed hulls, others raw mesh |

- **Input**: trajectories (goal poses), canonical meshes, CoACD decomps, `assembly_order.json`
- **Output**: `assets/urdf/fabrica/{assembly}/environments/{part_id}/` with three URDFs per step
- **`--viz`**: renders scene with collision method toggle (coacd/sdf/vhacd), shows start/goal poses and trajectory waypoints

```bash
python fabrica/benchmark_processing/step5_generate_table_urdfs.py --assembly beam
```

## Step 6: Validate Insertions

Teleports each part along its trajectory in IsaacGym and measures position/keypoint error. Requires GPU.

- **Input**: scene URDFs, trajectories, part URDFs
- **Output**: per-step collision metrics + video

```bash
TS=$(date +%Y-%m-%d_%H-%M-%S)
for part in 0 1 2 3 6; do
    python fabrica/benchmark_processing/step6_validate_insertions.py \
        --assembly beam --part $part --method coacd --timestamp $TS &
done; wait
```

**Pass**: final position error < 3mm and final keypoint error < 5mm for all parts. A constant ~2mm baseline is normal (physics `contact_offset`).

**If a part fails**: the collision error is usually caused by the CoACD decomposition of its **insertion target**, not its own. Check `inserts_into` to find the target, re-run step 4 for that part with tuned params, then redo steps 5-6. Changing any part's decomposition affects all parts that share a scene with it, so always re-validate all parts after any CoACD change.

---

## Output Structure

```
assets/urdf/fabrica/{assembly}/
├── assembly_order.json
├── canonical_transforms.json
├── {part_id}/
│   ├── {part_id}.obj                    # assembled-frame mesh
│   ├── {part_id}_canonical.obj          # canonical-frame mesh
│   ├── {assembly}_{part_id}.urdf        # V-HACD collision
│   ├── {assembly}_{part_id}_sdf.urdf    # SDF collision
│   └── coacd/                           # CoACD decomposition
│       ├── decomp_*.obj
│       └── {assembly}_{part_id}_coacd.urdf
├── trajectories/{part_id}/
│   └── pick_place.json
└── environments/{part_id}/
    ├── scene.urdf                       # table + completed parts (V-HACD)
    ├── scene_sdf.urdf                   # table + completed parts (SDF)
    └── scene_coacd.urdf                 # table + completed parts (CoACD for target)
```

---

## Per-Assembly Status

| Assembly | Parts | Status |
|----------|-------|--------|
| beam | 0,1,2,3,6 | done |
| car | 0-5 | TODO |
| cooling_manifold | 0-6 | TODO |
| duct | 0-7 | TODO |
| gamepad | 0-5 | TODO |
| plumbers_block | 0-4 | TODO |
| stool_circular | 0-8 | TODO |
