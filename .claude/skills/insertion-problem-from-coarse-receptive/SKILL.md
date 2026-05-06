---
name: insertion-problem-from-coarse-receptive
description: Set up new PegInHoleDynamicEnv insertion Problems with a fine CoACD only around the active hole and coarse `<box>` primitives for the rest of the receptive — including FMB peg-in-hole boards and fabrica-style multi-part fixtures.
---

# Insertion Problem from Coarse Receptive

This skill walks through how we built the `fmb.peg_board_1.peg_*` and
`fabrica.beam_2x.part_*` Problems in `peg_in_hole_dynamic/`. The recipe
is the same in both cases: **fine CoACD only where the inserter actually
contacts geometry; coarse axis-aligned boxes for everything else.**

Use this whenever you need to:
- Add a new peg-in-hole-board Problem (FMB-style flat plate with multiple holes).
- Add a new multi-part-fixture Problem (fabrica-style assembly where one
  pre-placed part receives the inserter).
- Or convert an existing fully-CoACD'd receptive into the coarse style for
  faster sim / smaller URDFs.

## The big picture

A receptive URDF has two parts:
1. **Active-hole geometry** — the few CoACD hulls (or sliced mesh patch) that
   actually touch the inserter during insertion. These need fine collision
   fidelity. Always use CoACD here, never V-HACD (Isaac Gym's V-HACD is bad).
2. **Bulk** — everything else on the board / fixture. The peg never reaches
   here, so a single axis-aligned `<box>` per part (or a 4-box "frame"
   around the hole on the *receiver* part) is sufficient.

A `Problem` registers an `(insertion_object_name, receptive_urdf, insert_pose_rel_receptive)`
triple. The insert pose is in the receptive's **A-frame** (assembled frame),
where the receptive URDF root sits at `(rand_x, rand_y, table_top_z + hole_z_offset)`.

## A-frame convention

For all fabrica/fmb assemblies and peg-in-hole boards we use:
- `min_z = 0` for the lowest fixture point (so loading the URDF at world
  `z = table_top_z` puts the bottom face on the table).
- XY centered at the receptive's geometric center (or at scene-frame origin
  for fabrica, where parts are already placed at `original_centroid`).

Each part's pose in A-frame is `(original_centroid_p, inverse(q_a→c_p))`,
read from `canonical_transforms.json`. The inserter's pose follows the same
formula and goes straight into the Problem as `insert_pose_rel_receptive`.

## Pipeline outline

```
step 1   manual hole/peg pairing → JSON      (only for peg-in-hole boards)
step 2   asset generation → URDFs + meshes
         per-problem receptive URDF =
            fine CoACD (hole region)
          + coarse <box> primitives (bulk)
step 3   register Object entries in NAME_TO_OBJECT
step 4   register Problem entries in PROBLEM_REGISTRY
step 5   visual verification (visualize_problems.py)
```

The split into steps is optional — for assembly-style fixtures (e.g.
fabrica.beam_2x) where the receiver/inserter pairing comes from
`assembly_order.json["inserts_into"]`, we collapse step 1 into step 2.

---

## Recipe A — peg-in-hole boards (FMB-style)

Input geometry: a flat plate with N rectangular/cylindrical holes. The
canonical example is `fmb.peg_board_1` — see
`peg_in_hole_dynamic/fmb/peg_board_problem_setup/`.

### Step 1: pair holes ↔ pegs and dial in poses (manual + viser)

```python
# step1_pair_and_visualize.py
# 1. Detect holes on the plate by clustering "horizontal-normal" triangle
#    centroids inside the board's z-slab via DBSCAN.
# 2. Auto-match holes ↔ long pegs by sum-of-squared-clearances on the
#    {0°, 90°} yawed XY bbox.  Don't use finer yaw candidates here — they
#    let star/asterisk pegs steal big square holes by rotating to a
#    smaller diagonal bbox.
# 3. Render all 9 placements in viser with per-peg dropdown + dx/dy/yaw
#    nudge sliders.  Highlight the selected peg in saturated yellow so the
#    user knows which is which.
# 4. Save (peg, hole_xy, yaw, nudge) per hole into
#    peg_board_1_assemblies.json.
```

The output JSON is the source of truth; downstream scripts read from it.

### Step 2: generate assets

```python
# step2_generate_assets.py
# Per (peg_id, hole_id) pair from the JSON:

# 2a. Slice the plate into a watertight column around the hole
#     using manifold3d boolean intersection (NOT trimesh slice_plane;
#     half-space slicing produces non-manifold output on >0% of holes).

#     box  = trimesh.creation.box(extents=hole_bbox + 5mm padding)
#     a    = manifold3d.Manifold(manifold3d.Mesh(board.vertices, board.faces))
#     b    = manifold3d.Manifold(manifold3d.Mesh(box.vertices,   box.faces))
#     mesh = (a ^ b).to_mesh()      # ^ = intersection in manifold3d

# 2b. CoACD the hole patch.  For square / hex holes set
#     max_convex_hull=20 — without the cap CoACD over-decomposes them
#     into hundreds of pieces.

# 2c. Coarse frame:
#     1 big plate AABB minus 1 hole AABB → 4 axis-aligned boxes
#     ("top", "bottom", "left", "right" of the hole).  Other holes on
#     the same board are intentionally NOT modelled — the bulk box
#     overlaps where they would have been, which is fine because each
#     receptive URDF is per-problem and only ever serves the active hole.

# 2d. Per-problem receptive URDF: a single <link name="plate"> containing
#     one <visual>/<collision> per bulk box and one per CoACD hull
#     (mesh referenced relative to the URDF's directory).

# 2e. Per-peg canonical mesh + URDF:
#     - XYZ-centered at bbox centroid.
#     - Convention: longest XY extent along X.  If the storage .obj
#       has its longer side along Y, apply 90° Z rotation when writing
#       the canonical .obj and persist the flag in canonical_meta.json.
#     - CoACD the canonical mesh (default tight threshold 0.03).
#     - Emit a single-link visual URDF and a multi-link CoACD URDF.
```

### Step 3: register Objects

```python
# fmb/objects.py (helper added once per long-peg family)
for peg_dir in (ASSETS_DIR / "pegs").iterdir():
    canonical_obj = peg_dir / f"{peg_dir.name}_canonical.obj"
    coacd_urdf    = peg_dir / "coacd" / f"{peg_dir.name}_coacd.urdf"
    if all_present:
        scale = rescale_by_factor(canonical_mesh.extents, factor=25)
        objects[f"fmb_{peg_dir.name}"]       = Object(visual_urdf,  scale, need_vhacd=False)
        objects[f"fmb_{peg_dir.name}_coacd"] = Object(coacd_urdf,   scale, need_vhacd=False)
```

### Step 4: register Problems

```python
# fmb/problems.py
# For each hole_id in <board>_assemblies.json:
#   pos.z = peg_height / 2          (canonical mesh is z-centered)
#   q     = R_x180 ∘ R_yaw_saved ∘ R_canonical_inv
#       R_canonical_inv = R_z(-90°) iff canonical_meta.json says
#       canonical_rotated_z90, else identity
#   PROBLEM_REGISTRY[f"fmb.{board}.{peg}"] = Problem(
#       insertion_object_name = f"fmb_{peg}_coacd",
#       receptive_urdf        = "urdf/fmb/boards/{board}/insertion_fixtures/{board}_{peg}.urdf",
#       insert_pose_rel_receptive = (px, py, peg_h/2, qx, qy, qz, qw),
#       hole_z_offset = 0.0,
#   )
```

---

## Recipe B — multi-part assembly fixtures (fabrica-style)

Input: an assembly described by `canonical_transforms.json` and
`assembly_order.json["inserts_into"]`. Example: `fabrica.beam_2x` — see
`peg_in_hole_dynamic/fabrica/beam_2x_problem_setup/`.

For each `inserter_id → receiver_id` pair in `inserts_into`:

```python
# step1_generate_assets.py  (single step here — pairing comes from JSON)

# 1. Compute every part's A-frame pose:
#    pos  = transforms[pid]["original_centroid"]
#    quat = R.from_quat(q_a→c_xyzw).inv().as_quat()   # = inverse(q_a→c)

# 2. Inserter mesh in A-frame = canonical mesh @ inserter pose.

# 3. Receiver-hull classification — REUSE existing CoACD output, do NOT
#    re-CoACD:
#    for hull in <receiver>/coacd/decomp_*.obj:
#        hull_A_bbox = bbox(receiver_pose @ hull.vertices)
#        if hull_A_bbox overlaps inserter_A_bbox + 5mm:
#            -> NEAR hull: emit as <mesh> with the receiver's A-frame
#               pose baked into the visual <origin>
#        else:
#            -> drop (covered by the bulk frame below)

# 4. Receiver bulk = 4-box frame around the inserter's A-frame bbox,
#    clipped to the receiver part's overall A-frame bbox.

# 5. Each non-receiver fixture part (steps[: idx(inserter)] − {receiver})
#    gets a single A-frame-aligned bbox box (one <box> primitive each).

# 6. Per-problem URDF: single <link name="plate"> with
#       <box>: receiver bulk frame (4 boxes, possibly fewer near edges)
#     + <box>: one per non-receiver fixture part
#     + <mesh>: one per near-hole receiver hull (origin = receiver A-pose,
#               filename = "../<receiver>/coacd/decomp_<i>.obj").
```

The matching `Problem` registration is unchanged from
`peg_in_hole_dynamic/fabrica/problems.py`; we just point the new URDF
at the same `insertion_fixtures/part_<inserter>.urdf` path.

---

## Required gotchas

- **Use manifold3d for slicing**, not `trimesh.intersections.slice_mesh_plane`.
  The latter produces non-watertight meshes on at least one peg_board_1 hole,
  and CoACD throws `unexpected code path was hit` on non-manifold input.
- **No V-HACD in Isaac Gym.** Always either CoACD or a single convex hull.
  V-HACD's collision is too coarse to feel right during insertion.
- **Cap CoACD hull count for hole patches** (`max_convex_hull=20`).
  Without the cap, a 45×42 mm hex hole produced 631 hulls in our tests.
- **Cache canonical-rotation flag.** Long pegs whose storage .obj has
  Y > X get rotated 90° about Z when written as canonical.  Persist the
  flag (e.g. `canonical_meta.json`) so the Problem's quaternion can
  multiply by `R_z(-90°)` to cancel it.
- **Per-problem URDFs** for boards with multiple holes — one shared URDF
  with all 9 holes' CoACD hulls + a unioned bulk box would overlap the
  active hole.  Per-problem URDFs sidestep this with a coarse box that
  covers the inactive hole regions (ignored, since the inserter never goes
  there for that problem).
- **The visualizer parser** (`peg_in_hole_dynamic/visualize_problems.py`)
  must be able to handle `<box>` primitives, single-link URDFs whose
  link is named "root", and per-visual `<origin>` tags.  If you add a
  new URDF style, sanity-check that `_parse_fixture_urdf` renders it.

## Visual verification

After every change, run the global problem viewer to confirm everything
loads and aligns:

```
.venv/bin/python -u peg_in_hole_dynamic/visualize_problems.py --port 80<NN>
```

The dropdown should now include the new Problem.  Use the opacity sliders
on the receptive + insertion to see how cleanly the inserter slots into
the hole.

## File map (current canonical implementations)

```
peg_in_hole_dynamic/
├── fabrica/
│   ├── beam_2x_problem_setup/
│   │   └── step1_generate_assets.py        # Recipe B (single-step)
│   ├── _pose_utils.py                      # write_fixture_urdf (legacy)
│   └── problems.py                         # registers Problem entries
├── fmb/
│   ├── peg_board_problem_setup/
│   │   ├── step1_pair_and_visualize.py     # Recipe A step 1
│   │   ├── step2_generate_assets.py        # Recipe A step 2
│   │   └── peg_board_1_assemblies.json     # source of truth
│   ├── objects.py                          # _load_long_peg_objects()
│   └── problems.py                         # _register_peg_board_problems()
├── visualize_problems.py                   # global verification tool
└── __init__.py                             # PROBLEM_REGISTRY + Problem
```
