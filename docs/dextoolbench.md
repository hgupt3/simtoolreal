# DexToolBench Reference

DexToolBench is a benchmark for dexterous tool manipulation: 6 tool categories × 2 object instances × 2 tasks each. This document covers the dataset, visualization tooling, and how to create new tasks and objects. For evaluating a policy on the benchmark, see the Quick Start in the main [README](../README.md).

## Data Structure

```
# ── Full DexToolBench data structure ──────────────────────────────────────────
# {object_category: {object_name: [task_name, ...]}}
DEXTOOLBENCH_DATA_STRUCTURE: Dict[str, Dict[str, List[str]]] = {
    "hammer": {
        "claw_hammer": ["swing_down", "swing_side"],
        "mallet_hammer": ["swing_down", "swing_side"],
    },
    "marker": {
        "sharpie_marker": ["draw_smile", "write_c"],
        "staples_marker": ["draw_smile", "write_c"],
    },
    "eraser": {
        "flat_eraser": ["wipe_smile", "wipe_c"],
        "handle_eraser": ["wipe_smile", "wipe_c"],
    },
    "brush": {
        "blue_brush": ["sweep_forward", "sweep_right"],
        "red_brush": ["sweep_forward", "sweep_right"],
    },
    "spatula": {
        "flat_spatula": ["serve_plate", "flip_over"],
        "spoon_spatula": ["serve_plate", "flip_over"],
    },
    "screwdriver": {
        "long_screwdriver": ["spin_vertical", "spin_horizontal"],
        "short_screwdriver": ["spin_vertical", "spin_horizontal"],
    },
}
```

See `dextoolbench/objects.py` and `assets/urdf/dextoolbench/<object_category>/<object_name>/<object_name>.urdf` for more details about the objects.

See `dextoolbench/trajectories` for the list of task names following the directory structure `dextoolbench/trajectories/<object_category>/<object_name>/<task_name>.json`, which is the output of `dextoolbench/process_poses.py`. These `.json` files are poses specified in world frame.

## Downloading the DexToolBench Dataset

To list all available options, run:

```
python download_dextoolbench_data.py --list
```

To download the data for a specific task, run:

```
python download_dextoolbench_data.py \
--object_category hammer \
--object_name claw_hammer \
--task_name swing_down
```

To download the data for a specific object, run:

```
python download_dextoolbench_data.py \
--object_category hammer \
--object_name claw_hammer
```

To download the data for a specific category, run:

```
python download_dextoolbench_data.py \
--object_category hammer
```

To download all data, run:

```
python download_dextoolbench_data.py
```

For each task, it will download the data into the `dextoolbench/data/<object_category>/<object_name>/<task_name>/` directory with the following structure:

```
dextoolbench/data/<object_category>/<object_name>/<task_name>/
├── cam_K.txt  // Camera intrinsics
├── depth  // Depth images
├── masks  // Object masks
├── poses.json  // Object poses in robot frame
└── rgb  // RGB images
```

## Visualize 1 Demo

To visualize 1 demo:
```
python dextoolbench/visualize_demo.py \
--object_category hammer \
--object_name claw_hammer \
--task_name swing_down
```

https://github.com/user-attachments/assets/b7532984-6642-497b-a20c-4aa6ed486cf2

## Object Models

See `dextoolbench/objects.py` for the list of object models.

## Visualizing the Objects

To visualize a DexToolBench object:

```
python dextoolbench/visualize_object.py \
--urdf_path assets/urdf/dextoolbench/hammer/claw_hammer/claw_hammer.urdf 
```

To visualize all DexToolBench objects:

```
python dextoolbench/visualize_all_objects.py
```

<img width="1082" height="899" alt="image" src="https://github.com/user-attachments/assets/1d112fee-1f29-450d-87de-657895a8cab1" />

To visualize training objects:

```
python dextoolbench/generate_training_objects.py
python dextoolbench/visualize_training_objects.py
```

<img width="705" height="696" alt="image" src="https://github.com/user-attachments/assets/34f8df95-f2c5-478e-ace9-1e786ee97d7e" />

## Visualizing the Task Trajectories

To visualize a DexToolBench task trajectory:

```
python dextoolbench/visualize_task.py \
--object_category hammer \
--object_name claw_hammer \
--task_name swing_down
```

To visualize all DexToolBench task trajectories:

```
python dextoolbench/visualize_all_tasks.py
```

https://github.com/user-attachments/assets/a5e631af-9afd-4410-9273-c4eab3c48e60

## Manually Creating a Task Trajectory

To manually create a task trajectory:
```
python dextoolbench/interactive_create_task_trajectory.py \
--object_category hammer \
--object_name claw_hammer \
--task_name my_new_task
```

## Manually Adjusting the Object Models

Use this to manually adjust the position and orientation of the object's origin frame, as well as the object's scale.

```
python dextoolbench/interactive_adjust_object.py \
--mesh_path assets/urdf/dextoolbench/hammer/claw_hammer/claw_hammer.obj \
--output_dir assets/urdf/dextoolbench/hammer/new_claw_hammer
```

## Data Collection and Processing

To collect new task demonstrations from the real world, you need a ZED camera
and the [FoundationPose fork](https://github.com/kushal2000/FoundationPose)
(installed in a separate environment). The pipeline is:
record RGB-D video → extract object mesh with SAM 2 + SAM 3D → extract 6D poses
with FoundationPose → process into DexToolBench task trajectories.

See [data_collection_and_processing.md](data_collection_and_processing.md)
for the full step-by-step guide.

## Acquiring Real-World Objects

To reproduce the real-world DexToolBench setup, see
[acquiring_real_world_objects.md](acquiring_real_world_objects.md) for
links and notes for purchasing or otherwise acquiring the physical objects.
