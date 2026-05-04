"""Register FMB assembly objects in the global NAME_TO_OBJECT registry.

Mirrors fabrica/objects.py but scans assets/urdf/fmb/ for assemblies.
Auto-discovers any subdirectory containing canonical_transforms.json.
"""

import json
from pathlib import Path

from dextoolbench.objects import Object, rescale_by_factor, NAME_TO_OBJECT
from isaacgymenvs.utils.utils import get_repo_root_dir

ASSETS_DIR = get_repo_root_dir() / "assets" / "urdf" / "fmb"


def _discover_assemblies():
    if not ASSETS_DIR.exists():
        return []
    return sorted(
        d.name
        for d in ASSETS_DIR.iterdir()
        if d.is_dir() and (d / "canonical_transforms.json").exists()
    )


ALL_FMB_ASSEMBLIES = _discover_assemblies()


def _load_assembly_objects(assembly_name):
    transforms_path = ASSETS_DIR / assembly_name / "canonical_transforms.json"
    if not transforms_path.exists():
        return {}

    with open(transforms_path) as f:
        transforms = json.load(f)

    objects = {}
    for pid, data in transforms.items():
        name = f"{assembly_name}_{pid}"
        part_dir = ASSETS_DIR / assembly_name / pid
        ext = data["canonical_extents"]
        scale = rescale_by_factor(tuple(ext), factor=25)

        urdf_path = part_dir / f"{name}.urdf"
        if urdf_path.exists():
            objects[name] = Object(urdf_path=urdf_path, scale=scale, need_vhacd=True)

        sdf_urdf_path = part_dir / f"{name}_sdf.urdf"
        if sdf_urdf_path.exists():
            objects[f"{name}_sdf"] = Object(urdf_path=sdf_urdf_path, scale=scale, need_vhacd=False)

        coacd_urdf_path = part_dir / "coacd" / f"{name}_coacd.urdf"
        if coacd_urdf_path.exists():
            objects[f"{name}_coacd"] = Object(urdf_path=coacd_urdf_path, scale=scale, need_vhacd=False)

    return objects


FMB_NAME_TO_OBJECT = {}
for _assembly in ALL_FMB_ASSEMBLIES:
    FMB_NAME_TO_OBJECT.update(_load_assembly_objects(_assembly))

NAME_TO_OBJECT.update(FMB_NAME_TO_OBJECT)
