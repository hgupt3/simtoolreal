"""Register the peg in the global NAME_TO_OBJECT registry.

Mirrors fabrica/objects.py. Imported by peg_eval.py so that FabricaEnv
can resolve "peg" via NAME_TO_OBJECT[object_name].urdf_path.
"""

from dextoolbench.objects import Object, rescale_by_factor, NAME_TO_OBJECT

from isaacgymenvs.utils.utils import get_repo_root_dir


ASSETS_DIR = get_repo_root_dir() / "assets" / "urdf" / "peg_in_hole"
FMB_ASSETS_DIR = get_repo_root_dir() / "assets" / "urdf" / "fmb"

PEG_NAME_TO_OBJECT = {
    "peg": Object(
        urdf_path=ASSETS_DIR / "peg" / "peg.urdf",
        scale=rescale_by_factor((0.25, 0.03, 0.02), factor=25),
        need_vhacd=False,
    ),
}

# FMB pegs (registered if canonical mesh + CoACD URDF exist)
_FMB_PEGS_DIR = FMB_ASSETS_DIR / "pegs"
if _FMB_PEGS_DIR.exists():
    for _peg_dir in sorted(_FMB_PEGS_DIR.iterdir()):
        _name = _peg_dir.name
        _coacd_urdf = _peg_dir / "coacd" / f"{_name}_coacd.urdf"
        _sdf_urdf = _peg_dir / f"{_name}_sdf.urdf"
        _canonical = _peg_dir / f"{_name}_canonical.obj"
        if _coacd_urdf.exists() and _canonical.exists():
            import trimesh as _tm
            _mesh = _tm.load(str(_canonical), force="mesh")
            _ext = tuple(float(x) for x in _mesh.extents)
            PEG_NAME_TO_OBJECT[_name] = Object(
                urdf_path=_coacd_urdf,
                scale=rescale_by_factor(_ext, factor=25),
                need_vhacd=False,
            )
            if _sdf_urdf.exists():
                PEG_NAME_TO_OBJECT[f"{_name}_sdf"] = Object(
                    urdf_path=_sdf_urdf,
                    scale=rescale_by_factor(_ext, factor=25),
                    need_vhacd=False,
                )

NAME_TO_OBJECT.update(PEG_NAME_TO_OBJECT)
