"""Smoke test: how does Isaac Sim's UrdfConverter set mass/inertia/CoM?

We compare four URDF inertial specifications against hand-computed ground
truth (via trimesh) using a single watertight mesh (beam_3x/2):

  A. <inertial><density value=R/></inertial>            # density only
  B. <inertial><mass value=M/></inertial>               # mass only
  C. <inertial><mass M/><origin xyz=CoM/><inertia I/>/  # mass + inertia (L-peg style)
  D. no <inertial> block                                # absent

For each variant we run UrdfConverter, open the produced USD, and read
back the authored MassAPI values (mass, center_of_mass, diagonalInertia,
principalAxes). Tells us:

  - Does <density> actually populate mass + inertia?
  - Is the inertia tensor mesh-accurate or simplified (collision-hull, AABB)?
  - Does Isaac respect mass+inertia overrides, or recompute them?

    .venv_isaacsim/bin/python debug_differences/smoke_urdf_density_inertia.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _launch_app():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = False
    return AppLauncher(args).app, args


_app, _args = _launch_app()


def _ground_truth(mesh_path: Path, rho: float) -> dict:
    """Hand-compute the 'right answer' from the mesh + density."""
    import numpy as np
    import trimesh

    m = trimesh.load(mesh_path, force="mesh")
    if not m.is_watertight:
        m = m.convex_hull
    V = float(m.volume)
    mass = V * rho
    com = m.center_mass.copy()
    I = m.moment_inertia * rho  # at CoM

    # Eigendecomp -> principal axes + diagonal inertia (matches USD's
    # MassAPI diagonalInertia + principalAxes representation).
    w, vecs = np.linalg.eigh(I)
    return {
        "volume_m3": V,
        "mass_kg": mass,
        "com_m": com.tolist(),
        "I_full_kgm2": I.tolist(),
        "I_diag_principal_kgm2": w.tolist(),
        "I_diag_world_kgm2": [float(I[0, 0]), float(I[1, 1]), float(I[2, 2])],
        "I_offdiag_world": {
            "ixy": float(I[0, 1]), "ixz": float(I[0, 2]), "iyz": float(I[1, 2]),
        },
    }


def _write_urdf(out_path: Path, mesh_rel: str, inertial_block: str) -> None:
    urdf = f"""<?xml version="1.0"?>
<robot name="test_part">
  <link name="part">
    <visual><geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry></visual>
    <collision><geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry></collision>
    {inertial_block}
  </link>
</robot>"""
    out_path.write_text(urdf)


def _read_mass_from_usd(usd_path: str) -> dict:
    """Walk the USD; report MassAPI values on any prim that has them."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    out = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.MassAPI):
            continue
        api = UsdPhysics.MassAPI(prim)

        def _get(attr):
            return (attr.Get(), bool(attr.HasAuthoredValue())) if attr else (None, False)

        mass_v, mass_auth = _get(api.GetMassAttr())
        density_v, density_auth = _get(api.GetDensityAttr())
        com_v, com_auth = _get(api.GetCenterOfMassAttr())
        diag_v, diag_auth = _get(api.GetDiagonalInertiaAttr())
        princ_v, princ_auth = _get(api.GetPrincipalAxesAttr())
        out.append({
            "path": str(prim.GetPath()),
            "mass": (float(mass_v) if mass_v is not None else None, mass_auth),
            "density": (float(density_v) if density_v is not None else None, density_auth),
            "com": ([float(x) for x in com_v] if com_v else None, com_auth),
            "diag_inertia": ([float(x) for x in diag_v] if diag_v else None, diag_auth),
            "principal_axes_xyzw": (
                [float(princ_v.GetImaginary()[i]) for i in range(3)] + [float(princ_v.GetReal())]
                if princ_v is not None else None, princ_auth,
            ),
        })
    return out


def _convert(urdf_path: Path, work_dir: Path) -> str:
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(work_dir / urdf_path.stem),
        force_usd_conversion=True,
        fix_base=False,
        merge_fixed_joints=True,
        make_instanceable=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0, damping=0.0,
            ),
        ),
    )
    return UrdfConverter(cfg).usd_path


def _fmt_val(v_authored):
    val, authored = v_authored
    if val is None:
        return "—"
    if isinstance(val, list):
        prefix = "" if authored else "(default) "
        return prefix + ", ".join(f"{x:+.4e}" for x in val)
    prefix = "" if authored else "(default) "
    return f"{prefix}{val:+.4e}"


def main() -> None:
    MESH = REPO_ROOT / "assets/urdf/fabrica/beam_3x/2/2_canonical.obj"
    RHO = 335.2

    # Hand-computed truth
    truth = _ground_truth(MESH, RHO)
    print("=" * 78)
    print(f"GROUND TRUTH (trimesh, ρ={RHO} kg/m³)")
    print(f"  mesh: {MESH.name}, V = {truth['volume_m3']*1e6:.2f} cm³")
    print(f"  mass: {truth['mass_kg']*1000:.3f} g")
    print(f"  CoM (m): {truth['com_m']}")
    print(f"  I world-frame (kg·m²):")
    print(f"    diag: {truth['I_diag_world_kgm2']}")
    print(f"    offdiag: {truth['I_offdiag_world']}")
    print(f"  I principal (kg·m²): {truth['I_diag_principal_kgm2']}")
    print()

    M_kg = truth["mass_kg"]
    cx, cy, cz = truth["com_m"]
    Ixx, Iyy, Izz = truth["I_diag_world_kgm2"]
    Ixy = truth["I_offdiag_world"]["ixy"]
    Ixz = truth["I_offdiag_world"]["ixz"]
    Iyz = truth["I_offdiag_world"]["iyz"]

    inertial_variants = {
        "A_density_only": f'<inertial><density value="{RHO}"/></inertial>',
        "B_mass_only": f'<inertial><mass value="{M_kg:.6f}"/></inertial>',
        "C_full_Lpeg_style": (
            f'<inertial>'
            f'<mass value="{M_kg:.6f}"/>'
            f'<origin xyz="{cx:.6f} {cy:.6f} {cz:.6f}" rpy="0 0 0"/>'
            f'<inertia ixx="{Ixx:.6e}" ixy="{Ixy:.6e}" ixz="{Ixz:.6e}" '
            f'iyy="{Iyy:.6e}" iyz="{Iyz:.6e}" izz="{Izz:.6e}"/>'
            f'</inertial>'
        ),
        "D_no_inertial": "",
    }

    work = Path(tempfile.mkdtemp(prefix="smoke_density_"))
    print(f"work dir: {work}")
    try:
        for tag, block in inertial_variants.items():
            print()
            print("=" * 78)
            print(f"VARIANT  {tag}")
            print("-" * 78)
            test_dir = work / tag
            test_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(MESH, test_dir / MESH.name)
            urdf = test_dir / "test.urdf"
            _write_urdf(urdf, MESH.name, block)

            try:
                usd = _convert(urdf, test_dir)
            except Exception as e:
                print(f"  CONVERT FAILED: {e}")
                continue

            print(f"  USD: {usd}")
            entries = _read_mass_from_usd(usd)
            for e in entries:
                print(f"  prim: {e['path']}")
                print(f"    mass(kg):      {_fmt_val(e['mass'])}")
                print(f"    density:       {_fmt_val(e['density'])}")
                print(f"    com(m):        {_fmt_val(e['com'])}")
                print(f"    diag_inertia:  {_fmt_val(e['diag_inertia'])}")
                print(f"    princ_axes:    {_fmt_val(e['principal_axes_xyzw'])}")
    finally:
        print()
        print("=" * 78)
        print(f"keeping work dir for inspection: {work}")


if __name__ == "__main__":
    main()
