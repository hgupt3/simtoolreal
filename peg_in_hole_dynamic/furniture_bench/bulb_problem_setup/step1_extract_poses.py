"""Step 1 of bulb-into-base problem-setup pipeline.

Regex-parse the upstream ``lamp.py`` snapshot in ``./upstream_snapshots/``
and emit ``assets/urdf/furniture_bench/lamp/assembly.json`` containing the
part list, assembled-relative poses (y-up, verbatim from upstream), and the
``bulb_screw`` task definition (bulb-into-base only — the hood join is
ignored here).

Run:
    .venv/bin/python -m peg_in_hole_dynamic.furniture_bench.bulb_problem_setup.step1_extract_poses
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAP_DIR = Path(__file__).resolve().parent / "upstream_snapshots"
OUT_PATH = REPO_ROOT / "assets" / "urdf" / "furniture_bench" / "lamp" / "assembly.json"


_PARTS_RE = re.compile(
    r"""\b\w+\(\s*furniture_conf\[["'](?P<name>\w+)["']\]\s*,\s*(?P<idx>\d+)\s*\)"""
)
_RELPOSE_LHS_RE = re.compile(
    r"""self\.assembled_rel_poses\[\(\s*(?P<p>\d+)\s*,\s*(?P<c>\d+)\s*\)\]\s*=\s*\["""
)
_RELPOSE_ALIAS_RE = re.compile(
    r"""self\.assembled_rel_poses\[\(\s*(?P<p2>\d+)\s*,\s*(?P<c2>\d+)\s*\)\]\s*=\s*self\.assembled_rel_poses\[\(\s*(?P<p1>\d+)\s*,\s*(?P<c1>\d+)\s*\)\]"""
)
_GET_MAT_RE = re.compile(
    r"""get_mat\(\s*\[(?P<pos>[-\d\.,\s]+)\]\s*,\s*\[(?P<rpy>[-\d\.,\s\w\*\(\)/\+]+?)\]\s*\)"""
)
_SHOULD_RE = re.compile(
    r"""self\.should_be_assembled\s*=\s*\[(?P<body>[^\]]*)\]""", re.DOTALL,
)
_PAIR_RE = re.compile(r"""\(\s*(\d+)\s*,\s*(\d+)\s*\)""")


def _find_matching_close(text: str, open_idx: int) -> int:
    assert text[open_idx] == "["
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"unmatched [ at index {open_idx}")


def _parse_get_mat(call_text: str):
    m = _GET_MAT_RE.search(call_text)
    if not m:
        raise ValueError(f"no get_mat in: {call_text!r}")
    pos = [float(x) for x in m.group("pos").split(",") if x.strip()]
    rpy_raw = "[" + m.group("rpy") + "]"
    import math
    import numpy as np  # noqa: F401  (used by eval)
    pi = math.pi
    np_pi = np.pi
    rpy = eval(rpy_raw, {"np": np, "math": math, "pi": pi})
    rpy = [float(v) for v in rpy]
    if len(pos) != 3 or len(rpy) != 3:
        raise ValueError(f"pos/rpy not length 3: {pos} {rpy}")
    return {"pos": pos, "rpy_xyz": rpy}


def _parse_pose_list(body: str):
    poses = []
    for m in re.finditer(
        r"""get_mat\(\s*\[[^\]]*\]\s*,\s*\[[^\]]*\]\s*\)""", body
    ):
        poses.append(_parse_get_mat(m.group(0)))
    if not poses:
        raise ValueError(f"no get_mat calls in body: {body!r}")
    return poses


def _parse_lamp(src: str):
    parts_seen = []
    for m in _PARTS_RE.finditer(src):
        parts_seen.append({"idx": int(m.group("idx")), "name": m.group("name")})
    parts_seen.sort(key=lambda d: d["idx"])

    rel_poses = {}
    for m in _RELPOSE_LHS_RE.finditer(src):
        open_idx = m.end() - 1
        close_idx = _find_matching_close(src, open_idx)
        body = src[open_idx + 1: close_idx]
        key = f"{m.group('p')}->{m.group('c')}"
        rel_poses[key] = _parse_pose_list(body)

    for m in _RELPOSE_ALIAS_RE.finditer(src):
        src_key = f"{m.group('p1')}->{m.group('c1')}"
        dst_key = f"{m.group('p2')}->{m.group('c2')}"
        if src_key not in rel_poses:
            raise ValueError(f"alias {dst_key} -> {src_key} but src not parsed")
        rel_poses[dst_key] = rel_poses[src_key]

    should = []
    m = _SHOULD_RE.search(src)
    if m:
        for pm in _PAIR_RE.finditer(m.group("body")):
            should.append([int(pm.group(1)), int(pm.group(2))])

    return {
        "parts": parts_seen,
        "assembled_rel_poses_yup": rel_poses,
        "should_be_assembled": should,
    }


def main():
    src = (SNAP_DIR / "lamp.py").read_text()
    base = _parse_lamp(src)

    # bulb_screw task = just the bulb-into-base subgoal, dropping the hood.
    bulb_screw = [pair for pair in base["should_be_assembled"] if pair == [0, 1]]
    if not bulb_screw:
        raise SystemExit(
            "expected (0, 1) in should_be_assembled, got "
            f"{base['should_be_assembled']}"
        )

    out = {
        "_meta": {
            "source": "peg_in_hole_dynamic/furniture_bench/bulb_problem_setup/upstream_snapshots/lamp.py",
            "upstream_repo": "github.com/clvrai/furniture-bench (MIT)",
            "frame": "y-up (verbatim from upstream).",
            "z_up_conversion_note": "Apply R_x(+90 deg) so (x,y,z)_yup -> (x,-z,y)_zup; same R rotates child-relative poses.",
        },
        "parts": base["parts"],
        "assembled_rel_poses_yup": base["assembled_rel_poses_yup"],
        "tasks": {
            "lamp": {"should_be_assembled": base["should_be_assembled"]},
            "bulb_screw": {"should_be_assembled": bulb_screw},
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"  parts: {[p['name'] for p in out['parts']]}")
    print(f"  rel_poses keys: {sorted(out['assembled_rel_poses_yup'].keys())}")
    print(f"  bulb_screw.should_be_assembled: {out['tasks']['bulb_screw']['should_be_assembled']}")


if __name__ == "__main__":
    main()
