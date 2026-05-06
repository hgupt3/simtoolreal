"""Step 1 of one_leg problem-setup pipeline.

Regex-parse the upstream `square_table.py` + `one_leg.py` snapshots in
``./upstream_snapshots/`` and emit a JSON of part list, assembled
relative poses, and per-task ``should_be_assembled`` orderings. The
JSON is the canonical input for step2 (visualize / tune in viser) and
step3 (generate per-problem URDFs and CoACD).

Output: ``assets/urdf/furniture_bench/square_table/assembly.json``.

Convention notes
----------------
* Upstream uses **y-up** axes. Poses here are saved in y-up verbatim;
  z-up conversion is the responsibility of consumers (problems.py /
  step2 / step3) so the JSON is one-to-one with upstream.
* Each ``assembled_rel_poses`` entry is a *list* — multiple poses mean
  the parts are symmetric (e.g., 4 legs of a square table → 4 corners).

Run:
    .venv/bin/python -m peg_in_hole_dynamic.furniture_bench.one_leg_problem_setup.step1_extract_poses
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAP_DIR = Path(__file__).resolve().parent / "upstream_snapshots"
OUT_PATH = REPO_ROOT / "assets" / "urdf" / "furniture_bench" / "square_table" / "assembly.json"


_PARTS_RE = re.compile(
    r"""\b\w+\(\s*furniture_conf\[["'](?P<name>\w+)["']\]\s*,\s*(?P<idx>\d+)\s*\)"""
)

# Locator for `self.assembled_rel_poses[(p, c)] = [` — the matching `]`
# is found by bracket counting because the body contains nested `[...]`
# inside each get_mat([...], [...]) call.
_RELPOSE_LHS_RE = re.compile(
    r"""self\.assembled_rel_poses\[\(\s*(?P<p>\d+)\s*,\s*(?P<c>\d+)\s*\)\]\s*=\s*\["""
)


def _find_matching_close(text: str, open_idx: int) -> int:
    """Index of the `]` that matches the `[` at ``text[open_idx]``."""
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
    raise ValueError("unmatched [ at index {}".format(open_idx))
_RELPOSE_ALIAS_RE = re.compile(
    r"""self\.assembled_rel_poses\[\(\s*(?P<p2>\d+)\s*,\s*(?P<c2>\d+)\s*\)\]\s*=\s*self\.assembled_rel_poses\[\(\s*(?P<p1>\d+)\s*,\s*(?P<c1>\d+)\s*\)\]"""
)
_GET_MAT_RE = re.compile(
    r"""get_mat\(\s*\[(?P<pos>[-\d\.,\s]+)\]\s*,\s*\[(?P<rpy>[-\d\.,\s\w\*\(\)/\+]+?)\]\s*\)""",
)
_SHOULD_RE = re.compile(
    r"""self\.should_be_assembled\s*=\s*\[(?P<body>[^\]]*)\]""", re.DOTALL,
)
_PAIR_RE = re.compile(r"""\(\s*(\d+)\s*,\s*(\d+)\s*\)""")
_ATTACH_RE = re.compile(r"""self\.skill_attach_part_idx\s*=\s*(\d+)""")
_NUM_LIST_RE = re.compile(r"""[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?""")


def _parse_get_mat(call_text: str):
    """One get_mat([...], [...]) call -> {pos, rpy_xyz}.

    The rpy can contain references like np.pi; we eval those in a
    minimal namespace because the upstream uses them.
    """
    m = _GET_MAT_RE.search(call_text)
    if not m:
        raise ValueError(f"no get_mat in: {call_text!r}")
    pos = [float(x) for x in m.group("pos").split(",") if x.strip()]
    # rpy can be e.g. "0, 0, np.pi" — eval safely.
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
    """Body of a [get_mat(...), get_mat(...), ...] list -> list of dicts."""
    poses = []
    # Split on top-level commas between get_mat(...) calls — easiest: regex
    # for each call individually.
    for m in re.finditer(
        r"""get_mat\(\s*\[[^\]]*\]\s*,\s*\[[^\]]*\]\s*\)""", body
    ):
        poses.append(_parse_get_mat(m.group(0)))
    if not poses:
        raise ValueError(f"no get_mat calls in body: {body!r}")
    return poses


def _parse_square_table(src: str):
    """Returns dict with parts, assembled_rel_poses, should_be_assembled,
    skill_attach_part_idx."""
    # Parts (only those matching the furniture_conf[...] pattern).
    parts_seen = []
    for m in _PARTS_RE.finditer(src):
        parts_seen.append({"idx": int(m.group("idx")), "name": m.group("name")})
    parts_seen.sort(key=lambda d: d["idx"])

    rel_poses = {}
    for m in _RELPOSE_LHS_RE.finditer(src):
        open_idx = m.end() - 1   # the `[` we just matched
        close_idx = _find_matching_close(src, open_idx)
        body = src[open_idx + 1: close_idx]
        # Aliases (`= self.assembled_rel_poses[...]`) don't match _RELPOSE_LHS_RE
        # — they have no `[` before the LHS bracket — so no skip needed here.
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

    attach = None
    m = _ATTACH_RE.search(src)
    if m:
        attach = int(m.group(1))

    return {
        "parts": parts_seen,
        "assembled_rel_poses_yup": rel_poses,
        "should_be_assembled": should,
        "skill_attach_part_idx": attach,
    }


def _parse_one_leg_override(src: str):
    """OneLeg(SquareTable): just overrides should_be_assembled."""
    out = {}
    m = _SHOULD_RE.search(src)
    if m:
        pairs = [[int(p[0]), int(p[1])] for p in _PAIR_RE.findall(m.group("body"))]
        out["should_be_assembled"] = pairs
    return out


def main():
    sq_src = (SNAP_DIR / "square_table.py").read_text()
    one_src = (SNAP_DIR / "one_leg.py").read_text()

    base = _parse_square_table(sq_src)
    one_leg = _parse_one_leg_override(one_src)

    out = {
        "_meta": {
            "source": "peg_in_hole_dynamic/furniture_bench/one_leg_problem_setup/upstream_snapshots/{square_table.py,one_leg.py}",
            "upstream_repo": "github.com/clvrai/furniture-bench (MIT)",
            "frame": "y-up (verbatim from upstream).",
            "z_up_conversion_note": "Apply R_x(+90 deg) so (x,y,z)_yup -> (x,-z,y)_zup; same R rotates child-relative poses.",
        },
        "parts": base["parts"],
        "assembled_rel_poses_yup": base["assembled_rel_poses_yup"],
        "tasks": {
            "square_table": {
                "should_be_assembled": base["should_be_assembled"],
                "skill_attach_part_idx": base["skill_attach_part_idx"],
            },
            "one_leg": {
                "should_be_assembled": one_leg["should_be_assembled"],
            },
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"  parts: {[p['name'] for p in out['parts']]}")
    print(f"  rel_poses keys: {sorted(out['assembled_rel_poses_yup'].keys())}")
    print(f"  square_table.should_be_assembled: {out['tasks']['square_table']['should_be_assembled']}")
    print(f"  one_leg.should_be_assembled:      {out['tasks']['one_leg']['should_be_assembled']}")


if __name__ == "__main__":
    main()
