"""Step 1 of peg_board_1 problem-setup pipeline.

Auto-match each peg-board hole to a long peg by cross-section
(Hungarian assignment), render all 9 placements in viser, expose a
per-hole dropdown with dx / dy / yaw nudge sliders for visual
fine-tuning, and save the final pose table to JSON. Subsequent steps
(URDF generation, NAME_TO_OBJECT registration, Problem registration)
read this JSON.

Run:
    .venv/bin/python -m peg_in_hole_dynamic.fmb.peg_board_problem_setup.step1_pair_and_visualize
"""
import json, time
from pathlib import Path
import numpy as np
import trimesh
import viser
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT / 'assets' / 'urdf' / 'fmb'
ext = json.load(open(ROOT / 'fmb_extents.json'))
SAVE_PATH = Path(__file__).with_name('peg_board_1_assemblies.json')

# ── Board → A-frame ──
b_scene = trimesh.load_mesh(str(ROOT/'boards/peg_board_1/peg_board_1.obj'), process=False)
T_to_A = np.array([
    -(b_scene.bounds[0,0] + b_scene.bounds[1,0]) / 2,
    -(b_scene.bounds[0,1] + b_scene.bounds[1,1]) / 2,
    -b_scene.bounds[0,2],
])
b = b_scene.copy()
b.vertices = b.vertices + T_to_A
top_z, bot_z = float(b.bounds[1,2]), float(b.bounds[0,2])
print(f"Board in A: z=[{bot_z:.4f},{top_z:.4f}]   T_scene→A={T_to_A.tolist()}")

# ── Detect 9 holes ──
mask = (np.abs(b.face_normals[:,2]) < 0.3) & \
       (b.triangles_center[:,2] > bot_z+0.001) & (b.triangles_center[:,2] < top_z-0.001)
xy = b.triangles_center[mask, :2]
labels = DBSCAN(eps=0.015, min_samples=3).fit_predict(xy)
holes = []
for k in range(labels.max() + 1):
    pts = xy[labels == k]
    cx = (pts[:,0].min() + pts[:,0].max()) / 2
    cy = (pts[:,1].min() + pts[:,1].max()) / 2
    dx = pts[:,0].max() - pts[:,0].min()
    dy = pts[:,1].max() - pts[:,1].min()
    if max(dx, dy) > 0.005:
        holes.append((float(cx), float(cy), float(dx), float(dy)))

# ── Long pegs: precompute rotated XY extents at multiple yaw candidates ──
# Aspect-ratio + clearance matching alone confuses two square-ish pegs/holes
# (clearance is symmetric on swap). Adding rotated-bbox sampling lets star /
# diagonal-mounted pegs be matched at non-axis-aligned yaws too.
# Yaws sampled for matching. Restricted to axis-aligned {0°, 90°} so that
# odd/symmetric pegs (e.g. star, asterisk) can't steal big square holes by
# rotating 45° to a smaller diagonal bbox. Diagonal-orientation pegs are
# corrected by hand via the GUI yaw slider after matching.
YAW_CANDIDATES_DEG = [0.0, 90.0]

long_pegs = []
peg_yaw_extents = {}   # name -> list of (yaw_deg, ext_x, ext_y)
for k, v in ext.items():
    if not k.startswith('peg_') or 'board' in k or 'fixture' in k: continue
    if v['extents_m'][2] < 0.13: continue
    long_pegs.append((k, v['extents_m'][0], v['extents_m'][1]))
    p = trimesh.load_mesh(str(ROOT/'pegs'/k/f'{k}.obj'), process=False)
    xy_v = p.vertices[:, :2].astype(np.float64).copy()
    xy_v -= (p.bounds[0, :2] + p.bounds[1, :2]) / 2     # centre
    el = []
    for yaw in YAW_CANDIDATES_DEG:
        c, s = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
        rx = c * xy_v[:, 0] - s * xy_v[:, 1]
        ry = s * xy_v[:, 0] + c * xy_v[:, 1]
        el.append((yaw, float(rx.max() - rx.min()), float(ry.max() - ry.min())))
    peg_yaw_extents[k] = el

# ── Hungarian assignment ──
# Cost = sum of squared per-axis clearances (m²). This penalises *lopsided*
# fits more heavily than just sum-of-clearances, so when two pegs have the
# same total clearance for a hole, the symmetric (well-fitting) one wins —
# which fixes the square-peg-vs-square-hole swap ambiguity. The yaw is the
# rotation that makes the rotated peg bbox fit best in the hole.
INF = 1e9
n_h, n_p = len(holes), len(long_pegs)
cost      = np.full((n_h, n_p), INF)
yaws_init = np.zeros((n_h, n_p))
for i, (_, _, dx, dy) in enumerate(holes):
    for j, (name, _, _) in enumerate(long_pegs):
        best, best_yaw = INF, 0.0
        for yaw, px, py in peg_yaw_extents[name]:
            cl = 0.001
            if not (px <= dx - cl and py <= dy - cl):
                continue
            c = (dx - px) ** 2 + (dy - py) ** 2
            if c < best:
                best, best_yaw = c, yaw
        cost[i, j], yaws_init[i, j] = best, best_yaw
row_ind, col_ind = linear_sum_assignment(cost)

assignments = {}  # hole_idx → dict
for r, c in zip(row_ind, col_ind):
    name = long_pegs[c][0]
    cx, cy, dx, dy = holes[r]
    assignments[r] = {
        'peg': name,
        'hole_xy_A': (cx, cy),
        'hole_bbox': (dx, dy),
        'peg_xy_extent': (long_pegs[c][1], long_pegs[c][2]),
        'yaw_deg': float(yaws_init[r, c]),
        'dx_mm': 0.0,
        'dy_mm': 0.0,
    }
    print(f"  hole_{r} bbox {dx*1000:.1f}×{dy*1000:.1f} → {name:>7s} "
          f"({long_pegs[c][1]*1000:.1f}×{long_pegs[c][2]*1000:.1f})  yaw_init={yaws_init[r,c]:.0f}°")

# ── viser scene ──
server = viser.ViserServer(host='0.0.0.0', port=8055)
print('http://localhost:8055')
server.scene.add_grid('/ground', width=0.6, height=0.6, cell_size=0.05)
server.scene.add_frame('/world', position=(0,0,0), wxyz=(1,0,0,0),
                        show_axes=True, axes_length=0.05, axes_radius=0.002)

board_handle = server.scene.add_mesh_simple(
    '/board1', vertices=b.vertices.astype(np.float32),
    faces=b.faces.astype(np.int32), color=(200,150,80), opacity=0.4,
)

# Cache each peg's centred (XY-centroid, z-min subtracted) base vertices
peg_bases = {}
peg_handles = {}
peg_palette = [(220,40,40),(40,180,40),(60,80,220),(220,160,40),
               (180,40,220),(40,200,200),(220,100,40),(150,150,40),(100,40,200)]

def _build_pose_vertices(peg_name, dx_m, dy_m, yaw_deg, hole_xy):
    """Apply 180°X flip + yaw + nudge, place at hole_xy with z=[0,0.15]."""
    base = peg_bases[peg_name]   # vertices in peg-local frame, z=[0,0.15], xy centred
    v = base.copy()
    # 180°X
    v[:, 1] = -v[:, 1]
    v[:, 2] = -v[:, 2]   # z=[-0.15,0]
    # yaw about Z
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    rx = c * v[:, 0] - s * v[:, 1]
    ry = s * v[:, 0] + c * v[:, 1]
    v[:, 0] = rx; v[:, 1] = ry
    # translate to hole + nudge, lift z so peg's bottom is at z=0
    v[:, 0] += hole_xy[0] + dx_m
    v[:, 1] += hole_xy[1] + dy_m
    v[:, 2] += 0.15
    return v.astype(np.float32)

for h_idx, info in assignments.items():
    name = info['peg']
    p = trimesh.load_mesh(str(ROOT/'pegs'/name/f'{name}.obj'), process=False)
    cx_s = (p.bounds[0,0] + p.bounds[1,0]) / 2
    cy_s = (p.bounds[0,1] + p.bounds[1,1]) / 2
    base = p.vertices.copy()
    base[:, 0] -= cx_s
    base[:, 1] -= cy_s
    base[:, 2] -= p.bounds[0, 2]
    peg_bases[name] = base.astype(np.float32)
    peg_handles[name] = {'faces': p.faces.astype(np.int32),
                         'color': peg_palette[h_idx % len(peg_palette)],
                         'h_idx': h_idx,
                         'h': None}

SELECTED_COLOR = (255, 240, 60)   # bright yellow highlight for the selected peg

def _refresh_peg(h_idx, selected=False):
    info = assignments[h_idx]
    name = info['peg']
    v = _build_pose_vertices(name, info['dx_mm']*1e-3, info['dy_mm']*1e-3,
                             info['yaw_deg'], info['hole_xy_A'])
    if peg_handles[name]['h'] is not None:
        try: peg_handles[name]['h'].remove()
        except Exception: pass
    color = SELECTED_COLOR if selected else peg_handles[name]['color']
    peg_handles[name]['h'] = server.scene.add_mesh_simple(
        f'/pegs/{name}', vertices=v, faces=peg_handles[name]['faces'],
        color=color, opacity=float(peg_op_slider.value),
    )

# ── GUI ──
with server.gui.add_folder('Scene opacity'):
    board_op = server.gui.add_slider('Board', min=0.0, max=1.0, step=0.05, initial_value=0.4)
    peg_op_slider = server.gui.add_slider('Pegs', min=0.0, max=1.0, step=0.05, initial_value=1.0)

with server.gui.add_folder('Tweak placement'):
    hole_dropdown = server.gui.add_dropdown(
        'Hole', options=[f'hole_{i} ({assignments[i]["peg"]})' for i in range(n_h)],
        initial_value=f'hole_0 ({assignments[0]["peg"]})',
    )
    dx_slider  = server.gui.add_slider('dx (mm)',  min=-10.0, max=10.0, step=0.1, initial_value=0.0)
    dy_slider  = server.gui.add_slider('dy (mm)',  min=-10.0, max=10.0, step=0.1, initial_value=0.0)
    yaw_slider = server.gui.add_slider('yaw (deg)', min=-180.0, max=180.0, step=1.0, initial_value=0.0)
    save_btn   = server.gui.add_button('Save → JSON')
    info_md    = server.gui.add_markdown('')

def _selected_idx():
    return int(hole_dropdown.value.split(' ')[0].split('_')[1])

def _refresh_all():
    """Re-render every peg, with the currently-selected one in highlight color."""
    sel = _selected_idx()
    for h_idx in assignments:
        _refresh_peg(h_idx, selected=(h_idx == sel))

def _show_info():
    i = _selected_idx()
    a = assignments[i]
    info_md.content = (
        f"**hole_{i}** → **{a['peg']}**  (highlighted yellow)\n"
        f"- hole xy (A) = ({a['hole_xy_A'][0]:+.4f}, {a['hole_xy_A'][1]:+.4f})\n"
        f"- nudge = ({a['dx_mm']:+.1f}, {a['dy_mm']:+.1f}) mm\n"
        f"- yaw = {a['yaw_deg']:.1f}°\n"
    )

@hole_dropdown.on_update
def _on_hole(_):
    a = assignments[_selected_idx()]
    dx_slider.value = a['dx_mm']
    dy_slider.value = a['dy_mm']
    yaw_slider.value = a['yaw_deg']
    _refresh_all()
    _show_info()

@dx_slider.on_update
def _on_dx(_):
    i = _selected_idx(); assignments[i]['dx_mm'] = float(dx_slider.value)
    _refresh_peg(i, selected=True); _show_info()

@dy_slider.on_update
def _on_dy(_):
    i = _selected_idx(); assignments[i]['dy_mm'] = float(dy_slider.value)
    _refresh_peg(i, selected=True); _show_info()

@yaw_slider.on_update
def _on_yaw(_):
    i = _selected_idx(); assignments[i]['yaw_deg'] = float(yaw_slider.value)
    _refresh_peg(i, selected=True); _show_info()

@board_op.on_update
def _on_bo(_): board_handle.opacity = float(board_op.value)

@peg_op_slider.on_update
def _on_po(_):
    for d in peg_handles.values():
        if d['h'] is not None:
            try: d['h'].opacity = float(peg_op_slider.value)
            except Exception: pass

@save_btn.on_click
def _on_save(_):
    out = {f'hole_{i}': {
        'peg': a['peg'],
        'hole_xy_A':    list(a['hole_xy_A']),
        'hole_bbox':    list(a['hole_bbox']),
        'peg_xy_extent': list(a['peg_xy_extent']),
        'yaw_deg':       a['yaw_deg'],
        'nudge_mm':     [a['dx_mm'], a['dy_mm']],
    } for i, a in assignments.items()}
    SAVE_PATH.write_text(json.dumps(out, indent=2))
    info_md.content = f"Saved → {SAVE_PATH}"
    print(f"Saved {SAVE_PATH}")

# Render all initial — selected one (hole_0) gets highlight color
_refresh_all()
_show_info()

print('Loaded. Press Ctrl+C to exit.')
try:
    while True: time.sleep(1)
except KeyboardInterrupt: pass
