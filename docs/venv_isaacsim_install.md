# Unified `.venv_isaacsim` install (Isaac Sim + Fast-FoundationStereo + FoundationPose + pyzed + ROS Noetic)

The depth-student pipeline under `isaacsimenvs/`, `peg_in_hole_dynamic/`, `deployment/`, and `third_party/FoundationPose/` runs out of a **single Python 3.11 venv** at `.venv_isaacsim/`. Isaac Sim, Fast-FoundationStereo, FoundationPose, pyzed, and ROS 1 (Noetic) clients all coexist in this one venv as long as packages are installed in the order below. No apt, no conda env.

The main [docs/installation.md](installation.md) still covers the original Isaac Gym (Python 3.8) workflow — that env is independent and should remain separate.

## Prerequisites

- Python 3.11 (Isaac Sim 5.x / Isaac Lab 2.3.x requirement)
- NVIDIA GPU with driver >= 525.60
- CUDA 12+
- `uv` for package management ([install instructions](https://docs.astral.sh/uv/getting-started/installation/))

## 1. Base Isaac Sim install

Create the venv and install the Isaac Sim stack. This step is also documented in [isaacsim_conversion/README.md](../isaacsim_conversion/README.md); we use `.venv_isaacsim` here (not `.venv-isaacsim-py311`) because that's the path the rest of the depth-student tooling assumes.

```bash
uv venv .venv_isaacsim --python 3.11
source .venv_isaacsim/bin/activate

# PyTorch for CUDA 12.6 (Isaac Sim 5.x is built against this)
uv pip install torch --index-url https://download.pytorch.org/whl/cu126

# Vendored rl_games + inference deps
uv pip install -e ./rl_games/
uv pip install omegaconf hydra-core "gym==0.23.1" scipy numpy yourdfpy requests tqdm tyro

# Isaac Lab + Isaac Sim (~15 GB download; first launch builds RTX shaders, takes ~2-5 min)
uv pip install "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com
```

Register the repo-local packages (`isaacsimenvs`, `peg_in_hole_dynamic`, `deployment`, `fabrica`, etc.) so they're importable from the venv:

```bash
uv pip install -e . --no-deps
```

`--no-deps` is required: the root `pyproject.toml` pins `numpy==1.23.0`, `warp-lang==0.10.1`, and `isaacgym-stubs`, which all conflict with Python 3.11 / Isaac Sim. The repo packages themselves install cleanly.

If you add a new top-level package directory, add it to `[tool.setuptools.packages.find]` in `pyproject.toml` and re-run the line above.

Verify:

```bash
.venv_isaacsim/bin/python -c "
import torch, isaaclab, isaacsimenvs
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('isaaclab:', isaaclab.__file__)
"
```

Optional speedup — point the Omniverse shader cache at a local SSD instead of NFS:

```bash
export OMNI_KIT_CACHE_PATH=/scratch/$USER/ov_cache
mkdir -p "$OMNI_KIT_CACHE_PATH"
```

## 2. Fast-FoundationStereo

Bring up [NVlabs/Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo) (CVPR 2026) for sim eval and real-robot deployment. Source is vendored at `third_party/Fast-FoundationStereo/` (stripped of upstream `.git/` and README media to keep the repo small).

### 2a. Python deps

```bash
uv pip install --python .venv_isaacsim/bin/python \
  timm einops omegaconf gdown \
  tensorrt-cu12 onnxruntime-gpu
```

**Do NOT install `xformers` or `nvidia-modelopt[torch]`.** Both silently pull `torch==2.12+cu130`, which breaks Isaac Sim's CUDA stack (cuDNN init fails, `torch.cuda.is_available() == False`). Fast-FS does not actually need `xformers` despite the upstream README mentioning it.

If torch ever gets clobbered, restore with:

```bash
uv pip install --python .venv_isaacsim/bin/python --reinstall 'torch==2.7.0' \
  --index-url https://download.pytorch.org/whl/cu126
```

### 2b. Download model weights

The four published Fast-FS variants live in a shared Google Drive folder:

```bash
.venv_isaacsim/bin/python -m gdown --folder \
  'https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap' \
  -O third_party/Fast-FoundationStereo/weights/
# Then flatten the nested weights/weights/ structure:
mv third_party/Fast-FoundationStereo/weights/weights/* \
   third_party/Fast-FoundationStereo/weights/ && \
   rmdir third_party/Fast-FoundationStereo/weights/weights
rm -rf third_party/Fast-FoundationStereo/weights/onnx   # public ONNX exports at non-matching resolutions
```

After this, `third_party/Fast-FoundationStereo/weights/` contains:

```
15-44-51/  20-26-39/  20-30-48/  23-36-37/
```

Each with `model_best_bp2_serialize.pth` + `cfg.yaml`. ~260 MB total. All `.pth` files are ignored by the FS-local `.gitignore`.

Public model trade-offs (from upstream `readme.md`, 4-iter bench on a 3090 at 640x480):

| model      | PyTorch | TRT     | peak mem |
|------------|---------|---------|----------|
| `23-36-37` | 41.1 ms | 18.4 ms | 653 MB   |
| `20-26-39` | 37.5 ms | 16.4 ms | 651 MB   |
| `20-30-48` | 29.3 ms | 14.0 ms | 646 MB   |

### 2c. Inference paths

#### PyTorch (bring-up / fallback)

```bash
.venv_isaacsim/bin/python third_party/Fast-FoundationStereo/scripts/run_demo.py \
  --model_dir third_party/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth \
  --valid_iters 4 --get_pc 0 \
  --out_dir /tmp/fast_fs_demo_pt
```

Loaded with `torch.load(..., weights_only=False)` (the checkpoint is a serialized `FastFoundationStereo` instance, not a state_dict). Forward signature:

```python
disp = model.forward(left, right, iters=4, test_mode=True,
                     optimize_build_volume='pytorch1')
```

Inputs: `(B, 3, H, W) float32` in `[0, 255]` (no ImageNet norm), padded to a multiple of 32 via `core.utils.utils.InputPadder`. Output: disparity in pixels.

#### TensorRT (deployment)

Two-step build because the model is split into a feature backbone (`feature_runner`) and an iterative-refinement head (`post_runner`):

```bash
ONNX_DIR=third_party/Fast-FoundationStereo/weights/23-36-37/onnx_384x224_iters4
.venv_isaacsim/bin/python third_party/Fast-FoundationStereo/scripts/make_onnx.py \
  --model_dir third_party/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth \
  --save_path $ONNX_DIR \
  --height 224 --width 384 --valid_iters 4 --max_disp 192
.venv_isaacsim/bin/python deployment/build_trt_engine.py --onnx_dir $ONNX_DIR
```

Build times (RTX 6000 Ada, fp16):
- `feature_runner.engine`: ~2.5 min, 20.6 MiB
- `post_runner.engine`:    ~10 min, 13.6 MiB

`build_trt_engine.py` uses the TensorRT Python API (`trt.Builder.build_serialized_network`), no `trtexec` binary needed (the pip-distributed `tensorrt-cu12` does not ship `trtexec`).

Inference is via `core.foundation_stereo.TrtRunner`:

```python
from core.foundation_stereo import TrtRunner
m = TrtRunner(cfg, f'{onnx_dir}/feature_runner.engine',
                   f'{onnx_dir}/post_runner.engine')
disp = m.forward(left, right)   # left/right: (1,3,H,W) float [0,255]
```

`cfg` is loaded from `$ONNX_DIR/onnx.yaml` (auto-written by `make_onnx.py`).

#### Measured throughput at 384x224, 4 iters, RTX 6000 Ada, FP16

| backend  | latency  | fps  | disparity match vs PyTorch              |
|----------|----------|------|------------------------------------------|
| PyTorch  | 25.94 ms | 39   | (reference)                              |
| **TRT**  | **3.87 ms** | **258** | mean abs 0.009 px, 0% pixels disagree > 1 px |

TRT is 6.7x faster and bit-equivalent. The team's 50 fps target (3090) maps to ~250 fps on the lab's RTX 6000 Ada — the resulting headroom lets the FS stage co-exist with the policy inference loop without stealing time from physics.

### 2d. Notes / gotchas

- `*.pth`, `*.onnx`, `*.engine` files in `third_party/Fast-FoundationStereo/weights/` are gitignored — regenerate from the steps above when re-cloning.
- ONNX export emits many `TracerWarning` lines for control flow that depends on tensor shapes; they are benign for a fixed-resolution engine (we bake the shape into the export) but invalidate the engine if you change `--height`/`--width`/`--valid_iters` — rebuild for each new resolution.
- TRT `enqueueV3` emits `Using default stream...` warnings on every call. Cosmetic only. For production deployment, switch to a non-default stream by patching `TrtRunner.forward` to pass an explicit `torch.cuda.Stream`.
- Engine files are GPU-architecture-specific. Engines built on RTX 6000 Ada (Ada Lovelace, sm_89) won't load on a 3090 (Ampere, sm_86) — rebuild per target GPU.
- The serialized `.pth` files in `weights/{model}/` are full `nn.Module` pickles, **not** state_dicts. `torch.load` needs `weights_only=False`. This is a security-relevant flag if you ever load checkpoints from untrusted sources.

## 3. ROS Noetic Python client (`rospypi`)

`rospypi/simple` ships pre-built ROS 1 Noetic Python wheels for Python 3.11 — no apt install, no source build, no separate conda env:

```bash
uv pip install --python .venv_isaacsim/bin/python \
  --extra-index-url https://rospypi.github.io/simple/ \
  rospy rosgraph rosmaster \
  std_msgs sensor_msgs geometry_msgs nav_msgs \
  tf2_ros tf2_msgs \
  cv_bridge
```

`rospypi` mirrors most `ros-noetic-*` message packages — add what you need. The 16-package install above is sufficient for our deployment nodes.

### Smoke test

```bash
# Start a roscore from the venv. rosmaster has no `__main__`, so invoke its main directly:
.venv_isaacsim/bin/python -c \
  "import sys; sys.argv=['rosmaster','--core','-p','11311']; \
   import rosmaster; rosmaster.rosmaster_main()" &

# Pub/sub round-trip:
.venv_isaacsim/bin/python - <<'PY'
import os, time
os.environ.setdefault("ROS_MASTER_URI", "http://localhost:11311")
import rospy
from std_msgs.msg import String
rospy.init_node("smoke", anonymous=True, disable_signals=True)
got = []
rospy.Subscriber("/hello", String, lambda m: got.append(m.data))
pub = rospy.Publisher("/hello", String, queue_size=1, latch=True)
time.sleep(1.0); pub.publish("ok"); time.sleep(1.0)
print("received:", got)
rospy.signal_shutdown("done")
PY
```

Expected output: `received: ['ok']`.

Or use the entrypoint script for the roscore: `.venv_isaacsim/bin/rosmaster --core -p 11311 &`.

### What's *not* in the venv

- C++ ROS tools (`rviz`, `rqt`, `rosbag` record/replay, `gazebo`) — those still need apt or RoboStack.
- `moveit` and other C++-heavy ROS packages.
- ROS 2 bindings — use Isaac Sim's bundled `isaacsim.ros2.bridge` extension (enabled via `AppLauncher`) for those.

To talk to a real-robot ROS Noetic master on another host:

```bash
export ROS_MASTER_URI=http://<host>:11311
export ROS_IP=<this-host-ip>
.venv_isaacsim/bin/python deployment/your_node.py
```

## 4. FoundationPose

Slim FoundationPose (live 6-DoF pose tracking + ROS publishing) is vendored at `third_party/FoundationPose/`. Source upstream: [kushal2000/FoundationPose](https://github.com/kushal2000/FoundationPose) (a fork of NVlabs/FoundationPose); only the files needed for the live ROS flow were copied — see `third_party/FoundationPose/`.

Upstream pins `torch==2.0.0+cu118`. We do **not** match that pin: we run FoundationPose on top of Isaac Sim's `torch 2.7+cu126`, rebuilding PyTorch3D, nvdiffrast, and `mycpp` against the modern CUDA. This is the only reason we don't follow the upstream conda recipe.

### 4a. Python deps

```bash
uv pip install --python .venv_isaacsim/bin/python \
  kornia open3d pyrender PyOpenGL PyOpenGL_accelerate \
  fvcore opencv-contrib-python scikit-learn scikit-image \
  ninja pybind11 ruamel.yaml colorama joblib roma transformations
```

**Then re-pin numpy<2.** The packages above resolve numpy ≥ 2.0 by default, but `numba==0.59.1` (already installed via Isaac Sim's `warp-lang` dependency tree) hard-errors against numpy 2.x:

```bash
uv pip install --python .venv_isaacsim/bin/python "numpy<2"
```

Versions already in the venv from §1 (Isaac Sim) — `trimesh`, `transformers`, `einops`, `omegaconf`, `imageio`, `opencv-python`, `h5py`, `psutil`, `pyglet`, `xatlas`, `rtree`, `timm`, `warp-lang` — are left alone. They are all newer than FoundationPose's upstream pins but compatible.

### 4b. Source-build PyTorch3D, nvdiffrast, mycpp

```bash
export CUDA_HOME=/usr/local/cuda          # system CUDA 12.x — no conda needed
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# PyTorch3D against torch 2.7+cu126. Slow (~5-10 min); compiles CUDA kernels.
uv pip install --python .venv_isaacsim/bin/python --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git"

# nvdiffrast — fast (~1 min). Version-flexible.
uv pip install --python .venv_isaacsim/bin/python --no-build-isolation \
  "git+https://github.com/NVlabs/nvdiffrast"

# mycpp (FoundationPose's pose-clustering pybind11 extension). Builds with
# system gcc-13 + system CUDA 12.x; no CUDA 11.8 / gcc-11 required despite
# what upstream FoundationPose's readme says.
PYTHON=.venv_isaacsim/bin/python bash third_party/FoundationPose/build_all.sh
```

`mycpp` output lands at `third_party/FoundationPose/mycpp/build/mycpp*.so`. `Utils.py:46` imports it as `mycpp.build.mycpp` — the build directory is the import path; don't move the `.so`.

### 4c. Download FoundationPose weights

Fetch the two pretrained checkpoint folders from upstream [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose#model-weights) (Google Drive folder `1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i`):

```bash
.venv_isaacsim/bin/python -m gdown --folder \
    'https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i' \
    -O third_party/FoundationPose/weights/

# gdown unpacks into weights/no_diffusion/<timestamp>/; flatten one level so
# the layout matches what FP expects (predict_score.py:123, predict_pose_refine.py:99).
cd third_party/FoundationPose/weights
mv no_diffusion/2023-10-28-18-33-37 . && \
mv no_diffusion/2024-01-11-20-02-45 . && \
rmdir no_diffusion
cd -
```

Final layout (~248 MB; both gitignored):

```
third_party/FoundationPose/weights/
  ├── 2023-10-28-18-33-37/   # scorer  — 66 MB
  │   ├── config.yml
  │   └── model_best.pth
  └── 2024-01-11-20-02-45/   # refiner — 182 MB
      ├── config.yml
      └── model_best.pth
```

Verify both checkpoints load under our torch 2.7+cu126:

```bash
.venv_isaacsim/bin/python -c "
import sys; sys.path.insert(0, 'third_party/FoundationPose')
from estimater import ScorePredictor, PoseRefinePredictor
ScorePredictor(); PoseRefinePredictor()
print('FP weights load OK')
"
```

### 4d. Smoke test (no camera required)

```bash
.venv_isaacsim/bin/python -c "
import sys; sys.path.insert(0, 'third_party/FoundationPose')
import torch, pytorch3d, nvdiffrast.torch as dr
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('pytorch3d:', pytorch3d.__version__)
from Utils import mycpp
assert mycpp is not None, 'mycpp .so missing — re-run build_all.sh'
print('mycpp ok')
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
print('FoundationPose imports ok')
"
```

### 4e. Notes / gotchas

- **Do NOT `pip install pyzed`.** The PyPI package by that name is a squatter and lacks `pyzed.sl`. See §5 for the correct path.
- **Re-pin numpy<2 after FP deps.** `numba` (Isaac Sim transitive) refuses numpy 2.x.
- **`mycpp` import path is hard-coded** as `mycpp.build.mycpp` (see `Utils.py:46`). The `try/except` wrap means missing `.so` silently disables pose clustering — re-run `build_all.sh` if the smoke test prints `mycpp .so missing`.
- **`transformers` pin.** Upstream FoundationPose leaves `transformers` unconstrained but the upstream-tested torch is 2.0; under torch 2.7 here, the Isaac Sim `transformers>=5.x` already in the venv works fine. The vendored `requirements.txt` no longer pins `transformers==4.41.2` (the pin we'd need under torch 2.0).
- **GPU-arch-specific.** PyTorch3D / nvdiffrast / mycpp builds work for the GPU arches listed in `TORCH_CUDA_ARCH_LIST` above (Ampere through Hopper); rebuild for older arches if you target them.

## 5. ZED SDK + `pyzed` (FoundationPose camera input)

`pyzed` is used by the two live tracking scripts under `third_party/FoundationPose/` (`live_tracking_zed_depth.py` and `live_tracking_fast_fs.py`). **Stereolabs's `pyzed` is NOT on PyPI** — the PyPI package by that name is unrelated and lacks the `.sl` submodule. The real wrapper ships *inside the ZED SDK installer*.

### 5a. Install the ZED SDK (system-wide, requires sudo)

Pick the SDK build that matches the system CUDA (12.x here) and Ubuntu version. Browse <https://www.stereolabs.com/developers/release> for the current URL.

```bash
wget -O /tmp/zed_sdk.run https://download.stereolabs.com/zedsdk/<version>/cu12/ubuntu24
sudo bash /tmp/zed_sdk.run
```

### 5b. Install the matching `pyzed` wheel into the venv

```bash
source .venv_isaacsim/bin/activate
python /usr/local/zed/get_python_api.py
```

The script fetches a wheel matching the active Python interpreter (3.11 here) and runs `pip install` against it.

### 5c. Verify

```bash
.venv_isaacsim/bin/python -c \
  "import pyzed.sl as sl; print('SDK', sl.Camera.get_sdk_version())"
```

If you see `ModuleNotFoundError: No module named 'pyzed.sl'`, you installed the PyPI squatter — uninstall (`uv pip uninstall pyzed`) and rerun `get_python_api.py`.

## 6. Run FoundationPose live tracking

Two entry points share the same FoundationPose pipeline; they differ only in
where depth comes from:

| script                       | depth source                                                         |
|------------------------------|----------------------------------------------------------------------|
| `live_tracking_zed_depth.py` | ZED SDK neural depth (`sl.DEPTH_MODE.NEURAL`)                        |
| `live_tracking_fast_fs.py`   | Fast-FoundationStereo on the ZED's left+right rectified RGB (§2)     |

Both accept `--ros` as an optional flag. Without it, the script just runs the
FoundationPose pipeline (debug overlays via `--debug 1`); useful for bring-up
without a ROS master. Pass `--ros` to publish `PoseStamped` on the topics
listed below.

### 6a. ZED depth

```bash
# Optional: ROS master on this host, or point at an existing one.
.venv_isaacsim/bin/rosmaster --core -p 11311 &
export ROS_MASTER_URI=http://localhost:11311

.venv_isaacsim/bin/python third_party/FoundationPose/live_tracking_zed_depth.py \
    --mesh_path /path/to/object.obj \
    --serial_number 15107 \
    --fps 40 \
    --ros
```

### 6b. Fast-FoundationStereo depth

Requires the Fast-FS weights from §2b. Fast-FS runs at 384×224 (the size the
TRT engine recipe is built for; depth is upsampled to `--width × --height`
for FoundationPose). Two backends:

**PyTorch path** (slower, ~27 ms / forward on RTX 6000 Ada):
```bash
.venv_isaacsim/bin/python third_party/FoundationPose/live_tracking_fast_fs.py \
    --mesh_path /path/to/object.obj \
    --serial_number 15107 \
    --fps 40 \
    --ros
```

**TRT path** (~4 ms / forward — recommended; requires engines built per §2c):
```bash
.venv_isaacsim/bin/python third_party/FoundationPose/live_tracking_fast_fs.py \
    --mesh_path /path/to/object.obj \
    --serial_number 15107 \
    --fps 40 --ros \
    --fast_fs_trt_dir third_party/Fast-FoundationStereo/weights/23-36-37/onnx_384x224_iters4 \
    --track_refine_iter 1
```

`--fast_fs_trt_dir` must contain `feature_runner.engine`, `post_runner.engine`,
and `onnx.yaml`. With TRT enabled, `--fast_fs_width`, `--fast_fs_height`, and
`--fast_fs_iters` are ignored (engine bakes them in). Drop
`--track_refine_iter` to 1 to push the combined pipeline above 30 Hz; see
the table below for measured budgets.

Measured throughput on RTX 6000 Ada at 384×224 (synthetic inputs):

| config                            | per-frame | rate    |
|-----------------------------------|-----------|---------|
| FP iter=1 + TRT FS 4-iter         | 20.6 ms   | 48.5 Hz |
| FP iter=2 + TRT FS 4-iter         | 34.3 ms   | 29.1 Hz |
| FP iter=1 + PyTorch FS            | 43.2 ms   | 23.1 Hz |
| FP iter=2 + PyTorch FS            | 56.9 ms   | 17.6 Hz |

On the first frame an OpenCV window opens for SAM bounding-box selection — click 4 corners around the object; FoundationPose registers the initial pose and tracks from there.

Published topics:

| topic                                  | type           | when                            |
|----------------------------------------|----------------|---------------------------------|
| `camera_frame/current_object_pose`     | `PoseStamped`  | always                          |
| `robot_frame/current_object_pose`      | `PoseStamped`  | only if `--calibration` passed  |

`--calibration` is **optional** in our vendored copy (upstream made it required). Pass a 4x4 `T_RC` matrix (`.txt` or `.npy`) for robot-frame output; omit during bring-up.

## Quick reference — full unified install from scratch

```bash
# 1. Base Isaac Sim venv
uv venv .venv_isaacsim --python 3.11
source .venv_isaacsim/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
uv pip install -e ./rl_games/
uv pip install omegaconf hydra-core "gym==0.23.1" scipy numpy yourdfpy requests tqdm tyro
uv pip install "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com
uv pip install -e . --no-deps

# 2. Fast-FoundationStereo Python deps
uv pip install --python .venv_isaacsim/bin/python \
  timm einops omegaconf gdown tensorrt-cu12 onnxruntime-gpu

# 3. ROS Noetic Python client
uv pip install --python .venv_isaacsim/bin/python \
  --extra-index-url https://rospypi.github.io/simple/ \
  rospy rosgraph rosmaster std_msgs sensor_msgs geometry_msgs \
  nav_msgs tf2_ros tf2_msgs cv_bridge

# 4. FoundationPose deps + source builds
uv pip install --python .venv_isaacsim/bin/python \
  kornia open3d pyrender PyOpenGL PyOpenGL_accelerate \
  fvcore opencv-contrib-python scikit-learn scikit-image \
  ninja pybind11 ruamel.yaml colorama joblib roma transformations
uv pip install --python .venv_isaacsim/bin/python "numpy<2"
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
uv pip install --python .venv_isaacsim/bin/python --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git" \
  "git+https://github.com/NVlabs/nvdiffrast"
PYTHON=.venv_isaacsim/bin/python bash third_party/FoundationPose/build_all.sh

# 5. ZED SDK (system, sudo) then pyzed wheel into the venv
sudo bash zed_sdk.run                       # URL per §5a
.venv_isaacsim/bin/python /usr/local/zed/get_python_api.py
```

# 6. Download FP weights
.venv_isaacsim/bin/python -m gdown --folder \
    'https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i' \
    -O third_party/FoundationPose/weights/
( cd third_party/FoundationPose/weights && \
  mv no_diffusion/2023-10-28-18-33-37 no_diffusion/2024-01-11-20-02-45 . && \
  rmdir no_diffusion )
```

Then download Fast-FS weights + build TRT engines per §2b–2c.
