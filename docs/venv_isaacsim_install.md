# Unified `.venv_isaacsim` install (Isaac Sim + Fast-FoundationStereo + ROS Noetic)

The depth-student pipeline under `isaacsimenvs/`, `peg_in_hole_dynamic/`, and `deployment/` runs out of a **single Python 3.11 venv** at `.venv_isaacsim/`. Isaac Sim, Fast-FoundationStereo, and ROS 1 (Noetic) clients all coexist in this one venv as long as packages are installed in the order below. No apt, no separate conda env.

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
```

Then download Fast-FS weights + build TRT engines per §2b–2c above if needed.
