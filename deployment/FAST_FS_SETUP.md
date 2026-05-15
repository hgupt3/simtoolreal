# Fast-FoundationStereo setup notes

Bringing up [NVlabs/Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo) (CVPR 2026) for sim eval and real-robot deployment. Source is vendored at `third_party/Fast-FoundationStereo/` (stripped of upstream `.git/` and README media to keep the repo small).

## One-time setup

### 1. Python deps (into `.venv_isaacsim`)

```bash
uv pip install --python .venv_isaacsim/bin/python \
  timm einops omegaconf gdown \
  tensorrt-cu12 onnxruntime-gpu
```

**Do NOT install `xformers` in this venv.** Fast-FS does not import it (it's only in the upstream README), and `xformers` will silently pull `torch==2.12.0+cu130`, which breaks our `torch==2.7.0+cu126` Isaac Sim stack (cuDNN init failures, `torch.cuda.is_available() == False`).

If torch ever gets clobbered, restore with:

```bash
uv pip install --python .venv_isaacsim/bin/python --reinstall 'torch==2.7.0' \
  --index-url https://download.pytorch.org/whl/cu126
```

`nvidia-modelopt[torch]` from the Fast-FS `requirements.txt` is also off-limits for the same reason (its `[torch]` extra forces a recent torch).

### 2. Download model weights

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

## Inference paths

### PyTorch (bring-up / fallback)

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

### TensorRT (deployment)

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

### Measured throughput at 384x224, 4 iters, RTX 6000 Ada, FP16

| backend  | latency  | fps  | disparity match vs PyTorch              |
|----------|----------|------|------------------------------------------|
| PyTorch  | 25.94 ms | 39   | (reference)                              |
| **TRT**  | **3.87 ms** | **258** | mean abs 0.009 px, 0% pixels disagree > 1 px |

TRT is 6.7x faster and bit-equivalent. The team's 50 fps target (3090) maps to ~250 fps on the lab's RTX 6000 Ada — the resulting headroom lets the FS stage co-exist with the policy inference loop without stealing time from physics.

## Notes / gotchas

- `*.pth`, `*.onnx`, `*.engine` files in `third_party/Fast-FoundationStereo/weights/` are gitignored — regenerate from the steps above when re-cloning.
- ONNX export emits many `TracerWarning` lines for control flow that depends on tensor shapes; they are benign for a fixed-resolution engine (we bake the shape into the export) but invalidate the engine if you change `--height`/`--width`/`--valid_iters` — rebuild for each new resolution.
- TRT `enqueueV3` emits `Using default stream...` warnings on every call. Cosmetic only. For production deployment, switch to a non-default stream by patching `TrtRunner.forward` to pass an explicit `torch.cuda.Stream`.
- Engine files are GPU-architecture-specific. Engines built on RTX 6000 Ada (Ada Lovelace, sm_89) won't load on a 3090 (Ampere, sm_86) — rebuild per target GPU.
- The serialized `.pth` files in `weights/{model}/` are full `nn.Module` pickles, **not** state_dicts. `torch.load` needs `weights_only=False`. This is a security-relevant flag if you ever load checkpoints from untrusted sources.
