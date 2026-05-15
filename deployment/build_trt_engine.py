"""ONNX -> TRT engine via the TensorRT Python API (no trtexec binary required).

Used to build the Fast-FoundationStereo `feature_runner.engine` +
`post_runner.engine` pair from the ONNX files emitted by
`third_party/Fast-FoundationStereo/scripts/make_onnx.py`.

Example end-to-end (23-36-37, 4 iters, 384x224, FP16):

  ONNX_DIR=third_party/Fast-FoundationStereo/weights/23-36-37/onnx_384x224_iters4
  .venv_isaacsim/bin/python third_party/Fast-FoundationStereo/scripts/make_onnx.py \\
    --model_dir third_party/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth \\
    --save_path $ONNX_DIR \\
    --height 224 --width 384 --valid_iters 4 --max_disp 192
  .venv_isaacsim/bin/python deployment/build_trt_engine.py --onnx_dir $ONNX_DIR

Build times on RTX 6000 Ada: feature_runner ~2.5 min, post_runner ~10 min.
Resulting engines: ~20 MiB + ~14 MiB, TRT FP16 inference ~3.9 ms / frame
at 384x224 / 4 iters (~258 fps), bit-equivalent to PyTorch FP16 baseline.
"""
import argparse, sys, time
import tensorrt as trt

def build(onnx_path: str, engine_path: str, fp16: bool = True,
          workspace_gb: int = 4) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser  = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        raise RuntimeError(f"OnnxParser failed on {onnx_path}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 workspace_gb * (1 << 30))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    dt = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError(f"build_serialized_network returned None for {onnx_path}")
    buf = bytes(serialized)
    with open(engine_path, "wb") as f:
        f.write(buf)
    print(f"  built {engine_path}  ({len(buf)/1024/1024:.1f} MiB, {dt:.1f}s)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_dir", required=True)
    ap.add_argument("--fp16", type=int, default=1)
    args = ap.parse_args()
    for name in ("feature_runner", "post_runner"):
        print(f"=== building {name} ===")
        build(f"{args.onnx_dir}/{name}.onnx",
              f"{args.onnx_dir}/{name}.engine",
              fp16=bool(args.fp16))
