"""One-image SVD-XT pilot. Heavy dependencies are imported only after CLI parsing."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MODEL = "stabilityai/stable-video-diffusion-img2vid-xt"
WIDTH, HEIGHT, FRAMES = 1024, 576, 25


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_size(width, height):
    """Fit inside the canvas without cropping or changing aspect ratio."""
    if width <= 0 or height <= 0:
        raise ValueError("Invalid source dimensions")
    scale = min(WIDTH / width, HEIGHT / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Local JPG/JPEG")
    p.add_argument("--output-dir", type=Path, required=True, help="Must not exist")
    p.add_argument("--revision", required=True, help="Exact 40-character model commit SHA")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--motion-bucket-id", type=int, default=127)
    p.add_argument("--noise-aug-strength", type=float, default=0.02)
    p.add_argument("--conditioning-fps", type=int, default=7)
    p.add_argument("--fps", type=int, default=6, help="MP4 playback FPS; 25/6 = 4.17 seconds")
    p.add_argument("--min-guidance-scale", type=float, default=1.0)
    p.add_argument("--max-guidance-scale", type=float, default=3.0)
    p.add_argument("--decode-chunk-size", type=int, default=5)
    p.add_argument("--min-free-vram-gib", type=float, default=12.0)
    p.add_argument("--cpu-offload", action="store_true", help="Use system RAM to reduce VRAM use")
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/model_cache/svd_xt"))
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--check-only", action="store_true", help="Image/GPU validation only; no model download or generation")
    return p


def validate_args(args):
    if not re.fullmatch(r"[a-fA-F0-9]{40}", args.revision):
        raise ValueError("--revision must be an immutable 40-character model commit SHA")
    if not 0 <= args.seed < 2**63:
        raise ValueError("Seed must be in [0, 2**63)")
    if args.steps < 1 or not 0 <= args.motion_bucket_id <= 255:
        raise ValueError("Steps must be positive; motion bucket must be 0..255")
    if args.fps < 1 or args.conditioning_fps < 1 or not 1 <= args.decode_chunk_size <= FRAMES:
        raise ValueError("FPS must be positive; decode chunk must be 1..25")
    numeric = [args.noise_aug_strength, args.min_guidance_scale, args.max_guidance_scale, args.min_free_vram_gib]
    if not all(math.isfinite(x) and x >= 0 for x in numeric):
        raise ValueError("Noise, guidance and VRAM thresholds must be finite and nonnegative")
    if args.min_guidance_scale > args.max_guidance_scale:
        raise ValueError("Minimum guidance exceeds maximum")
    if args.input.suffix.lower() not in {".jpg", ".jpeg"} or not args.input.is_file():
        raise ValueError("Input must be an existing JPG/JPEG")
    if args.output_dir.exists():
        raise FileExistsError("Output directory already exists; choose a new run directory")
    if args.output_dir.resolve() == args.cache_dir.resolve() or args.cache_dir.resolve().is_relative_to(args.output_dir.resolve()):
        raise ValueError("Model cache must be outside the run output directory")


def conditioning_image(path):
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        if image.format != "JPEG":
            raise ValueError("File contents are not JPEG")
        image.verify()
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    size = fit_size(*rgb.size)
    resized = rgb.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (127, 127, 127))
    left, top = (WIDTH - size[0]) // 2, (HEIGHT - size[1]) // 2
    canvas.paste(resized, (left, top))
    return canvas, {"oriented_source_size": list(rgb.size), "resized_size": list(size),
                    "content_box_xyxy": [left, top, left + size[0], top + size[1]],
                    "padding_rgb": [127, 127, 127], "crop": False,
                    "exif_orientation_applied": True, "resampling": "PIL LANCZOS"}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    args = parser().parse_args(argv)
    validate_args(args)
    import torch
    image, preprocessing = conditioning_image(args.input)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this interpreter; verify driver and cu128 wheel")
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    capability = torch.cuda.get_device_capability(device)
    gpu = {"name": torch.cuda.get_device_name(device), "compute_capability": list(capability),
           "compiled_architectures": torch.cuda.get_arch_list(),
           "free_vram_gib": free / 2**30, "total_vram_gib": total / 2**30}
    if free < args.min_free_vram_gib * 2**30:
        raise RuntimeError(f"Insufficient free VRAM: {free / 2**30:.2f} GiB; requested floor {args.min_free_vram_gib}")
    # Also exercise an actual CUDA kernel: availability alone does not prove wheel/GPU compatibility.
    probe = torch.ones((8, 8), device=device)
    assert (probe @ probe).sum().item() == 512
    del probe
    torch.cuda.synchronize()
    if args.check_only:
        print(json.dumps({"check_only": True, "gpu": gpu, "preprocessing": preprocessing,
                          "torch": torch.__version__, "cuda_runtime": torch.version.cuda}, indent=2))
        return

    import numpy as np
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from huggingface_hub import snapshot_download
    from diffusers import StableVideoDiffusionPipeline

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    # Planning floor, not a measured peak: cache/env space may be on separate drives.
    if shutil.disk_usage(args.output_dir.parent).free < 2 * 2**30:
        raise RuntimeError("Need at least 2 GiB free on output drive")
    if not args.local_files_only and shutil.disk_usage(args.cache_dir).free < 30 * 2**30:
        raise RuntimeError("Reserve at least 30 GiB free on model-cache drive before download")
    args.output_dir.mkdir(exist_ok=False)
    started = time.perf_counter()
    metadata = {
        "status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "revision": args.revision.lower(), "variant": "fp16", "dtype": "float16",
        "source_sha256": sha256(args.input), "script_sha256": sha256(__file__),
        "preprocessing": preprocessing, "gpu": gpu,
        "generation": {"frames": FRAMES, "width": WIDTH, "height": HEIGHT, "seed": args.seed,
                       "steps": args.steps, "motion_bucket_id": args.motion_bucket_id,
                       "noise_aug_strength": args.noise_aug_strength, "conditioning_fps": args.conditioning_fps,
                       "playback_fps": args.fps, "duration_seconds": FRAMES / args.fps,
                       "min_guidance_scale": args.min_guidance_scale, "max_guidance_scale": args.max_guidance_scale,
                       "decode_chunk_size": args.decode_chunk_size, "cpu_offload": args.cpu_offload},
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "python": sys.version, "cuda_runtime": torch.version.cuda,
        "audit_status": "UNREVIEWED; hallucinated anatomy possible; not approved for training",
        "reproducibility": "Seed and pinned versions/revision recorded; bitwise cross-device reproducibility is not guaranteed."
    }
    write_json(args.output_dir / "request.json", metadata)
    try:
        snapshot = Path(snapshot_download(
            MODEL, revision=args.revision.lower(), cache_dir=str(args.cache_dir),
            allow_patterns=["*.json", "*.txt", "*.fp16.safetensors"],
            local_files_only=args.local_files_only))
        model_files = sorted(p for p in snapshot.rglob("*") if p.is_file())
        metadata["model_file_sha256"] = {p.relative_to(snapshot).as_posix(): sha256(p) for p in model_files}
        image.save(args.output_dir / "conditioning.png")  # RGB re-encode does not retain source EXIF.
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            str(snapshot), variant="fp16", torch_dtype=torch.float16,
            use_safetensors=True, local_files_only=True)
        pipe.unet.enable_forward_chunking()
        if args.cpu_offload:
            pipe.enable_model_cpu_offload(gpu_id=0)
        else:
            pipe.to(device)
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            frames = pipe(image=image, height=HEIGHT, width=WIDTH, num_frames=FRAMES,
                          num_inference_steps=args.steps, fps=args.conditioning_fps,
                          motion_bucket_id=args.motion_bucket_id, noise_aug_strength=args.noise_aug_strength,
                          min_guidance_scale=args.min_guidance_scale, max_guidance_scale=args.max_guidance_scale,
                          decode_chunk_size=args.decode_chunk_size, generator=generator, output_type="pil").frames[0]
        torch.cuda.synchronize()
        metadata["inference_seconds"] = time.perf_counter() - inference_started
        if len(frames) != FRAMES or any(frame.size != (WIDTH, HEIGHT) for frame in frames):
            raise RuntimeError("Unexpected generated frame count or dimensions")
        video = args.output_dir / "svd_xt.mp4"
        with imageio.get_writer(str(video), fps=args.fps, codec="libx264", pixelformat="yuv420p",
                                macro_block_size=16, ffmpeg_log_level="error") as writer:
            for frame in frames:
                array = np.asarray(frame.convert("RGB"))
                if not np.isfinite(array).all():
                    raise RuntimeError("Nonfinite frame")
                writer.append_data(array)
        # Verify the encoded file rather than trusting encoder metadata alone.
        with imageio.get_reader(str(video)) as reader:
            encoded_fps = reader.get_meta_data()["fps"]
            count = 0
            for array in reader:
                if array.shape != (HEIGHT, WIDTH, 3):
                    raise RuntimeError("Encoded frame dimensions changed")
                count += 1
        if count != FRAMES or abs(encoded_fps - args.fps) > 0.01:
            raise RuntimeError("Encoded video frame count/FPS mismatch")
        sheet = Image.new("RGB", (5 * 256, 5 * 166 + 28), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 6), "SVD-XT | UNREVIEWED synthetic anatomy | all 25 frames", fill="black")
        for i, frame in enumerate(frames):
            x, y = i % 5 * 256, 28 + i // 5 * 166
            draw.text((x + 4, y + 2), f"Frame {i + 1:02d}", fill="black")
            sheet.paste(frame.resize((256, 144), Image.Resampling.LANCZOS), (x, y + 22))
        sheet.save(args.output_dir / "contact_sheet.png")
        metadata.update(status="completed", encoded_frames=count, encoded_fps=encoded_fps,
                        elapsed_seconds=time.perf_counter() - started,
                        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                        peak_cuda_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
        metadata["artifact_sha256"] = {name: sha256(args.output_dir / name)
                                      for name in ["conditioning.png", "svd_xt.mp4", "contact_sheet.png", "request.json"]}
        write_json(args.output_dir / "generation_metadata.json", metadata)
        print("Completed: svd_xt.mp4, contact_sheet.png, generation_metadata.json (visual audit required)")
    except Exception as error:
        # Keep partial output; never retry silently or overwrite a failed run.
        write_json(args.output_dir / "failure.json", {"status": "failed", "error_type": type(error).__name__,
                                                     "elapsed_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
