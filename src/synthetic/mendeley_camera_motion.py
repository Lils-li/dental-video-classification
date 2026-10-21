"""Five-photo depth-based yaw pilot; no generative synthesis or hole inpainting."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
FRAMES, FPS = 48, 12


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def select_photos(root):
    """Deterministic midpoint candidate per class; byte-distinct across all five."""
    preferred = ["Lower Front", "Upper Front", "Lower Left", "Upper Right", "Lower Occlusal"]
    folders = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.casefold())
    folders.sort(key=lambda p: (preferred.index(p.name) if p.name in preferred else len(preferred), p.name.casefold()))
    chosen, seen = [], set()
    for folder in folders:
        candidates = sorted((p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg"}), key=lambda p: p.relative_to(root).as_posix().casefold())
        mid = len(candidates) // 2
        for p in candidates[mid:] + candidates[:mid]:
            digest = sha(p)
            if digest not in seen:
                chosen.append((p, digest)); seen.add(digest); break
        if len(chosen) == 5:
            return chosen
    raise ValueError("Need at least five classes containing byte-distinct JPEG photographs")


def yaw_degrees(index):
    # One smooth loop: zero -> positive -> zero -> negative -> zero, no jitter.
    return 0.5 * math.sin(2 * math.pi * index / (FRAMES - 1))


def project(z, angle):
    """Pure optical-center yaw: no translation, zoom, roll or pitch."""
    h, w = z.shape
    v, u = np.indices(z.shape, dtype=np.float32)
    f, cx, cy = float(w), (w - 1) / 2, (h - 1) / 2
    a = math.radians(angle)
    x, y = (u - cx) * z / f, (v - cy) * z / f
    xx = math.cos(a) * x + math.sin(a) * z
    zz = -math.sin(a) * x + math.cos(a) * z
    return f * xx / zz + cx, f * y / zz + cy, zz


def splat(rgb, pu, pv, z):
    """Method B bilinear splatting/2% z-buffer; holes remain neutral gray."""
    h, w = z.shape
    if not all(np.isfinite(a).all() for a in [pu, pv, z]) or (z <= 0).any():
        raise ValueError("Nonfinite or nonpositive projected geometry")
    x, y = np.floor(pu).astype(np.int32), np.floor(pv).astype(np.int32)
    fu, fv = pu - x, pv - y
    parts = []
    for dx, dy, wt in [(0, 0, (1-fu)*(1-fv)), (1, 0, fu*(1-fv)), (0, 1, (1-fu)*fv), (1, 1, fu*fv)]:
        xx, yy = x + dx, y + dy
        valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h) & (wt > 1e-6)
        parts.append(((yy[valid]*w+xx[valid]).ravel(), wt[valid], z[valid], rgb[valid].astype(np.float32)))
    nearest = np.full(h*w, np.inf, np.float32)
    for idx, wt, zz, col in parts:
        np.minimum.at(nearest, idx, zz)
    sumw, sumc = np.zeros(h*w), np.zeros((h*w, 3))
    for idx, wt, zz, col in parts:
        keep = zz <= nearest[idx]*1.02
        idx, wt, col = idx[keep], wt[keep], col[keep]
        sumw += np.bincount(idx, weights=wt, minlength=h*w)
        for channel in range(3):
            sumc[:, channel] += np.bincount(idx, weights=wt*col[:, channel], minlength=h*w)
    holes = (sumw < 1e-6).reshape(h, w)
    frame = np.clip(sumc / np.maximum(sumw[:, None], 1e-6), 0, 255).astype(np.uint8).reshape(h, w, 3)
    frame[holes] = 127
    return frame, holes


def artifact_metrics(pu, pv, holes):
    du_y, du_x = np.gradient(pu); dv_y, dv_x = np.gradient(pv)
    det = du_x*dv_y - du_y*dv_x
    # Singular values of the local 2x2 Jacobian quantify directional stretch.
    trace = du_x**2 + du_y**2 + dv_x**2 + dv_y**2
    disc = np.sqrt(np.maximum(trace**2 - 4*det**2, 0))
    largest = np.sqrt(np.maximum((trace + disc)/2, 0))
    smallest = np.sqrt(np.maximum((trace - disc)/2, 0))
    h, w = holes.shape
    band = max(2, int(math.ceil(max(h, w)*0.02)))
    interior = holes[band:-band, band:-band]
    return {"hole_fraction": float(holes.mean()),
            "interior_hole_fraction": float(interior.mean()) if interior.size else float(holes.mean()),
            "foldover_fraction": float((det <= 0).mean()),
            "extreme_stretch_fraction": float(((largest > 1.15) | (smallest < 0.85)).mean()),
            "max_local_scale": float(largest.max()), "min_local_scale": float(smallest.min())}


def failures(metrics):
    limits = {"hole_fraction": 0.025, "interior_hole_fraction": 0.001,
              "foldover_fraction": 0.0, "extreme_stretch_fraction": 0.001}
    return [key for key, limit in limits.items() if metrics[key] > limit]


def run_photo(path, digest, out, root, processor, model, device, common):
    import cv2
    import torch
    import imageio.v2 as imageio
    from PIL import Image, ImageOps, ImageDraw
    out.mkdir(exist_ok=False)
    save(out/"request.json", dict(common, source_relative_path=path.relative_to(root).as_posix(), source_sha256=digest, frames=FRAMES, fps=FPS, yaw_amplitude_degrees=.5))
    with Image.open(path) as original:
        if original.format != "JPEG":
            raise ValueError("Not JPEG content")
        image = ImageOps.exif_transpose(original).convert("RGB")
    w, h = image.size
    if min(w, h) < 64:
        raise ValueError("Photograph is too small")
    rgb = np.asarray(image)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        prediction = model(**inputs).predicted_depth
        raw = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False)[0, 0].cpu().numpy()
    if not np.isfinite(raw).all() or raw.std() < 1e-5:
        raise ValueError("Invalid/nonvarying learned depth")
    lo, hi = np.percentile(raw, [2, 98])
    if hi <= lo:
        raise ValueError("Degenerate depth percentiles")
    disparity = cv2.bilateralFilter(np.clip((raw-lo)/(hi-lo), 0, 1).astype(np.float32), 5, .05, 3)
    z = 1/(.8 + .4*disparity)  # Same bounded, nonmetric relative geometry as Method B.
    np.savez_compressed(out/"depth.npz", predicted_inverse_depth=raw, normalized_inverse_depth=disparity, relative_z=z)
    Image.fromarray(cv2.cvtColor(cv2.applyColorMap(np.uint8(disparity*255), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)).save(out/"depth.png")
    sheet = Image.new("RGB", (6*240, 8*160), "white"); draw = ImageDraw.Draw(sheet)
    stats, previous, changes = [], None, []
    video = out/"camera_yaw.mp4"
    # Pad only an odd right/bottom edge for H.264; never resize/stretch the source.
    with imageio.get_writer(str(video), fps=FPS, codec="libx264", pixelformat="yuv420p", macro_block_size=1, ffmpeg_log_level="error") as writer:
        for index in range(FRAMES):
            pu, pv, zz = project(z, yaw_degrees(index))
            frame, holes = splat(rgb, pu, pv, zz)
            metrics = artifact_metrics(pu, pv, holes)
            metrics.update(frame=index+1, yaw_degrees=yaw_degrees(index), failed_checks=failures(metrics))
            stats.append(metrics)
            if previous is not None:
                changes.append(float(np.abs(frame.astype(np.float32)-previous).mean()))
            previous = frame.astype(np.float32)
            encoded = np.pad(frame, ((0, h % 2), (0, w % 2), (0, 0)), constant_values=127)
            writer.append_data(encoded)
            thumb = Image.fromarray(frame); thumb.thumbnail((240, 132), Image.Resampling.LANCZOS)
            x, y = index % 6*240, index // 6*160
            draw.text((x+3, y+2), f'{index+1:02d} yaw={yaw_degrees(index):+.3f}', fill="black")
            sheet.paste(thumb, (x+(240-thumb.width)//2, y+24))
    sheet.save(out/"contact_sheet.png")
    count = 0
    with imageio.get_reader(str(video)) as reader:
        fps = reader.get_meta_data()["fps"]
        for frame in reader:
            if frame.shape != (h+h % 2, w+w % 2, 3):
                raise ValueError("Encoded dimensions mismatch")
            count += 1
    if count != FRAMES or abs(fps-FPS) > .01:
        raise ValueError("Encoded duration/frame-count check failed")
    failed = any(s["failed_checks"] for s in stats)
    metadata = dict(common, source_relative_path=path.relative_to(root).as_posix(), source_sha256=digest,
                    source_size=[w, h], encoded_size=[w+w % 2, h+h % 2], odd_edge_padding=[w % 2, h % 2],
                    fixed_depth_per_photo=True, frames=FRAMES, fps=FPS, duration_seconds=4,
                    geometry="Z=1/(0.8+0.4*d); assumed focal length = original width; optical-center yaw only",
                    translation=[0, 0, 0], zoom=1, yaw_amplitude_degrees=.5,
                    checks_status="FAIL" if failed else "PASS_NUMERICAL_ONLY", visual_audit="REQUIRED; anatomy not guaranteed",
                    hole_handling="neutral gray; no inpainting, diffusion, hidden-surface synthesis or source-edge cloning",
                    per_frame=stats, consecutive_mae_0_255=changes,
                    artifacts={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    save(out/"metadata.json", metadata)
    return {"folder": out.name, "checks_status": metadata["checks_status"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--revision", help="Immutable model SHA; otherwise resolve main once and record it")
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/model_cache/depth_anything_v2_small"))
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory; previous results are never overwritten")
    root = args.dataset_root.resolve()
    photos = select_photos(root)
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(4); torch.manual_seed(42)
    revision = args.revision or HfApi().model_info(MODEL).sha
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Model revision must be a full commit SHA")
    snapshot = Path(snapshot_download(MODEL, revision=revision, cache_dir=str(args.cache_dir), allow_patterns=["*.json", "model.safetensors"]))
    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True, use_fast=False)
    model = AutoModelForDepthEstimation.from_pretrained(snapshot, local_files_only=True, use_safetensors=True).to(args.device).eval().requires_grad_(False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    common = {"model": MODEL, "model_revision": revision, "model_file_sha256": {p.relative_to(snapshot).as_posix(): sha(p) for p in snapshot.rglob("*") if p.is_file()},
              "script_sha256": sha(__file__), "device": args.device, "dtype": "float32",
              "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
              "selection": "preferred five distinct classes, sorted midpoint candidate, unique source-byte hashes",
              "thresholds": {"max_holes": .025, "max_interior_holes": .001, "max_foldovers": 0, "max_extreme_stretch": .001, "local_scale_range": [.85, 1.15]}}
    results = []
    for i, (photo, digest) in enumerate(photos, 1):
        out = args.output_dir/f"photo_{i:02d}"
        started = time.perf_counter()
        try:
            result = run_photo(photo, digest, out, root, processor, model, args.device, common)
            result["elapsed_seconds"] = time.perf_counter()-started
        except Exception as error:
            out.mkdir(exist_ok=True)
            result = {"folder": out.name, "checks_status": "ERROR", "error_type": type(error).__name__, "message": str(error)}
            save(out/"failure.json", result)
        results.append(result)
        print(json.dumps(result), flush=True)
    save(args.output_dir/"summary.json", {"results": results, "visual_audit": "REQUIRED", "all_numerical_pass": all(r["checks_status"] == "PASS_NUMERICAL_ONLY" for r in results)})
    if any(r["checks_status"] != "PASS_NUMERICAL_ONLY" for r in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
