"""Compare depth parallax and planar handheld motion on the exact five prior yaw sources."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mendeley_camera_motion as base

AMPLITUDES = (0.0005, 0.001, 0.002, 0.004)


def load_sources(yaw_root, dataset_root):
    """Never select replacements. All source hashes must match saved baseline evidence."""
    sources = []
    for i in range(1, 6):
        folder = yaw_root / f"photo_{i:02d}"
        evidence = folder / "metadata.json"
        if not evidence.is_file():
            evidence = folder / "request.json"
        if not evidence.is_file():
            raise ValueError(f"Missing prior source evidence for photo_{i:02d}; no reselection allowed")
        record = json.loads(evidence.read_text(encoding="utf-8"))
        relative = record["source_relative_path"].replace("\\", "/")
        path = (dataset_root / relative).resolve()
        if not path.is_relative_to(dataset_root.resolve()) or not path.is_file():
            raise ValueError("Baseline source must exist inside the supplied dataset root")
        if base.sha(path) != record["source_sha256"]:
            raise ValueError("Baseline source hash mismatch; refusing replacement")
        sources.append({"path": path, "relative": relative, "sha256": record["source_sha256"],
                        "baseline": record, "baseline_file": evidence, "baseline_evidence_sha256": base.sha(evidence)})
    if len({s["sha256"] for s in sources}) != 5:
        raise ValueError("Baseline must identify five byte-distinct sources")
    revisions = {s["baseline"]["model_revision"] for s in sources}
    if len(revisions) != 1 or not re.fullmatch(r"[0-9a-f]{40}", next(iter(revisions))):
        raise ValueError("All five baselines must record the same immutable model revision")
    return sources, next(iter(revisions))


def smooth(index):
    t = index / (base.FRAMES - 1)
    return t*t*(3-2*t)


def parallax_project(z, translation):
    h, w = z.shape
    v, u = np.indices(z.shape, dtype=np.float32)
    # Camera moves laterally by translation in bounded relative-Z units.
    return u - w*translation/z, v, z


def parallax_metrics(z, translation):
    displacement = -z.shape[1]*translation/z
    flat = -z.shape[1]*translation/np.median(z)
    near, far = np.percentile(z, [10, 90])
    return {"depth_vs_flat_p95_px": float(np.percentile(np.abs(displacement-flat), 95)),
            "near_far_differential_px": float(abs(z.shape[1]*translation*(1/near-1/far))),
            "total_displacement_p95_px": float(np.percentile(np.abs(displacement), 95))}


def handheld_homography(shape, index):
    """Single smooth planar camera drift; no per-tooth or independently varying deformation."""
    h, w = shape
    f, cx, cy = float(w), (w-1)/2, (h-1)/2
    s = smooth(index)
    zoom = 1 + .012*s
    roll, pitch = math.radians(-.22*s), math.radians(.18*s)
    rz = np.array([[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.]])
    scale = np.array([[zoom, 0, cx*(1-zoom)], [0, zoom, cy*(1-zoom)], [0, 0, 1.]])
    drift = np.array([[1, 0, .0015*w*s], [0, 1, -.001*h*s], [0, 0, 1.]])
    return drift @ scale @ k @ rz @ rx @ np.linalg.inv(k), {
        "zoom": zoom, "roll_degrees": math.degrees(roll), "pitch_degrees": math.degrees(pitch),
        "translation_pixels": [.0015*w*s, -.001*h*s]}


def render(rgb, z, mode, index, amplitude=0):
    if mode == "parallax":
        translation = amplitude*(2*smooth(index)-1)
        pu, pv, zz = parallax_project(z, translation)
        frame, holes = base.splat(rgb, pu, pv, zz)
        pose = dict(translation_relative_units=translation, **parallax_metrics(z, translation))
    else:
        import cv2
        h, w = rgb.shape[:2]
        matrix, pose = handheld_homography((h, w), index)
        v, u = np.indices((h, w), dtype=np.float64)
        points = matrix @ np.stack([u.ravel(), v.ravel(), np.ones(h*w)])
        pu, pv = (points[0]/points[2]).reshape(h, w), (points[1]/points[2]).reshape(h, w)
        if not np.isfinite(points).all() or (points[2] <= 0).any():
            raise ValueError("Invalid homography geometry")
        frame = cv2.warpPerspective(rgb, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(127,127,127))
        coverage = cv2.warpPerspective(np.ones((h,w), np.float32), matrix, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        holes = coverage < .999
        frame[holes] = 127
    metrics = base.artifact_metrics(pu, pv, holes)
    metrics.update(frame=index+1, pose=pose, failed_checks=base.failures(metrics))
    return frame, metrics


def evaluate_amplitudes(rgb, z, amplitudes=AMPLITUDES):
    trials = []
    for amplitude in amplitudes:
        stats = [render(rgb, z, "parallax", i, amplitude)[1] for i in range(base.FRAMES)]
        trials.append({"amplitude_relative_units": amplitude,
                       "pass_artifacts": not any(s["failed_checks"] for s in stats),
                       "max_hole_fraction": max(s["hole_fraction"] for s in stats),
                       "max_interior_hole_fraction": max(s["interior_hole_fraction"] for s in stats),
                       "max_foldover_fraction": max(s["foldover_fraction"] for s in stats),
                       "max_extreme_stretch_fraction": max(s["extreme_stretch_fraction"] for s in stats),
                       "max_depth_vs_flat_p95_px": max(s["pose"]["depth_vs_flat_p95_px"] for s in stats),
                       "max_near_far_differential_px": max(s["pose"]["near_far_differential_px"] for s in stats),
                       "per_frame": stats})
    passing = [t["amplitude_relative_units"] for t in trials if t["pass_artifacts"]]
    # If all fail, render the smallest trial as explicitly failed diagnostic evidence.
    return max(passing) if passing else min(amplitudes), trials, bool(passing)


def encode_experiment(rgb, z, folder, mode, common, amplitude=0, eligible=True, trials=None):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    h, w = rgb.shape[:2]
    sheet = Image.new("RGB", (1440, 1280), "white"); draw = ImageDraw.Draw(sheet)
    stats, changes, previous = [], [], None
    with imageio.get_writer(str(folder/"video.mp4"), fps=base.FPS, codec="libx264", pixelformat="yuv420p", macro_block_size=1, ffmpeg_log_level="error") as writer:
        for i in range(base.FRAMES):
            frame, metrics = render(rgb, z, mode, i, amplitude)
            stats.append(metrics)
            if previous is not None:
                changes.append(float(np.abs(frame.astype(np.float32)-previous).mean()))
            previous = frame.astype(np.float32)
            writer.append_data(np.pad(frame, ((0,h%2),(0,w%2),(0,0)), constant_values=127))
            thumb = Image.fromarray(frame); thumb.thumbnail((240,132), Image.Resampling.LANCZOS)
            x,y = i%6*240, i//6*160
            draw.text((x+3,y+3), f'{mode} {i+1:02d}', fill="black")
            sheet.paste(thumb, (x+(240-thumb.width)//2,y+24))
    sheet.save(folder/"contact_sheet.png")
    with imageio.get_reader(str(folder/"video.mp4")) as reader:
        fps = reader.get_meta_data()["fps"]; count = 0
        for frame in reader:
            if frame.shape != (h+h%2,w+w%2,3):
                raise ValueError("Encoded dimensions mismatch")
            count += 1
    if count != 48 or abs(fps-12) > .01:
        raise ValueError("Encoded frame-count/FPS mismatch")
    magnitude = max((s["pose"].get("depth_vs_flat_p95_px", 0) for s in stats), default=0)
    differential = max((s["pose"].get("near_far_differential_px", 0) for s in stats), default=0)
    visibility_ok = mode != "parallax" or (magnitude >= .25 and differential >= .5)
    ok = eligible and visibility_ok and not any(s["failed_checks"] for s in stats)
    metadata = dict(common, experiment=mode, frames=48, fps=12, duration_seconds=4,
                    source_size=[w,h], encoded_size=[w+w%2,h+h%2], odd_edge_padding=[w%2,h%2],
                    amplitude_relative_units=amplitude if mode=="parallax" else None,
                    candidate_trials=trials, per_frame=stats, consecutive_mae_0_255=changes,
                    parallax_visibility_gate_pass=visibility_ok, max_depth_vs_flat_p95_px=magnitude,
                    max_near_far_differential_px=differential,
                    checks_status="PASS_NUMERICAL_ONLY" if ok else "FAIL",
                    max_hole_fraction=max(s["hole_fraction"] for s in stats),
                    max_interior_hole_fraction=max(s["interior_hole_fraction"] for s in stats),
                    visual_audit="REQUIRED; no anatomical fidelity claim",
                    artifacts={p.name: base.sha(p) for p in folder.iterdir() if p.is_file()})
    base.save(folder/"metadata.json",metadata)
    return metadata


def estimate_depth(image, processor, model, device):
    import cv2
    import torch
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        prediction = model(**inputs).predicted_depth
        raw = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=(image.height,image.width), mode="bicubic",align_corners=False)[0,0].cpu().numpy()
    if not np.isfinite(raw).all() or raw.std()<1e-5:
        raise ValueError("Invalid learned depth")
    lo,hi = np.percentile(raw,[2,98])
    if hi<=lo:
        raise ValueError("Degenerate depth percentiles")
    disparity = cv2.bilateralFilter(np.clip((raw-lo)/(hi-lo),0,1).astype(np.float32),5,.05,3)
    return raw,disparity,1/(.8+.4*disparity)


def write_report(output, records):
    lines = ["# Parallax / handheld / prior yaw comparison", "",
             "Numerical results only. Visual inspection is mandatory. Differences in motion magnitude and resampling prevent a causal method ranking.", "",
             "| Photo | Mode | Status | Max holes | Max interior holes | Depth-vs-flat p95 px | Near/far differential px |", "|---|---|---|---:|---:|---:|---:|"]
    for row in records:
        for mode in ["yaw","parallax","handheld"]:
            m = row.get(mode,{})
            if not m or "per_frame" not in m:
                lines.append(f'| {row["photo"]} | {mode} | {m.get("checks_status","UNAVAILABLE")} | n/a | n/a | n/a | n/a |'); continue
            stats=m["per_frame"]
            hole=max(s["hole_fraction"] for s in stats)
            interior=max(s["interior_hole_fraction"] for s in stats)
            # Pure yaw and planar homography have zero depth-dependent parallax by construction.
            lines.append(f'| {row["photo"]} | {mode} | {m.get("checks_status","UNKNOWN")} | {hole:.6%} | {interior:.6%} | {m.get("max_depth_vs_flat_p95_px",0):.4f} | {m.get("max_near_far_differential_px",0):.4f} |')
    lines += ["", "Parallax metrics are model-relative projected displacement versus a median-depth flat plane, not measured physical tooth motion. Near/far uses the depth 10th/90th percentiles. All candidate and per-frame foldover/stretch metrics are in each metadata.json. Missing baseline metrics are explicitly unavailable; no historical result was fabricated.", "", "No new surface is reconstructed. Gray uncovered regions are not filled. Reject failed or visually distorted clips before training; group all derivatives of a source in one split."]
    with (output/"comparison_report.md").open("x",encoding="utf-8") as f:f.write("\n".join(lines)+"\n")
    base.save(output/"comparison.json",records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--yaw-run",type=Path,default=Path("outputs/runs/mendeley_camera_yaw_02"))
    p.add_argument("--parallax-output",type=Path,default=Path("outputs/runs/mendeley_parallax_01"))
    p.add_argument("--handheld-output",type=Path,default=Path("outputs/runs/mendeley_handheld_01"))
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--cache-dir",type=Path,default=Path("outputs/model_cache/depth_anything_v2_small"))
    args=p.parse_args()
    a,b=args.parallax_output.resolve(),args.handheld_output.resolve()
    if a.exists() or b.exists() or a==b or a.is_relative_to(b) or b.is_relative_to(a):
        raise ValueError("Both output roots must be new and separate")
    for output in [a,b]:
        for protected in [args.yaw_run.resolve(),args.dataset_root.resolve()]:
            if output==protected or output.is_relative_to(protected) or protected.is_relative_to(output):
                raise ValueError("Output roots must be separate from baseline evidence and original dataset")
    sources,revision=load_sources(args.yaw_run,args.dataset_root.resolve())
    import torch
    import cv2
    from PIL import Image,ImageOps
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor,AutoModelForDepthEstimation
    if args.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4);torch.manual_seed(42)
    snapshot=Path(snapshot_download(base.MODEL,revision=revision,cache_dir=str(args.cache_dir),allow_patterns=["*.json","model.safetensors"]))
    processor=AutoImageProcessor.from_pretrained(snapshot,local_files_only=True,use_fast=False)
    model=AutoModelForDepthEstimation.from_pretrained(snapshot,local_files_only=True,use_safetensors=True).to(args.device).eval().requires_grad_(False)
    model_hashes={p.relative_to(snapshot).as_posix():base.sha(p) for p in snapshot.rglob("*") if p.is_file()}
    for output in [a,b]:output.mkdir(parents=True,exist_ok=False)
    records=[]
    for i,source in enumerate(sources,1):
        label=f"photo_{i:02d}"
        baseline=source["baseline"]
        common={"source_relative_path":source["relative"],"source_sha256":source["sha256"],
                "baseline_evidence_sha256":source["baseline_evidence_sha256"],"model":base.MODEL,
                "model_revision":revision,"model_file_sha256":model_hashes,
                "script_sha256":base.sha(__file__),"renderer_sha256":base.sha(base.__file__),
                "device":args.device,"dtype":"float32","packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},
                "thresholds":{"max_holes":.025,"max_interior_holes":.001,"max_foldovers":0,"max_extreme_stretch":.001,"local_scale_range":[.85,1.15]},
                "parallax_visibility_thresholds":{"depth_vs_flat_p95_px":.25,"near_far_differential_px":.5},
                "geometry":"bounded monocular relative Z; focal length = original width; no calibrated geometry",
                "hole_handling":"neutral gray; no inpainting or generative filling"}
        row={"photo":label,"yaw":baseline}
        for root in [a,b]:
            (root/label).mkdir(exist_ok=False);base.save(root/label/"request.json",common)
        try:
            with Image.open(source["path"]) as original:
                if original.format!="JPEG":raise ValueError("Expected JPEG")
                image=ImageOps.exif_transpose(original).convert("RGB")
            if min(image.size)<64:raise ValueError("Source too small")
            rgb=np.asarray(image)
        except Exception as error:
            for mode,root in [("parallax",a),("handheld",b)]:
                row[mode]={"checks_status":"ERROR","error_type":type(error).__name__}
                base.save(root/label/"failure.json",row[mode])
            records.append(row);continue
        # Independent branches: a depth failure must not prevent planar handheld rendering.
        for mode,root in [("parallax",a),("handheld",b)]:
            started=time.perf_counter()
            try:
                if mode=="parallax":
                    raw,disparity,z=estimate_depth(image,processor,model,args.device)
                    np.savez_compressed(root/label/"depth.npz",predicted_inverse_depth=raw,normalized_inverse_depth=disparity,relative_z=z)
                    Image.fromarray(cv2.cvtColor(cv2.applyColorMap(np.uint8(disparity*255),cv2.COLORMAP_INFERNO),cv2.COLOR_BGR2RGB)).save(root/label/"depth.png")
                    amplitude,trials,eligible=evaluate_amplitudes(rgb,z)
                    row[mode]=encode_experiment(rgb,z,root/label,mode,common,amplitude,eligible,trials)
                else:
                    row[mode]=encode_experiment(rgb,None,root/label,mode,common)
                base.save(root/label/"timing.json",{"elapsed_seconds":time.perf_counter()-started})
            except Exception as error:
                row[mode]={"checks_status":"ERROR","error_type":type(error).__name__,"message":str(error)}
                base.save(root/label/"failure.json",row[mode])
            print(label,mode,row[mode]["checks_status"],flush=True)
        records.append(row)
    for output in [a,b]:write_report(output,records)
    if any(row[mode]["checks_status"]!="PASS_NUMERICAL_ONLY" for row in records for mode in ["parallax","handheld"]):raise SystemExit(2)


if __name__=="__main__":main()
