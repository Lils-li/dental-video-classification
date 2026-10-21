# Five-photograph dental yaw pilot (PC 2)

This separate runner follows Method B's fixed learned inverse depth, bounded Z, bilinear splatting and 2% soft z-buffer. The original Method B and previous outputs are unchanged. It produces five clips from five distinct view-class directories using deterministic midpoint selection with byte-hash deduplication. Preferred classes are Lower Front, Upper Front, Lower Left, Upper Right, Lower Occlusal; other classes are deterministic fallbacks. It fails if five distinct classes/photographs are unavailable. Actual choices and hashes are recorded on PC 2, not bundled in Git.

Motion is one smooth sinusoidal yaw cycle, bounded by +/-0.5 degrees, over 48 frames at 12 FPS (4 seconds). There is no translation, zoom, pitch, roll, frame jitter or random camera motion. Original pixel dimensions/aspect ratio are retained after EXIF orientation; no crop or image stretch occurs. An odd width/height gets a single neutral-gray right/bottom padding pixel for H.264, recorded in metadata. Contact-sheet thumbnails preserve aspect ratio too.

The rotation is around the optical center, so projection coordinates for a given source ray are independent of depth: this is a physical limitation of pure rotational camera motion. The code still estimates one fixed depth map per photograph and uses it for 3D projection/z-buffer visibility, but does not claim translational depth parallax or reconstructed hidden surfaces. Introducing parallax would require translation/orbiting, outside this yaw-only pilot.

## PC 2 environment

Use the isolated Python 3.13 environment from docs/svd_xt_pilot_windows.md (PyTorch 2.11.0/cu128, Transformers 4.55.4, Pillow, NumPy, ImageIO/FFmpeg and Hugging Face Hub). Add the pinned OpenCV dependency in that environment only:

```powershell
& "$env:USERPROFILE\.venvs\dental_svd_xt\Scripts\python.exe" -m pip install -r .\requirements-camera-motion.txt
& "$env:USERPROFILE\.venvs\dental_svd_xt\Scripts\python.exe" -m pip check
```

Stop on installation errors. This does not modify the base environment. If that isolated environment has not been created, follow the SVD guide's environment setup first; SVD weights/generation are not needed. Initial execution downloads only Depth Anything V2 Small to the ignored cache, resolves its immutable revision once, and records model-file hashes. For repeat experiments supply --revision with the SHA from a previous metadata.json. There is no model download or inference on PC 1.

After pulling the approved commit on PC 2, run all five experiments from the repository root with this one command:

```powershell
& "$env:USERPROFILE\.venvs\dental_svd_xt\Scripts\python.exe" .\src\synthetic\mendeley_camera_motion.py --dataset-root 'C:\Users\ESAI406_student2\Mendeley dataset 19.4.2024\Teeth or Dental Image Dataset' --output-dir .\outputs\runs\mendeley_camera_yaw_01 --device cuda
```

Choose a new output root for every attempt; no automatic retries/overwrite. CPU depth inference is available with --device cpu. Rasterization remains CPU NumPy; large images may take substantial time/RAM. Runtime/VRAM fit on PC 2 are unmeasured. Default processor settings preserve aspect ratio and learned depth is interpolated to the exact original dimensions, robustly normalized at the 2nd/98th percentiles and lightly bilateral-smoothed, as in Method B. The model is loaded once; there is one depth-network forward pass per photo, no per-frame estimation.

Each photo_01..photo_05 folder contains camera_yaw.mp4, all-48-frame contact_sheet.png, depth.png, raw/normalized depth.npz and metadata.json with source hash/path, model revision, source dimensions, poses, package versions, artifact hashes and per-frame checks. summary.json reports all five; errors retain failure.json and continue to the next photo. Error runs may have incomplete artifacts; no replacement photograph or motion is silently substituted. Exit code 2 means at least one numerical failure/error. The actual encoded MP4 is decoded to verify dimensions, FPS and all 48 frames.

## Artifacts and anatomy audit

No diffusion model, inpainting or generative filling is used. Only source RGB samples are splatted; uncovered pixels stay gray. Sampling/blending/encoding can still soften, distort or mix tooth boundaries, and the learned depth can be wrong. No guarantee of preserved anatomy or clinical fidelity follows from using source pixels. Gray borders are preferred to inventing hidden teeth. No cropping/zoom conceals missing coverage.

Per-frame gates: total uncovered area <=2.5%; interior uncovered area <=0.1% (interior excludes a 2%-of-max-dimension edge band); zero numerical foldovers; <=0.1% pixels outside local directional scales [0.85,1.15]. Jacobian singular values quantify stretch. Border holes are expected from yaw without zoom. These are transparent engineering thresholds, not clinically validated acceptance criteria. Any violation flags FAIL but retains the MP4 for diagnosis. Numeric PASS is explicitly PASS_NUMERICAL_ONLY, never training approval. Consecutive-frame differences are recorded for temporal inspection.

Inspect all full-resolution frames and playback against the source: tooth count, identity, contours, restorations, gingiva, occlusion, view label, missing regions, temporal changes and compression. Review contact sheets and depth maps too. Do not train on failed or unreviewed videos; keep all derivatives from one source in the same data split. Outputs, source images and model weights are ignored and must not be committed.

Reference: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf . This model predicts relative depth, not calibrated dental geometry. Implementation validation on PC 1 covers synthetic arrays only, not dental data, inference or PC 2 runtime.
