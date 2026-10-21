# Parallax versus handheld dental motion (PC 2)

This runner reads the five source_relative_path/source_sha256 entries from photo_01..photo_05/metadata.json in the prior mendeley_camera_yaw_02 run. If completed metadata is missing, it uses that same folder's request.json; baseline numeric comparison then reports unavailable. It never selects replacements. Missing paths, mismatched source bytes, duplicate images or inconsistent model revisions stop the run before generation. PC 1 cannot see this PC 2 run: the actual sources and results are verified/reported only when executed there.

Use the dedicated environment and requirements-camera-motion.txt from docs/mendeley_camera_motion.md. No new dependencies. Pull the new commit, then run this one PowerShell command from the repository root:

```powershell
& "$env:USERPROFILE\.venvs\dental_svd_xt\Scripts\python.exe" .\src\synthetic\mendeley_motion_comparison.py --dataset-root 'C:\Users\ESAI406_student2\Mendeley dataset 19.4.2024\Teeth or Dental Image Dataset' --yaw-run .\outputs\runs\mendeley_camera_yaw_02 --parallax-output .\outputs\runs\mendeley_parallax_01 --handheld-output .\outputs\runs\mendeley_handheld_01 --device cuda
```

Both output roots must not exist. No previous run, source image or original Method B is modified. The baseline model revision is reused exactly; required weights are downloaded into the ignored cache on PC 2 if absent. Model/source/script hashes and package versions are recorded. If the prior metadata has a different schema, stop and inspect it rather than running another source selection.

## Experiment A: genuine depth-dependent translation

One Depth Anything V2 Small forward pass per photograph estimates fixed inverse depth. The same percentile normalization, bilateral smoothing and bounded Z=1/(0.8+0.4*d) as the yaw pipeline are retained. The code translates the camera laterally: u'=u-f*t/Z, with f=original width. A smooth cubic trajectory moves once from -amplitude to +amplitude, with zero endpoint velocity; no repeated oscillation, zoom or shaking.

The candidate amplitude grid is 0.0005, 0.001, 0.002, 0.004 relative-Z units, not millimeters. All 48 frames are rendered/evaluated at every amplitude. Choose the largest passing all shared artifact gates; rerender that amplitude for encoding. These repeated full-resolution tests can be slow and memory intensive because splatting runs on CPU even with CUDA depth inference. There is no unrecorded coarse-to-fine search or automatic motion retry.

All candidate/per-frame results are saved, including failures. If no candidate passes, encode the smallest only as FAIL diagnostic evidence. Even if artifact gates pass, require peak depth-versus-median-plane p95 displacement >=0.25 px and near/far differential >=0.5 px; insufficient depth-dependent motion flags FAIL. These engineering visibility thresholds do not prove perceptual visibility or clinical acceptability. Never boost motion beyond the tested grid merely to make it look dramatic.

Metrics report actual computed projection displacement from the inferred depth: p95 deviation from a median-depth plane, near/far difference using 10th/90th depth percentiles, and p95 total image displacement. Units are original-resolution pixels. A flat-depth source yields no depth parallax. These metrics describe projected geometry, not measured motion of physical teeth or the effects of codec loss. Tooth-boundary depth errors, disocclusions and stretching remain possible. Uncovered pixels stay gray; no inpainting or generative filling is used.

## Experiment B: planar handheld-like drift

A single image-wide projective transform combines a slow 1.0 to 1.012 isotropic zoom, roll 0 to -0.22 degrees, pitch 0 to +0.18 degrees, horizontal drift 0 to 0.15% width and vertical drift 0 to -0.1% height. Each follows one smooth cubic progression, with no random jitter or periodic left-right reversal. This is a deliberately designed handheld-like path, not motion measured from a clinician. No tooth is deformed independently; all source pixels share the homography. Pitch is a planar approximation and provides no real novel viewpoint. There is no invented hidden surface; out-of-source regions remain gray.

Frames keep original EXIF-oriented dimensions and aspect ratio. Projective foreshortening is part of the motion simulation, not a stretch-to-fit resize. Slight zoom can move edge content outside the fixed canvas; original images remain unchanged. An odd right/bottom dimension receives one gray padding pixel for H.264, as in the previous pilot. Thumbnails preserve aspect ratio. Numerical checks and visual review must catch unacceptable global perspective distortion.

## Output and comparison

Each root contains photo_01..photo_05 with video.mp4 (48 frames, 12 FPS, 4 seconds), contact_sheet.png (all 48 frames), request.json, metadata.json and timing.json. Experiment A also saves depth.npz and depth.png. The actual encoded video is decoded to verify every frame's dimensions and FPS. The contact sheet is pre-encoding; inspect actual playback too.

Each root gets comparison_report.md and comparison.json against the saved yaw metrics on the identical sources. Pure yaw and planar handheld have zero depth-dependent parallax by construction; unavailable historical metrics are labeled unavailable. Comparisons are descriptive, not matched-motion causal ablations: trajectories, resampling and motion amplitudes differ. Maximum total/interior holes appear in the report; full per-frame stretch, foldover, temporal-change and displacement metrics are in metadata. The same yaw-pipeline gates apply: total holes <=2.5%, interior holes <=0.1%, zero foldovers, <=0.1% pixels outside local directional scales [0.85,1.15]. These are unvalidated engineering gates, not dental safety thresholds.

Failures are retained and flagged; exceptions create failure.json and other photographs/branches continue. Exit code 2 means at least one failed/error video. A model-load/source-validation failure stops the batch. Errors can leave incomplete videos, so exactly ten usable videos are conditional on successful execution and review; failures are never disguised as successful clips. No actual PC 2 result is claimed by this code delivery.

Monocular depth cannot recover occluded surfaces or guarantee anatomical fidelity. Review tooth count/identity, restorations, contours, gingiva, occlusion, view label, depth-boundary artifacts, gray holes, temporal smoothness and compression before VideoSwin training. Retain reviewer decisions and keep all derivatives of a source in one data split. No scientific ranking or training benefit is established by this pilot. Photos, weights and generated results remain ignored.
