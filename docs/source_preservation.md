# Preserved research tools

These sources supplement the committed VideoSwin training/evaluation baseline. No new scientific validation or results are claimed. No training, inference, or data processing was performed during preservation.

## Dependencies and local inputs

Use the baseline requirements for dataset audits and VideoSwin explanations. The legacy NPZ builder additionally requires tqdm. Method A needs NumPy, Pillow and OpenCV. Method B additionally needs psutil, PyTorch, Transformers and its model-loading dependencies. The historical successful depth environment used CPU torch 2.5.1 and transformers 4.46.3; keep it separate from VideoSwin. No fresh environment installation was validated.

Copy datasets, CSV labels, the completed Excel audit, checkpoints, manifests and original run evidence separately. They remain ignored and are not distributed with this source. No source here grants dataset/annotation redistribution rights. prepare_quadrant_metadata.py is deliberately excluded because it embeds annotations.

## Invocation and portability

Existing preserved scripts keep their established relative data/output layout. Direct script execution now starts from the repository root and resolves dataset imports. They retain fixed cases and filenames: inspect destinations before running. Several scripts still perform work on import; do not import them as libraries or collect the diagnostic test_*.py files as automated tests. Windows mapping inspection requires ctypes.windll and Windows Explorer sorting; no cross-platform replacement was substituted. The smoke diagnostic may download pretrained weights if absent.

Copied scripts accept explicit paths:

- src/explainability/run_norm_cam.py: --audit-dir --checkpoint --output-dir. Requires CUDA and original audit JSON/per-clip predictions. Checkpoint hashes, epoch and prediction/reconstruction gates remain enforced.
- src/synthetic/run_method_a.py: --manifest --archive --output-dir.
- src/synthetic/run_method_b.py: --manifest --archive --output-dir --method-a-dir --model-dir --download-record. Reads Method A video/metrics from its input directory; generates Method B into a separate new directory. Depth model loading remains offline, CPU float32, with recorded file-hash checks. Both methods remain MD01-only, 16 frames at 8 FPS.

For these three copied tools, --output-dir must not exist. They retain .git/index provenance checks and expect a normal Git checkout, not a source archive or linked worktree. Input hashes and numerical checks are unchanged. Paths printed in newly generated provenance reflect the active workstation. Historical report text embedded in the scripts describes the original pilot and is not a generalized installation guide.

The legacy NPZ builder has a corrected repository root but still needs segments_with_uuid_92.csv and test_multiclip_5x16_metadata_27.csv, which are absent locally. It includes held-out test processing and is not part of the supported training pipeline.

## Limitations

Attention-rollout scripts and test_gradient_attention.py are explicitly unvalidated prototypes. They rely on private Torchvision APIs; the hook-based gradient diagnostic does not establish attribution correctness. Window attention and captured QKV tensors are not full rollout or class-specific explanations. Older PowerShell launchers are marked legacy: they retain historical hashes, paths and (for the follow-up launcher) expired deadlines and process-management behavior. Do not launch them as workstation setup commands.

Many older tools overwrite fixed output filenames; this preservation pass does not redesign their workflows. create_manual_audit.py creates a blank CSV, not the authored Excel decisions. Binary metadata preparation retains its historical cleaning/merge behavior; validate audit completeness before any future regeneration. Scientific algorithms, labels and historical results were not changed.

## Historical originals

Copies below preserve source without removing or rewriting original run files. SHA-256 values identify the original bytes; portable copies intentionally have different hashes.

- `data/build_gt_npz.py` -> `src/data/legacy/build_gt_npz.py`; original SHA-256 `1d870badf740361e31dc22f73eda67ca38f98a31747a83db462f7c3b0dd6715a`.
- `outputs/runs/prediction_agreement_audit_2026-10-15_01/explainability_original81_epoch01_norm_cam/run_norm_cam.py` -> `src/explainability/run_norm_cam.py`; original SHA-256 `29fad748dd7b12f90c373a3704d375308fc4b3a434f100f020420f83a5c9f7ea`.
- `outputs/runs/synthetic_dental_md01_2026-10-20_01/run_method_a.py` -> `src/synthetic/run_method_a.py`; original SHA-256 `7d6f55c42a0fa43ad6ead67391ac27be00b6d77df5a69f74a850382f39685535`.
- `outputs/runs/synthetic_dental_md01_2026-10-20_01/run_method_b.py` -> `src/synthetic/run_method_b.py`; original SHA-256 `78e3585b01dd50927bdb1df99b238f26da99ca4cd457d4d8c9e1abc1530df587`.

## Explicit preservation file list

- `scripts/run_binary_comparison.ps1`
- `scripts/run_followup_after_overnight.ps1`
- `scripts/run_overnight_supervisor.ps1`
- `src/analyze_gradcam_temporal.py`
- `src/compare_gradcam_cases.py`
- `src/compare_gradient_attention_rollout.py`
- `src/data/analyze_data_quality.py`
- `src/data/audit_data_quality.py`
- `src/data/check_mapping.py`
- `src/data/create_contact_sheets.py`
- `src/data/create_manual_audit.py`
- `src/data/prepare_binary_metadata.py`
- `src/data/test_dataset.py`
- `src/explainability/gradcam_style_all_classes.py`
- `src/explainability/inspect_videoswin_attention.py`
- `src/explainability/select_val_examples.py`
- `src/explainability/visualize_window_attention.py`
- `src/gradcam_videoswin.py`
- `src/gradient_attention_rollout.py`
- `src/inspect_gradcam_layer.py`
- `src/models/smoke_test_videoswin.py`
- `src/summarize_gradcam_validation.py`
- `src/test_gradient_attention.py`
- `src/data/legacy/build_gt_npz.py`
- `src/explainability/run_norm_cam.py`
- `src/synthetic/run_method_a.py`
- `src/synthetic/run_method_b.py`
