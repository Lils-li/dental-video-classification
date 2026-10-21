# Dental Video Classification

Video Swin Transformer Tiny (Torchvision `swin3d_t`) research on Vident-real intra-oral video segments. The project includes original quadrant classification (Q1-Q4), binary Upper/Lower comparisons, checkpoint verification, per-quadrant error analysis, and exploratory explanations.

Q1/Q2 map to Upper (0); Q3/Q4 map to Lower (1). The completed experiments below used training and validation data only. The held-out test set was not evaluated.

## Version-controlled baseline

- `src/data/dataset.py`: shared frame loading and exact temporal sampling.
- `src/run_binary_comparison.py`: controlled original-versus-cleaned balanced training.
- `src/run_followup_experiment.py`: corrected weighted cross-entropy and partial fine-tuning.
- `src/overnight_binary_analysis.py`: preflight, metrics, per-quadrant analysis, exploratory occlusion, and reporting.
- `src/evaluate_fourclass_as_binary.py`: original four-class checkpoint evaluation and comparison.
- `src/audit_binary_pipeline.py`, `src/audit_prediction_agreement.py`: pipeline and prediction provenance checks.
- `src/build_followup_report.py`: CPU-only follow-up report generation.
- `configs/`: exact settings for the completed comparison and follow-up experiments.
- `scripts/run_overnight_supervisor_with_fourclass.ps1`: sequential, hash-locked overnight launcher.

The legacy `src/train_videoswin_gt.py`, `src/train_videoswin_binary_clean.py`, and `src/analyze_binary_predictions.py` are retained for historical provenance. Current runners/evaluators hash these files. They are not recommended launch commands: they use fixed output paths and some execute work at import time.

Other historical tools may exist locally but are outside this minimal baseline. Supervisor annotation scripts, datasets, generated artifacts, checkpoints, and the manual audit are excluded from Git. No redistribution permission is established by this repository.

## Completed Vident-real experiments

The original training table contains 81 segments (63 Upper, 18 Lower). The cleaned table retains 53 segments (46 Upper, 7 Lower) after removal of 28 manually marked Ambiguous segments. All experiments use the same 13 validation segments (4 Upper, 9 Lower).

Best-epoch validation results recorded in the completed local runs:

| Model | Best epoch | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Original four-class checkpoint, quadrant-argmax mapped to binary | 9 | 61.54% | 72.22% | 0.6154 |
| Original 81, balanced sampling | 1 | 61.54% | 72.22% | 0.6154 |
| Cleaned 53, balanced sampling | 3 | 61.54% | 72.22% | 0.6154 |
| Cleaned 53, corrected weighted CE | 5 | 61.54% | 72.22% | 0.6154 |
| Exploratory partial last-block fine-tuning | 2 of 3 | 53.85% | 66.67% | 0.5357 |

The first four models have the same segment-level binary predictions and confusion matrix, with true rows and predicted columns ordered Upper, Lower: `[[4, 0], [5, 4]]`. Fresh checkpoint inference reproduced saved predictions; different checkpoint/head hashes and probabilities distinguish the models. Identical metrics alone do not imply cached prediction reuse.

The binary comparison uses seed 42, 15 fixed epochs, frozen KINETICS400_V1 backbone, AdamW head learning rate 0.001, weight decay 0.0001, unweighted CE and balanced segment sampling. Original and cleaned arms have 126 versus 92 updates per epoch, so equal epochs do not mean equal update budgets. The corrected weighted run uses 53 non-oversampled training segments per epoch and inverse-square-root class weights with singleton CE `reduction="sum"`. It addresses the legacy singleton mean-reduction cancellation bug but also changes sampling and update count. The three-epoch partial pilot warms up from the corrected weighted best checkpoint and unfreezes only the last block and head, using backbone/head learning rates 0.00001/0.0001.

Shared preprocessing is RGB GT, PIL bilinear square resize to 224x224, division by 255, and ImageNet normalization. Each segment supplies five clips at temporal positions 10/30/50/70/90%, with 16 consecutive frames per clip when available. Validation averages clip probabilities. Four-class evaluation averages quadrant probabilities, takes the quadrant argmax, then maps to binary; summing Q3+Q4 is a separate diagnostic. Frozen backbones retain training mode during training, including stochastic depth; evaluation uses eval mode.

Limitations: one seed, only 13 validation segments, repeated validation/best-epoch selection, unknown patient-level independence, and unequal optimization exposure. One validation segment changes accuracy by 7.69 percentage points. Partial fine-tuning and occlusion are exploratory. These results do not establish generalization or a causal benefit of cleaning/weighting.

## Required local inputs

Run from the repository root. Obtain data and annotations through authorized channels and preserve their original structure:

```text
data/raw/Vident-real/{train,val}/{uuid}/GT/{frame_number:05d}.png
data/labels/quadrant_metadata.csv
data/labels/binary_clean_metadata.csv
outputs/data_quality/manual_segment_audit_colored.xlsx
outputs/checkpoints/videoswin_gt_head_only_best.pt
```

The metadata supplies segment IDs, source UUIDs, frame boundaries, quadrant labels and split membership. The original Excel audit is an authored cleaning input, not a generated result; its blank CSV template is not a substitute. Both binary runners require it for provenance hashing even when reading an already-cleaned CSV. Preserve these inputs and their backups; do not regenerate them over existing files. The quadrant CSV also records the held-out test split; current experiment code selects train/validation only.

Training/preflight additionally require the cached pretrained file `swin3d_t-7615ae03.pth` in the current user's `.cache/torch/hub/checkpoints/` directory (or the configured Torch hub cache). The runners refuse an implicit download. Expected SHA-256:

```text
7615ae035996b65eb38dad437ae533d2dfcd36f9f89d28c0f0fa7bfb8e6b3130
```

Four-class evaluation requires the existing four-class checkpoint (SHA-256 `50c390800416c4e0b2d2e8bb37e8199db20b80f3f0f4eb4e999e6d33653106b7`). The historical pipeline audit also needs both old binary checkpoints under `outputs/checkpoints/` and `outputs/binary_val_predictions.csv`.

## Environment and reproduction entry points

Recorded environment: Windows, Python 3.11.9, Torch 2.14.0+cu126, Torchvision 0.29.0+cu126; NVIDIA Quadro P2000 5 GB. `requirements.txt` records observed direct dependencies, not a full lockfile or validated fresh-install recipe. Rebuilding requires a wheel source that supplies the exact CUDA-tagged Torch/Torchvision versions (the CUDA 12.6 PyTorch index, `https://download.pytorch.org/whl/cu126`); verify availability and driver compatibility before creating a separate environment. Do not replace an existing research environment.

Use the repository interpreter explicitly:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
```

Main interfaces (documentation only; do not launch without choosing fresh output paths and approving runtime):

| Entry point | Arguments / role |
|---|---|
| `src/run_binary_comparison.py` | `--config <original-or-cleaned.json> --run-dir <new-path-under-outputs/runs> --smoke`; replace `--smoke` with `--approve-full-training` for the configured full run |
| `src/run_followup_experiment.py` | `--config <followup.json> --run-dir <new-path> --start-before <ISO-time-with-offset> --finish-by <ISO-time-with-offset>` and `--smoke` or `--approve-full-training`; partial mode also requires `--warm-start <weighted-checkpoint>` |
| `src/evaluate_fourclass_as_binary.py` | `evaluate --output <new-path>`; `compare --evaluation <fourclass-run> --original <original-run> --cleaned <cleaned-run> --output <new-path>` |
| `src/overnight_binary_analysis.py` | Stages `preflight`, `analyze`, `occlude`, `report`; requires `--original`, `--cleaned`, `--output`, plus stage-specific analysis/occlusion inputs |
| `src/audit_binary_pipeline.py` | `--run-dir <new-audit-path>` |
| `src/audit_prediction_agreement.py` | `--output <new-audit-path>`; currently targets the exact historical run directories below |
| `src/build_followup_report.py` | `--current <overnight-run> --followup <followup-run>`; creates a new `report/` within the follow-up run |

The standalone full overnight launcher uses the explicit interpreter, unique run directories, sequential GPU stages, separate logs, and source/input hash checks:

```powershell
powershell.exe -NoProfile -File .\scripts\run_overnight_supervisor_with_fourclass.ps1
```

The launcher starts full training when manually launched; it is not a dry run. Preserve hash-locked source bytes, including line endings. Do not silently update expected hashes to bypass a mismatch. The older local follow-up launcher is not part of this baseline and contains historical deadlines; use the documented Python interface with explicitly chosen future deadlines for a separately approved rerun.

## Local evidence and reproducibility boundaries

Completed evidence is retained under these ignored locations:

```text
outputs/runs/overnight_20261014_201445_87e92c7a/
outputs/runs/followup_20261014_214855_def8c9c5/
outputs/runs/prediction_agreement_audit_2026-10-15_01/
```

Runs preserve configuration, input hashes, manifests, sampling orders, epoch metrics/predictions, checkpoints, logs and completion markers. The agreement audit depends on those exact historical run paths. Reports and clinical-image figures remain local.

The latest three-case signed/positive class-evidence analysis is in the audit's `explainability_original81_epoch01_norm_cam/` folder. Its verified script currently lives inside ignored outputs and is not included in this minimal baseline. It checks clip probabilities and reconstructs both class logits at `model.norm`; native maps are 8x7x7, not pixel-level localization. The saved per-clip CSV and all-frame HTML viewer also lack standalone source generators in this baseline. Thus a public checkout alone cannot recreate all explanation/audit artifacts; preserve the original local evidence until reusable generators are separately approved.

## Local-only material and future Mendeley work

The entire `data/` and `outputs/` trees, checkpoints, generated videos, environments and secrets are ignored. Ignoring files does not back them up or prevent script overwrites. Keep the verified research backup and separately preserve later results.

For the future Mendeley experiment, reuse the existing `src/` and `configs/` structure, keep downloaded inputs under local `data/`, generated clips/frames under `data/processed/`, and results in unique `outputs/runs/` directories. Do not duplicate the training project. License review, label design, source-image-level split rules and generation settings must be established before that experiment; no Mendeley training or generation is part of the completed results above.
