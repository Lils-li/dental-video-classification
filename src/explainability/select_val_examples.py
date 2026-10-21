# Source preservation: command-line execution resolves repository-relative inputs.
if __name__ == '__main__':
    import os as _preserve_os, sys as _preserve_sys
    from pathlib import Path as _PreservePath
    _preserve_root = _PreservePath(__file__).resolve().parents[1]
    _preserve_os.chdir(_preserve_root)
    _preserve_sys.path.insert(0, str(_preserve_root))
    _preserve_sys.path.insert(0, str(_preserve_root / 'src' / 'data'))

from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision.models.video import swin3d_t

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(PROJECT_ROOT / "src" / "data")
)

from dataset import DentalQuadrantDataset

METADATA = PROJECT_ROOT / "data" / "labels" / "quadrant_metadata.csv"

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "videoswin_gt_head_only_best.pt"
)

CLASS_NAMES = ["Q1", "Q2", "Q3", "Q4"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Validation dataset ONLY
# ============================================================

dataset = DentalQuadrantDataset(
    metadata_csv=METADATA,
    split="val",
    frame_source="GT",
)

print("Validation segments:", len(dataset))


# ============================================================
# Model
# ============================================================

model = swin3d_t(weights=None)

in_features = model.head.in_features
model.head = nn.Linear(in_features, 4)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("Loaded checkpoint epoch:", checkpoint["epoch"])
print("Checkpoint val Macro-F1:", checkpoint["val_macro_f1"])


# ============================================================
# Normalization
# ============================================================

mean = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
    device=DEVICE
).view(1, 3, 1, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
    device=DEVICE
).view(1, 3, 1, 1, 1)


def prepare_clip(clip):

    clip = clip.to(DEVICE)

    # [T,C,H,W] -> [1,C,T,H,W]
    clip = (
        clip
        .permute(1, 0, 2, 3)
        .unsqueeze(0)
    )

    clip = (clip - mean) / std

    return clip


# ============================================================
# Run all VAL segments
# ============================================================

results = []

with torch.no_grad():

    for i in range(len(dataset)):

        sample = dataset[i]

        clips = sample["clips"]

        true_id = int(
            sample["label"].item()
        )

        clip_probs = []
        clip_predictions = []

        for clip_idx in range(
            clips.shape[0]
        ):

            clip = prepare_clip(
                clips[clip_idx]
            )

            logits = model(clip)

            probs = torch.softmax(
                logits,
                dim=1
            )

            clip_probs.append(
                probs.cpu()
            )

            clip_predictions.append(
                CLASS_NAMES[
                    int(probs.argmax(dim=1).item())
                ]
            )

        # SAME aggregation as training/validation
        mean_probs = torch.cat(
            clip_probs,
            dim=0
        ).mean(dim=0)

        pred_id = int(
            mean_probs.argmax().item()
        )

        confidence = float(
            mean_probs[pred_id].item()
        )

        correct = (
            pred_id == true_id
        )

        result = {
            "dataset_index": i,
            "segment_id": sample["segment_id"],
            "video_id": sample["video_id"],
            "uuid": sample["uuid"],
            "true": CLASS_NAMES[true_id],
            "pred": CLASS_NAMES[pred_id],
            "confidence": confidence,
            "correct": correct,
            "clip_predictions": clip_predictions,
        }

        results.append(result)


# ============================================================
# Print
# ============================================================

print()
print("=" * 100)
print("VALIDATION PREDICTIONS — BEST CHECKPOINT")
print("=" * 100)

for r in results:

    status = "CORRECT" if r["correct"] else "WRONG"

    print()
    print(
        f"Index {r['dataset_index']:2d} | "
        f"Segment {r['segment_id']} | "
        f"True={r['true']} | "
        f"Pred={r['pred']} | "
        f"Conf={r['confidence']:.3f} | "
        f"{status}"
    )

    print(
        "  Clips:",
        r["clip_predictions"]
    )

    print(
        "  UUID:",
        r["uuid"]
    )


# ============================================================
# Useful candidates for explainability
# ============================================================

print()
print("=" * 100)
print("CANDIDATES FOR EXPLAINABILITY")
print("=" * 100)


for class_name in CLASS_NAMES:

    correct_examples = [
        r for r in results
        if r["true"] == class_name
        and r["correct"]
    ]

    wrong_examples = [
        r for r in results
        if r["true"] == class_name
        and not r["correct"]
    ]

    print()
    print(class_name)

    if correct_examples:

        r = correct_examples[0]

        print(
            "  Correct example:",
            f"index={r['dataset_index']},",
            f"segment={r['segment_id']},",
            f"{r['true']}->{r['pred']}"
        )

    else:

        print(
            "  Correct example: NONE"
        )

    if wrong_examples:

        r = wrong_examples[0]

        print(
            "  Wrong example:",
            f"index={r['dataset_index']},",
            f"segment={r['segment_id']},",
            f"{r['true']}->{r['pred']}"
        )

    else:

        print(
            "  Wrong example: NONE"
        )