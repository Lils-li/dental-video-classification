from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn as nn
from torchvision.models.video import swin3d_t

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT / "src" / "data")
)

from dataset import DentalQuadrantDataset

METADATA = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "quadrant_metadata.csv"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "videoswin_binary_clean_balanced_best.pt"
)

OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "binary_val_predictions.csv"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = ["Upper", "Lower"]

# Original dataset still returns Q1-Q4 as 0,1,2,3
QUADRANT_TO_BINARY = {
    0: 0,  # Q1 -> Upper
    1: 0,  # Q2 -> Upper
    2: 1,  # Q3 -> Lower
    3: 1,  # Q4 -> Lower
}


# ============================================================
# Dataset
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
model.head = nn.Linear(in_features, 2)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("Loaded checkpoint:", CHECKPOINT)
print("Checkpoint epoch:", checkpoint["epoch"])


# ============================================================
# Normalization
# ============================================================

mean = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
    device=DEVICE,
).view(1, 3, 1, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
    device=DEVICE,
).view(1, 3, 1, 1, 1)


def prepare_clip(clip):

    clip = clip.to(
        DEVICE,
        non_blocking=True,
    )

    # [T,C,H,W] -> [1,C,T,H,W]
    clip = (
        clip
        .permute(1, 0, 2, 3)
        .unsqueeze(0)
    )

    clip = (clip - mean) / std

    return clip


# ============================================================
# Prediction
# ============================================================

# ============================================================
# Prediction
# ============================================================

# Read validation metadata once
val_metadata = pd.read_csv(METADATA)

val_metadata = (
    val_metadata[
        val_metadata["split"] == "val"
    ]
    .reset_index(drop=True)
)

assert len(val_metadata) == len(dataset)

rows = []

with torch.no_grad():

    for index in range(len(dataset)):

        sample = dataset[index]

        clips = sample["clips"]

        quadrant_id = int(
            sample["label"].item()
        )

        true_binary = QUADRANT_TO_BINARY[
            quadrant_id
        ]

        clip_probabilities = []

        for clip_index in range(clips.shape[0]):

            clip = prepare_clip(
                clips[clip_index]
            )

            logits = model(clip)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            clip_probabilities.append(
                probabilities.cpu()
            )

        # Average probabilities over 5 clips
        mean_probability = torch.cat(
            clip_probabilities,
            dim=0,
        ).mean(dim=0)

        predicted_binary = int(
            mean_probability.argmax().item()
        )

        # Metadata for this validation segment
        row_metadata = val_metadata.iloc[index]

        segment_id = int(
            row_metadata["segment_id"]
        )

        original_label = str(
            row_metadata["label"]
        )

        p_upper = float(
            mean_probability[0].item()
        )

        p_lower = float(
            mean_probability[1].item()
        )

        rows.append(
            {
                "segment_id": segment_id,
                "original_label": original_label,
                "true_binary": CLASS_NAMES[
                    true_binary
                ],
                "predicted_binary": CLASS_NAMES[
                    predicted_binary
                ],
                "p_upper": p_upper,
                "p_lower": p_lower,
                "correct": (
                    true_binary
                    == predicted_binary
                ),
            }
        )

# ============================================================
# Results
# ============================================================

results = pd.DataFrame(rows)

results = results.sort_values(
    ["correct", "segment_id"]
)

print()
print("=" * 90)
print("VALIDATION PREDICTIONS")
print("=" * 90)

print(
    results.to_string(
        index=False,
        formatters={
            "p_upper": "{:.4f}".format,
            "p_lower": "{:.4f}".format,
        },
    )
)

print()
print("=" * 90)
print("MISCLASSIFIED SEGMENTS")
print("=" * 90)

misclassified = results[
    results["correct"] == False
]

print(
    misclassified.to_string(
        index=False,
        formatters={
            "p_upper": "{:.4f}".format,
            "p_lower": "{:.4f}".format,
        },
    )
)

print(
    "\nTotal misclassified:",
    len(misclassified),
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_csv(
    OUTPUT,
    index=False,
)

print("\nSaved:")
print(OUTPUT)