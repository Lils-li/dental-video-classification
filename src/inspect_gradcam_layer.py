# Source preservation: command-line execution resolves repository-relative inputs.
if __name__ == '__main__':
    import os as _preserve_os, sys as _preserve_sys
    from pathlib import Path as _PreservePath
    _preserve_root = _PreservePath(__file__).resolve().parents[0]
    _preserve_os.chdir(_preserve_root)
    _preserve_sys.path.insert(0, str(_preserve_root))
    _preserve_sys.path.insert(0, str(_preserve_root / 'src' / 'data'))

from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn as nn
from torchvision.models.video import swin3d_t

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT / "src" / "data")
)

from dataset import DentalQuadrantDataset


# ============================================================
# Configuration
# ============================================================

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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TARGET_SEGMENT_ID = 4


# ============================================================
# Dataset
# ============================================================

dataset = DentalQuadrantDataset(
    metadata_csv=METADATA,
    split="val",
    frame_source="GT",
)

val_metadata = pd.read_csv(METADATA)

val_metadata = (
    val_metadata[
        val_metadata["split"] == "val"
    ]
    .reset_index(drop=True)
)

# Find segment 4 inside validation set
matches = val_metadata.index[
    val_metadata["segment_id"] == TARGET_SEGMENT_ID
].tolist()

if len(matches) != 1:
    raise ValueError(
        f"Expected exactly one segment {TARGET_SEGMENT_ID}, "
        f"found {len(matches)}"
    )

dataset_index = matches[0]

sample = dataset[dataset_index]
clips = sample["clips"]

print("Segment ID:", TARGET_SEGMENT_ID)
print("Dataset index:", dataset_index)
print("Original label:", val_metadata.iloc[dataset_index]["label"])
print("Number of clips:", clips.shape[0])
print("Raw clips shape:", tuple(clips.shape))


# ============================================================
# Model
# ============================================================

model = swin3d_t(weights=None)

model.head = nn.Linear(
    model.head.in_features,
    2,
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("Checkpoint epoch:", checkpoint["epoch"])


# ============================================================
# Hook
# ============================================================

activation = None


def forward_hook(module, input, output):
    global activation
    activation = output.detach()


target_layer = model.features[6][1]

hook_handle = target_layer.register_forward_hook(
    forward_hook
)


# ============================================================
# Normalization
# ============================================================

mean = torch.tensor(
    [0.485, 0.456, 0.406],
    device=DEVICE,
).view(1, 3, 1, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225],
    device=DEVICE,
).view(1, 3, 1, 1, 1)


# ============================================================
# Forward pass
# ============================================================

print("\nTarget layer:")
print(target_layer)

print("\nForward passes:")

with torch.no_grad():

    for clip_index in range(clips.shape[0]):

        clip = clips[clip_index].to(DEVICE)

        # [T,C,H,W] -> [1,C,T,H,W]
        clip = (
            clip
            .permute(1, 0, 2, 3)
            .unsqueeze(0)
        )

        clip = (clip - mean) / std

        logits = model(clip)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        print(
            f"\nClip {clip_index + 1}"
        )

        print(
            "Activation shape:",
            tuple(activation.shape),
        )

        print(
            f"P(Upper): {probabilities[0].item():.4f}"
        )

        print(
            f"P(Lower): {probabilities[1].item():.4f}"
        )


hook_handle.remove()

print("\nDone.")