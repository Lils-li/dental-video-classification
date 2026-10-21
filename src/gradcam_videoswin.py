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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.video import swin3d_t


# ============================================================
# Paths / configuration
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "gradcam"
    / "segment_4_clip_4"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TARGET_SEGMENT_ID = 4

# Human numbering:
# Clip 4 -> Python index 3
TARGET_CLIP_INDEX = 3

# We explain why the model predicted Upper
TARGET_CLASS = 0

CLASS_NAMES = [
    "Upper",
    "Lower",
]


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

matches = val_metadata.index[
    val_metadata["segment_id"]
    == TARGET_SEGMENT_ID
].tolist()

if len(matches) != 1:
    raise ValueError(
        f"Expected one validation segment with ID "
        f"{TARGET_SEGMENT_ID}, found {len(matches)}."
    )

dataset_index = matches[0]

sample = dataset[dataset_index]

clips = sample["clips"]

clip_original = clips[
    TARGET_CLIP_INDEX
].clone()

print("Segment:", TARGET_SEGMENT_ID)
print("Original label:",
      val_metadata.iloc[dataset_index]["label"])
print("Clip:", TARGET_CLIP_INDEX + 1)
print("Original clip shape:",
      tuple(clip_original.shape))


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

print("Checkpoint epoch:",
      checkpoint["epoch"])


# ============================================================
# Grad-CAM hooks
# ============================================================

activations = None
gradients = None


def forward_hook(module, inputs, output):
    global activations

    # Keep computational graph.
    # Do NOT detach here.
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients

    gradients = grad_output[0]


target_layer = model.features[6][1]

forward_handle = (
    target_layer.register_forward_hook(
        forward_hook
    )
)

backward_handle = (
    target_layer.register_full_backward_hook(
        backward_hook
    )
)


# ============================================================
# Prepare input
# ============================================================

clip = clip_original.to(DEVICE)

# [T,C,H,W] -> [1,C,T,H,W]
clip = (
    clip
    .permute(1, 0, 2, 3)
    .unsqueeze(0)
)

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

clip_normalized = (
    clip - mean
) / std


# ============================================================
# Forward pass
# ============================================================

model.zero_grad()

logits = model(
    clip_normalized
)

probabilities = torch.softmax(
    logits,
    dim=1,
)[0]

predicted_class = int(
    probabilities.argmax().item()
)

print()
print("Prediction:")
print(
    f"Upper: "
    f"{probabilities[0].item():.4f}"
)
print(
    f"Lower: "
    f"{probabilities[1].item():.4f}"
)
print(
    "Predicted:",
    CLASS_NAMES[predicted_class],
)


# ============================================================
# Backward pass for Upper
# ============================================================

target_score = logits[
    0,
    TARGET_CLASS
]

target_score.backward()


print()
print(
    "Activation shape:",
    tuple(activations.shape),
)

print(
    "Gradient shape:",
    tuple(gradients.shape),
)


# ============================================================
# 3D Grad-CAM
# ============================================================

# VideoSwin output:
#
# [B, T, H, W, C]
#
# [1, 8, 7, 7, 768]

# Average gradients across
# temporal + spatial dimensions.
#
# Result:
# [1, 1, 1, 1, 768]

weights = gradients.mean(
    dim=(1, 2, 3),
    keepdim=True,
)

# Weighted combination of channels.
#
# Result:
# [1, 8, 7, 7]

cam = (
    weights * activations
).sum(dim=-1)

# Standard Grad-CAM ReLU
cam = F.relu(cam)

# Add channel dimension:
#
# [1,8,7,7]
# ->
# [1,1,8,7,7]

cam = cam.unsqueeze(1)

# Upsample to original clip dimensions:
#
# 8x7x7
# ->
# 16x224x224

cam = F.interpolate(
    cam,
    size=(
        clip_original.shape[0],
        clip_original.shape[2],
        clip_original.shape[3],
    ),
    mode="trilinear",
    align_corners=False,
)

cam = cam[
    0,
    0,
]

# Normalize entire spatiotemporal CAM
# into [0,1].

cam_min = cam.min()
cam_max = cam.max()

cam = (
    cam - cam_min
) / (
    cam_max
    - cam_min
    + 1e-8
)

cam = (
    cam
    .detach()
    .cpu()
    .numpy()
)


# ============================================================
# Save overlays
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

frames = (
    clip_original
    .permute(0, 2, 3, 1)
    .cpu()
    .numpy()
)

# Safety clipping
frames = np.clip(
    frames,
    0.0,
    1.0,
)

print()
print("Saving Grad-CAM frames...")

for frame_index in range(
    frames.shape[0]
):

    frame = frames[
        frame_index
    ]

    heatmap = cam[
        frame_index
    ]

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ax.imshow(frame)

    ax.imshow(
        heatmap,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )

    ax.set_title(
        f"Segment {TARGET_SEGMENT_ID} | "
        f"Clip {TARGET_CLIP_INDEX + 1} | "
        f"Frame {frame_index + 1}\n"
        f"True: Lower | "
        f"Predicted: Upper "
        f"({probabilities[0].item():.3f})"
    )

    ax.axis("off")

    output_path = (
        OUTPUT_DIR
        / f"frame_{frame_index + 1:02d}.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Cleanup
# ============================================================

forward_handle.remove()
backward_handle.remove()

print()
print("Grad-CAM complete.")
print("Saved to:")
print(OUTPUT_DIR)