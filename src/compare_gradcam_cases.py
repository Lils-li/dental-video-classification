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
# Configuration
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
    / "gradcam_comparison"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = ["Upper", "Lower"]

QUADRANT_TO_BINARY = {
    "Q1": 0,
    "Q2": 0,
    "Q3": 1,
    "Q4": 1,
}

# Same temporal sampling position for every segment
CLIP_INDEX = 3  # Clip 4



# ============================================================
# Dataset / metadata
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

assert len(val_metadata) == len(dataset)

SEGMENT_IDS = val_metadata["segment_id"].tolist()

print("Validation segments:", SEGMENT_IDS)
print("Number of validation segments:", len(SEGMENT_IDS))
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

print("Device:", DEVICE)
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


# ============================================================
# Hooks
# ============================================================

activations = None
gradients = None


def forward_hook(module, inputs, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


target_layer = model.features[6][1]

forward_handle = target_layer.register_forward_hook(
    forward_hook
)

backward_handle = target_layer.register_full_backward_hook(
    backward_hook
)


# ============================================================
# Grad-CAM function
# ============================================================

def generate_gradcam(clip_original, target_class):

    global activations, gradients

    activations = None
    gradients = None

    clip = clip_original.to(DEVICE)

    # [T,C,H,W] -> [1,C,T,H,W]
    clip_input = (
        clip
        .permute(1, 0, 2, 3)
        .unsqueeze(0)
    )

    clip_input = (
        clip_input - mean
    ) / std

    model.zero_grad()

    logits = model(clip_input)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    predicted_class = int(
        probabilities.argmax().item()
    )

    score = logits[
        0,
        target_class
    ]

    score.backward()

    if activations is None or gradients is None:
        raise RuntimeError(
            "Grad-CAM hooks did not capture tensors."
        )

    # activations / gradients:
    # [B,T,H,W,C]

    weights = gradients.mean(
        dim=(1, 2, 3),
        keepdim=True,
    )

    cam = (
        weights * activations
    ).sum(dim=-1)

    cam = F.relu(cam)

    # [B,T,H,W] -> [B,1,T,H,W]
    cam = cam.unsqueeze(1)

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

    cam = cam[0, 0]

    cam_min = cam.min()
    cam_max = cam.max()

    cam = (
        (cam - cam_min)
        / (cam_max - cam_min + 1e-8)
    )

    return (
        cam.detach().cpu().numpy(),
        probabilities.detach().cpu().numpy(),
        predicted_class,
    )


# ============================================================
# Run comparison
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

summary_rows = []


for segment_id in SEGMENT_IDS:

    matches = val_metadata.index[
        val_metadata["segment_id"] == segment_id
    ].tolist()

    if len(matches) != 1:
        raise ValueError(
            f"Expected one segment {segment_id}, "
            f"found {len(matches)}."
        )

    dataset_index = matches[0]

    row = val_metadata.iloc[
        dataset_index
    ]

    original_label = str(
        row["label"]
    )

    true_class = QUADRANT_TO_BINARY[
        original_label
    ]

    sample = dataset[
        dataset_index
    ]

    clip_original = sample[
        "clips"
    ][CLIP_INDEX].clone()

    # --------------------------------------------------------
    # First forward pass:
    # determine what the model predicts.
    #
    # We then explain THE PREDICTED CLASS.
    # --------------------------------------------------------

    clip_temp = (
        clip_original
        .to(DEVICE)
        .permute(1, 0, 2, 3)
        .unsqueeze(0)
    )

    clip_temp = (
        clip_temp - mean
    ) / std

    with torch.no_grad():

        temp_logits = model(
            clip_temp
        )

        temp_probabilities = torch.softmax(
            temp_logits,
            dim=1,
        )[0]

        predicted_class = int(
            temp_probabilities.argmax().item()
        )

    # Explain the class the model actually chose.
    cam, probabilities, predicted_class = (
        generate_gradcam(
            clip_original,
            predicted_class,
        )
    )

    true_name = CLASS_NAMES[
        true_class
    ]

    predicted_name = CLASS_NAMES[
        predicted_class
    ]

    correct = (
        true_class
        == predicted_class
    )

    print()
    print("=" * 60)
    print("Segment:", segment_id)
    print("Original:", original_label)
    print("True:", true_name)
    print("Predicted:", predicted_name)
    print(
        f"P(Upper): {probabilities[0]:.4f}"
    )
    print(
        f"P(Lower): {probabilities[1]:.4f}"
    )
    print("Correct:", correct)

    # --------------------------------------------------------
    # Frames
    # --------------------------------------------------------

    frames = (
        clip_original
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )

    frames = np.clip(
        frames,
        0.0,
        1.0,
    )

    case_dir = (
        OUTPUT_DIR
        / f"segment_{segment_id}"
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    np.save(
    case_dir / "gradcam_raw.npy",
    cam,
    )

    print(
        f"Saved raw CAM for segment {segment_id}: "
        f"shape={cam.shape}"
    )

    for frame_index in range(
        frames.shape[0]
    ):

        fig, ax = plt.subplots(
            figsize=(6, 6)
        )

        ax.imshow(
            frames[frame_index]
        )

        ax.imshow(
            cam[frame_index],
            cmap="jet",
            alpha=0.45,
            vmin=0,
            vmax=1,
        )

        status = (
            "CORRECT"
            if correct
            else "MISCLASSIFIED"
        )

        ax.set_title(
            f"Segment {segment_id} | "
            f"{original_label} | "
            f"Clip {CLIP_INDEX + 1} | "
            f"Frame {frame_index + 1}\n"
            f"True: {true_name} | "
            f"Pred: {predicted_name} "
            f"({probabilities[predicted_class]:.3f}) "
            f"| {status}"
        )

        ax.axis("off")

        fig.savefig(
            case_dir
            / f"frame_{frame_index + 1:02d}.png",
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

    summary_rows.append(
        {
            "segment_id": segment_id,
            "original_label": original_label,
            "clip": CLIP_INDEX + 1,
            "true": true_name,
            "predicted": predicted_name,
            "p_upper": probabilities[0],
            "p_lower": probabilities[1],
            "correct": correct,
            "explained_class": predicted_name,
        }
    )


# ============================================================
# Save summary
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUTPUT_DIR
    / "comparison_summary.csv",
    index=False,
)

print()
print("=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)

print(
    summary.to_string(
        index=False,
        formatters={
            "p_upper": "{:.4f}".format,
            "p_lower": "{:.4f}".format,
        },
    )
)

print()
print("Saved to:")
print(OUTPUT_DIR)


forward_handle.remove()
backward_handle.remove()