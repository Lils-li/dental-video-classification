from pathlib import Path
import sys
import time
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

from torchvision.models.video import (
    swin3d_t,
    Swin3D_T_Weights,
)


# ============================================================
# Import our dataset
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT / "src" / "data")
)

from dataset import DentalQuadrantDataset


# ============================================================
# Configuration
# ============================================================

METADATA = PROJECT_ROOT / "data" / "labels" / "quadrant_metadata.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR / "videoswin_gt_head_only_best.pt"
)

NUM_CLASSES = 4

MAX_EPOCHS = 15
PATIENCE = 3

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

CLASS_NAMES = ["Q1", "Q2", "Q3", "Q4"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("Max epochs:", MAX_EPOCHS)
print("Patience:", PATIENCE)
print("Learning rate:", LEARNING_RATE)
print()


# ============================================================
# Dataset
# ============================================================

train_dataset = DentalQuadrantDataset(
    metadata_csv=METADATA,
    split="train",
    frame_source="GT",
)

val_dataset = DentalQuadrantDataset(
    metadata_csv=METADATA,
    split="val",
    frame_source="GT",
)

print("Train segments:", len(train_dataset))
print("Val segments:", len(val_dataset))
print()


# ============================================================
# Model
# ============================================================

weights = Swin3D_T_Weights.DEFAULT

model = swin3d_t(
    weights=weights
)


# Freeze entire pretrained backbone
for param in model.parameters():
    param.requires_grad = False


# Replace Kinetics-400 head with Q1-Q4 head
in_features = model.head.in_features

model.head = nn.Linear(
    in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)


trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print("Total parameters:", f"{total_params:,}")
print("Trainable parameters:", f"{trainable_params:,}")

print(
    "Trainable percentage:",
    f"{100 * trainable_params / total_params:.4f}%"
)

print()


# ============================================================
# Class weights
#
# TRAIN:
# Q1 = 37
# Q2 = 26
# Q3 = 10
# Q4 = 8
#
# Same idea as previous baseline:
# weight = 1 / sqrt(class_count)
# then normalize so average weight = 1
# ============================================================

class_counts = torch.tensor(
    [37, 26, 10, 8],
    dtype=torch.float32
)

class_weights = 1.0 / torch.sqrt(class_counts)

class_weights = (
    class_weights
    / class_weights.mean()
)

class_weights = class_weights.to(DEVICE)

print(
    "Class weights:",
    {
        name: round(float(weight), 3)
        for name, weight
        in zip(CLASS_NAMES, class_weights.cpu())
    }
)

print()


criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


optimizer = torch.optim.AdamW(
    model.head.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


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
    """
    Input from dataset:
        [T, C, H, W]

    VideoSwin expects:
        [B, C, T, H, W]
    """

    clip = clip.to(
        DEVICE,
        non_blocking=True
    )

    clip = clip.permute(
        1, 0, 2, 3
    ).unsqueeze(0)

    clip = (
        clip - mean
    ) / std

    return clip


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    segment_predictions = []
    segment_targets = []

    start_time = time.time()

    for segment_index in range(
        len(train_dataset)
    ):

        sample = train_dataset[
            segment_index
        ]

        clips = sample["clips"]

        target_id = int(
            sample["label"].item()
        )

        target = torch.tensor(
            [target_id],
            dtype=torch.long,
            device=DEVICE
        )

        optimizer.zero_grad()

        clip_probabilities = []

        segment_loss = 0.0


        # ----------------------------------------------------
        # Five clips from this segment
        # ----------------------------------------------------

        for clip_index in range(
            clips.shape[0]
        ):

            clip = prepare_clip(
                clips[clip_index]
            )

            logits = model(clip)

            loss = criterion(
                logits,
                target
            )

            # Average gradient contribution
            # across five temporal clips.
            scaled_loss = (
                loss / clips.shape[0]
            )

            scaled_loss.backward()

            segment_loss += float(
                scaled_loss.item()
            )

            probabilities = torch.softmax(
                logits.detach(),
                dim=1
            )

            clip_probabilities.append(
                probabilities.cpu()
            )


        optimizer.step()


        # ----------------------------------------------------
        # Segment prediction =
        # mean probability over 5 clips
        # ----------------------------------------------------

        mean_probability = torch.cat(
            clip_probabilities,
            dim=0
        ).mean(dim=0)

        prediction = int(
            mean_probability.argmax().item()
        )

        segment_predictions.append(
            prediction
        )

        segment_targets.append(
            target_id
        )

        total_loss += segment_loss


    epoch_time = (
        time.time() - start_time
    )

    accuracy = accuracy_score(
        segment_targets,
        segment_predictions
    )

    macro_f1 = f1_score(
        segment_targets,
        segment_predictions,
        average="macro",
        zero_division=0
    )

    avg_loss = (
        total_loss
        / len(train_dataset)
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "time": epoch_time,
    }


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate_validation():

    model.eval()

    predictions = []
    targets = []

    total_loss = 0.0


    for segment_index in range(
        len(val_dataset)
    ):

        sample = val_dataset[
            segment_index
        ]

        clips = sample["clips"]

        target_id = int(
            sample["label"].item()
        )

        target = torch.tensor(
            [target_id],
            dtype=torch.long,
            device=DEVICE
        )

        clip_probabilities = []

        segment_loss = 0.0


        for clip_index in range(
            clips.shape[0]
        ):

            clip = prepare_clip(
                clips[clip_index]
            )

            logits = model(clip)

            loss = criterion(
                logits,
                target
            )

            segment_loss += (
                float(loss.item())
                / clips.shape[0]
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            clip_probabilities.append(
                probabilities.cpu()
            )


        mean_probability = torch.cat(
            clip_probabilities,
            dim=0
        ).mean(dim=0)

        prediction = int(
            mean_probability.argmax().item()
        )

        predictions.append(
            prediction
        )

        targets.append(
            target_id
        )

        total_loss += segment_loss


    accuracy = accuracy_score(
        targets,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            targets,
            predictions
        )
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1, 2, 3]
    )

    avg_loss = (
        total_loss
        / len(val_dataset)
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
    }


# ============================================================
# Training loop
# ============================================================

best_macro_f1 = -1.0
best_epoch = -1

epochs_without_improvement = 0


print("=" * 70)
print("START TRAINING")
print("=" * 70)
print()


for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    print(
        f"Epoch {epoch}/{MAX_EPOCHS}"
    )


    train_metrics = (
        train_one_epoch()
    )

    val_metrics = (
        evaluate_validation()
    )


    print(
        f"Train | "
        f"loss={train_metrics['loss']:.4f} | "
        f"acc={train_metrics['accuracy']:.4f} | "
        f"macroF1={train_metrics['macro_f1']:.4f}"
    )

    print(
        f"Val   | "
        f"loss={val_metrics['loss']:.4f} | "
        f"acc={val_metrics['accuracy']:.4f} | "
        f"bal_acc={val_metrics['balanced_accuracy']:.4f} | "
        f"macroF1={val_metrics['macro_f1']:.4f} | "
        f"weightedF1={val_metrics['weighted_f1']:.4f}"
    )

    print(
        f"Epoch time: "
        f"{train_metrics['time']:.1f} sec"
    )


    # --------------------------------------------------------
    # Best checkpoint based ONLY on validation Macro-F1
    # --------------------------------------------------------

    if (
        val_metrics["macro_f1"]
        > best_macro_f1
    ):

        best_macro_f1 = (
            val_metrics["macro_f1"]
        )

        best_epoch = epoch

        epochs_without_improvement = 0


        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": (
                    model.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "val_macro_f1": (
                    best_macro_f1
                ),
                "class_names": (
                    CLASS_NAMES
                ),
                "frame_source": "GT",
                "num_clips": 5,
                "frames_per_clip": 16,
            },
            BEST_MODEL_PATH
        )


        print(
            "✓ New best checkpoint saved."
        )


    else:

        epochs_without_improvement += 1

        print(
            "No Macro-F1 improvement.",
            f"Patience: "
            f"{epochs_without_improvement}/{PATIENCE}"
        )


    print()
    print(
        "Validation confusion matrix:"
    )

    print(
        val_metrics[
            "confusion_matrix"
        ]
    )

    print()
    print("-" * 70)
    print()


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "Early stopping."
        )

        break


# ============================================================
# Final training summary
# ============================================================

print()
print("=" * 70)
print("TRAINING FINISHED")
print("=" * 70)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation Macro-F1:",
    round(best_macro_f1, 4)
)

print(
    "Best checkpoint:",
    BEST_MODEL_PATH
)

print()
print(
    "TEST SPLIT WAS NOT EVALUATED."
)