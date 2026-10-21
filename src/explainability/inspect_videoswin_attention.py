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
    / "videoswin_gt_head_only_best.pt"
)

CLASS_NAMES = ["Q1", "Q2", "Q3", "Q4"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Validation examples ONLY
# ============================================================

EXAMPLES = {
    "correct_Q3": 6,
    "wrong_Q4": 2,
}


# ============================================================
# Dataset
# ============================================================

dataset = DentalQuadrantDataset(
    metadata_csv=METADATA,
    split="val",
    frame_source="GT",
)


# ============================================================
# Model
# ============================================================

model = swin3d_t(weights=None)

in_features = model.head.in_features

model.head = nn.Linear(
    in_features,
    4
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


print("=" * 80)
print("MODEL")
print("=" * 80)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)

print(
    "Checkpoint val Macro-F1:",
    checkpoint["val_macro_f1"]
)

print()


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

    # Dataset:
    # [T, C, H, W]

    clip = clip.to(DEVICE)

    # VideoSwin:
    # [B, C, T, H, W]

    clip = (
        clip
        .permute(1, 0, 2, 3)
        .unsqueeze(0)
    )

    clip = (
        clip - mean
    ) / std

    return clip


# ============================================================
# Select attention block
#
# Last VideoSwin attention block:
# features.6.1.attn
#
# qkv is the Linear layer generating:
# Query + Key + Value
# ============================================================

attention_block = (
    model
    .features[6][1]
    .attn
)

qkv_layer = attention_block.qkv


print("=" * 80)
print("ATTENTION MODULE")
print("=" * 80)

print(
    "Attention block:",
    type(attention_block).__name__
)

print(
    "QKV layer:",
    qkv_layer
)

print()


# ============================================================
# Hook storage
# ============================================================

captured = {}


def qkv_hook(
    module,
    inputs,
    output
):

    # input to qkv Linear
    # and output containing Q,K,V concatenated

    captured["qkv_input_shape"] = tuple(
        inputs[0].shape
    )

    captured["qkv_output_shape"] = tuple(
        output.shape
    )

    # Keep a CPU copy for later inspection.
    # We DO NOT call this an attention map yet.

    captured["qkv_output"] = (
        output
        .detach()
        .cpu()
    )


hook_handle = qkv_layer.register_forward_hook(
    qkv_hook
)


# ============================================================
# Run chosen validation examples
# ============================================================

try:

    for example_name, dataset_index in EXAMPLES.items():

        sample = dataset[
            dataset_index
        ]

        clips = sample["clips"]

        true_id = int(
            sample["label"].item()
        )

        print()
        print("=" * 80)
        print(example_name)
        print("=" * 80)

        print(
            "Dataset index:",
            dataset_index
        )

        print(
            "Segment ID:",
            sample["segment_id"]
        )

        print(
            "Video ID:",
            sample["video_id"]
        )

        print(
            "UUID:",
            sample["uuid"]
        )

        print(
            "True label:",
            CLASS_NAMES[true_id]
        )

        print(
            "All clips shape:",
            tuple(clips.shape)
        )

        print()


        clip_probabilities = []


        for clip_idx in range(
            clips.shape[0]
        ):

            captured.clear()

            clip = prepare_clip(
                clips[clip_idx]
            )

            with torch.no_grad():

                logits = model(
                    clip
                )

                probs = torch.softmax(
                    logits,
                    dim=1
                )

            pred_id = int(
                probs.argmax(
                    dim=1
                ).item()
            )

            confidence = float(
                probs[
                    0,
                    pred_id
                ].item()
            )

            clip_probabilities.append(
                probs.cpu()
            )


            print(
                f"Clip {clip_idx + 1}"
            )

            print(
                "  Input:",
                tuple(clip.shape)
            )

            print(
                "  Prediction:",
                CLASS_NAMES[pred_id]
            )

            print(
                "  Confidence:",
                f"{confidence:.4f}"
            )

            print(
                "  QKV input shape:",
                captured.get(
                    "qkv_input_shape"
                )
            )

            print(
                "  QKV output shape:",
                captured.get(
                    "qkv_output_shape"
                )
            )

            print(
                "  Frame indices:",
                sample[
                    "frame_indices"
                ][clip_idx].tolist()
            )

            print()


        # ====================================================
        # Segment-level prediction
        # SAME aggregation as our classifier
        # ====================================================

        mean_probs = torch.cat(
            clip_probabilities,
            dim=0
        ).mean(dim=0)

        segment_pred_id = int(
            mean_probs.argmax().item()
        )

        print(
            "SEGMENT PREDICTION:",
            CLASS_NAMES[
                segment_pred_id
            ]
        )

        print(
            "SEGMENT CONFIDENCE:",
            f"{float(mean_probs[segment_pred_id]):.4f}"
        )


finally:

    hook_handle.remove()


print()
print("=" * 80)
print("DONE")
print("=" * 80)

print(
    "No test data was used."
)

print(
    "The captured QKV tensors are NOT yet "
    "attention rollout maps."
)