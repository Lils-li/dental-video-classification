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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision.models.video import swin3d_t
from torchvision.models.video.swin_transformer import (
    _compute_pad_size_3d,
    _compute_attention_mask_3d,
    _get_window_and_shift_size,
)


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

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "explainability"
    / "window_attention"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


CLASS_NAMES = [
    "Q1",
    "Q2",
    "Q3",
    "Q4"
]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Chosen validation examples
# ============================================================

EXAMPLES = {
    "correct_Q3": 6,
    "wrong_Q4_to_Q1": 2,
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

model = swin3d_t(
    weights=None
)

model.head = nn.Linear(
    model.head.in_features,
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

    x = (
        clip
        .permute(1, 0, 2, 3)
        .unsqueeze(0)
    )

    x = (
        x - mean
    ) / std

    return x


# ============================================================
# Last attention block
# ============================================================

attn_module = (
    model
    .features[6][1]
    .attn
)


print("=" * 80)
print("ATTENTION MODULE")
print("=" * 80)

print(
    "window_size:",
    attn_module.window_size
)

print(
    "shift_size:",
    attn_module.shift_size
)

print(
    "num_heads:",
    attn_module.num_heads
)

print()


# ============================================================
# Capture INPUT to last attention block
#
# This hook works because the actual attn module itself
# IS called, even though qkv Linear is used through F.linear.
# ============================================================

captured = {}


def attn_input_hook(
    module,
    inputs
):
    captured["x"] = (
        inputs[0]
        .detach()
    )


hook = attn_module.register_forward_pre_hook(
    attn_input_hook
)


# ============================================================
# Recompute REAL attention weights
# following torchvision implementation
# ============================================================

def compute_attention_volume(
    x,
    module
):
    """
    x:
        [B, T, H, W, C]

    Returns:
        attention importance volume
        [B, T, H, W]

    Importance is mean attention RECEIVED by each token,
    averaged over heads and query positions.
    """

    b, t, h, w, c = x.shape

    size_dhw = [
        t,
        h,
        w
    ]

    window_size = (
        module.window_size.copy()
    )

    shift_size = (
        module.shift_size.copy()
    )

    window_size, shift_size = (
        _get_window_and_shift_size(
            shift_size,
            size_dhw,
            window_size
        )
    )

    relative_position_bias = (
        module.get_relative_position_bias(
            window_size
        )
    )


    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    pad_size = _compute_pad_size_3d(
        (t, h, w),
        (
            window_size[0],
            window_size[1],
            window_size[2]
        )
    )

    x_pad = F.pad(
        x,
        (
            0, 0,
            0, pad_size[2],
            0, pad_size[1],
            0, pad_size[0]
        )
    )

    _, tp, hp, wp, _ = (
        x_pad.shape
    )

    padded_size = (
        tp,
        hp,
        wp
    )


    # --------------------------------------------------------
    # Cyclic shift
    # --------------------------------------------------------

    if sum(shift_size) > 0:

        x_pad = torch.roll(
            x_pad,
            shifts=(
                -shift_size[0],
                -shift_size[1],
                -shift_size[2]
            ),
            dims=(1, 2, 3)
        )


    # --------------------------------------------------------
    # Window partition
    # --------------------------------------------------------

    nt = (
        tp // window_size[0]
    )

    nh = (
        hp // window_size[1]
    )

    nw = (
        wp // window_size[2]
    )

    num_windows = (
        nt * nh * nw
    )

    x_windows = (
        x_pad
        .view(
            b,
            nt,
            window_size[0],
            nh,
            window_size[1],
            nw,
            window_size[2],
            c,
        )
        .permute(
            0,
            1,
            3,
            5,
            2,
            4,
            6,
            7
        )
        .reshape(
            b * num_windows,
            (
                window_size[0]
                * window_size[1]
                * window_size[2]
            ),
            c
        )
    )


    # --------------------------------------------------------
    # Q K V
    # --------------------------------------------------------

    qkv = F.linear(
        x_windows,
        module.qkv.weight,
        module.qkv.bias
    )

    qkv = (
        qkv
        .reshape(
            x_windows.size(0),
            x_windows.size(1),
            3,
            module.num_heads,
            c // module.num_heads
        )
        .permute(
            2,
            0,
            3,
            1,
            4
        )
    )

    q = qkv[0]
    k = qkv[1]

    q = q * (
        c // module.num_heads
    ) ** -0.5


    # --------------------------------------------------------
    # Attention logits
    # --------------------------------------------------------

    attn = q.matmul(
        k.transpose(-2, -1)
    )

    attn = (
        attn
        + relative_position_bias
    )


    # --------------------------------------------------------
    # Shifted-window mask
    # --------------------------------------------------------

    if sum(shift_size) > 0:

        attn_mask = (
            _compute_attention_mask_3d(
                x_windows,
                padded_size,
                window_size,
                shift_size
            )
        )

        attn = attn.view(
            b,
            num_windows,
            module.num_heads,
            x_windows.size(1),
            x_windows.size(1)
        )

        attn = (
            attn
            + attn_mask
            .unsqueeze(1)
            .unsqueeze(0)
        )

        attn = attn.view(
            -1,
            module.num_heads,
            x_windows.size(1),
            x_windows.size(1)
        )


    # --------------------------------------------------------
    # REAL attention probabilities
    # --------------------------------------------------------

    attn = F.softmax(
        attn,
        dim=-1
    )


    # ========================================================
    # Convert attention matrix -> one importance per token
    #
    # attn shape:
    # [B*nWindows, heads, query, key]
    #
    # Average:
    # heads + queries
    #
    # Result:
    # [B*nWindows, key_token]
    # ========================================================

    importance = attn.mean(
        dim=(1, 2)
    )


    # --------------------------------------------------------
    # Restore 3D windows
    # --------------------------------------------------------

    importance = (
        importance
        .view(
            b,
            nt,
            nh,
            nw,
            window_size[0],
            window_size[1],
            window_size[2]
        )
        .permute(
            0,
            1,
            4,
            2,
            5,
            3,
            6
        )
        .reshape(
            b,
            tp,
            hp,
            wp
        )
    )


    # --------------------------------------------------------
    # Reverse cyclic shift
    # --------------------------------------------------------

    if sum(shift_size) > 0:

        importance = torch.roll(
            importance,
            shifts=(
                shift_size[0],
                shift_size[1],
                shift_size[2]
            ),
            dims=(1, 2, 3)
        )


    # --------------------------------------------------------
    # Remove padding
    # --------------------------------------------------------

    importance = importance[
        :,
        :t,
        :h,
        :w
    ]

    return importance


# ============================================================
# Upsample attention back to video dimensions
# ============================================================

def upsample_attention(
    volume,
    target_t=16,
    target_h=224,
    target_w=224
):

    # [B,T,H,W]
    # ->
    # [B,1,T,H,W]

    volume = volume.unsqueeze(1)

    volume = F.interpolate(
        volume,
        size=(
            target_t,
            target_h,
            target_w
        ),
        mode="trilinear",
        align_corners=False
    )

    volume = volume.squeeze(1)

    # Normalize separately per clip
    vmin = volume.amin(
        dim=(1, 2, 3),
        keepdim=True
    )

    vmax = volume.amax(
        dim=(1, 2, 3),
        keepdim=True
    )

    volume = (
        volume - vmin
    ) / (
        vmax - vmin + 1e-8
    )

    return volume


# ============================================================
# Save visualization
# ============================================================

def save_visualization(
    original_clip,
    attention_volume,
    output_path,
    title
):
    """
    Use middle frame of the 16-frame clip.

    original_clip:
        [T,C,H,W]

    attention_volume:
        [T,H,W]
    """

    frame_index = (
        original_clip.shape[0]
        // 2
    )

    frame = (
    original_clip[
        frame_index
    ]
    .detach()
    .permute(1, 2, 0)
    .cpu()
    .numpy()
)

    heatmap = (
    attention_volume[
        frame_index
    ]
    .detach()
    .cpu()
    .numpy()
)


    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )


    # Original
    axes[0].imshow(frame)
    axes[0].set_title(
        "GT frame"
    )
    axes[0].axis("off")


    # Heatmap
    axes[1].imshow(
        heatmap,
        cmap="jet"
    )
    axes[1].set_title(
        "Window attention"
    )
    axes[1].axis("off")


    # Overlay
    axes[2].imshow(frame)

    axes[2].imshow(
        heatmap,
        cmap="jet",
        alpha=0.45
    )

    axes[2].set_title(
        "Overlay"
    )

    axes[2].axis("off")


    fig.suptitle(
        title,
        fontsize=11
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# Run examples
# ============================================================

try:

    for (
        example_name,
        dataset_index
    ) in EXAMPLES.items():

        print()
        print("=" * 80)
        print(example_name)
        print("=" * 80)

        sample = dataset[
            dataset_index
        ]

        clips = sample["clips"]

        true_id = int(
            sample["label"].item()
        )

        example_dir = (
            OUTPUT_ROOT
            / example_name
        )

        example_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        segment_probabilities = []


        for clip_idx in range(
            clips.shape[0]
        ):

            captured.clear()

            original_clip = (
                clips[clip_idx]
            )

            model_input = (
                prepare_clip(
                    original_clip
                )
            )

            with torch.no_grad():

                logits = model(
                    model_input
                )

                probs = torch.softmax(
                    logits,
                    dim=1
                )


            if "x" not in captured:
                raise RuntimeError(
                    "Attention input hook "
                    "did not capture tensor."
                )


            # -----------------------------------------------
            # Last-block attention
            # -----------------------------------------------

            attention_volume = (
                compute_attention_volume(
                    captured["x"],
                    attn_module
                )
            )

            attention_volume = (
                upsample_attention(
                    attention_volume,
                    target_t=16,
                    target_h=224,
                    target_w=224
                )
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

            segment_probabilities.append(
                probs.cpu()
            )


            output_path = (
                example_dir
                / f"clip_{clip_idx + 1}.png"
            )

            title = (
                f"{example_name} | "
                f"clip {clip_idx + 1} | "
                f"true={CLASS_NAMES[true_id]} | "
                f"pred={CLASS_NAMES[pred_id]} | "
                f"conf={confidence:.3f}"
            )

            save_visualization(
                original_clip,
                attention_volume[0],
                output_path,
                title
            )


            print(
                f"Clip {clip_idx + 1}: "
                f"{CLASS_NAMES[pred_id]} "
                f"({confidence:.3f})"
            )

            print(
                "  Last block input:",
                tuple(
                    captured["x"].shape
                )
            )

            print(
                "  Attention volume:",
                tuple(
                    attention_volume.shape
                )
            )

            print(
                "  Saved:",
                output_path
            )


        # ====================================================
        # Segment prediction
        # ====================================================

        mean_probs = torch.cat(
            segment_probabilities,
            dim=0
        ).mean(dim=0)

        segment_pred = int(
            mean_probs.argmax().item()
        )

        print()
        print(
            "True segment label:",
            CLASS_NAMES[true_id]
        )

        print(
            "Final segment prediction:",
            CLASS_NAMES[segment_pred]
        )

        print(
            "Final confidence:",
            f"{float(mean_probs[segment_pred]):.3f}"
        )


finally:

    hook.remove()


print()
print("=" * 80)
print("FINISHED")
print("=" * 80)

print(
    "Saved to:",
    OUTPUT_ROOT
)

print()
print(
    "IMPORTANT:"
)

print(
    "These are last-block VideoSwin "
    "window-attention visualizations."
)

print(
    "They are NOT full Attention Rollout "
    "and NOT class-specific attribution maps."
)

print(
    "No test data was used."
)