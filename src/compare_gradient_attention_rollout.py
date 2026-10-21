# LEGACY / UNVALIDATED PROTOTYPE: attribution correctness is not established.
# Source preservation: command-line execution resolves repository-relative inputs.
if __name__ == '__main__':
    import os as _preserve_os, sys as _preserve_sys
    from pathlib import Path as _PreservePath
    _preserve_root = _PreservePath(__file__).resolve().parents[0]
    _preserve_os.chdir(_preserve_root)
    _preserve_sys.path.insert(0, str(_preserve_root))
    _preserve_sys.path.insert(0, str(_preserve_root / 'src' / 'data'))

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision.models.video import swin3d_t
from torchvision.models.video.swin_transformer import (
    _compute_pad_size_3d,
    _compute_attention_mask_3d,
    _get_window_and_shift_size,
)


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DentalQuadrantDataset


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

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "gradient_attention_rollout_all"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

# Clip 4
CLIP_INDEX = 3

CLASS_NAMES = [
    "Upper",
    "Lower",
]


# ============================================================
# MODIFIED ATTENTION FORWARD
# ============================================================

def attention_forward_with_grad(self, x):
    """
    Torchvision ShiftedWindowAttention3d computation with
    access to the real attention tensor used in the forward
    graph.

    This allows gradients of the predicted class score with
    respect to attention to be obtained.
    """

    b, t, h, w, c = x.shape

    size_dhw = [
        t,
        h,
        w,
    ]

    window_size = self.window_size.copy()
    shift_size = self.shift_size.copy()

    window_size, shift_size = (
        _get_window_and_shift_size(
            shift_size,
            size_dhw,
            window_size,
        )
    )

    self.effective_window_size = (
        window_size
    )

    self.effective_shift_size = (
        shift_size
    )

    self.input_size_dhw = (
        size_dhw
    )

    relative_position_bias = (
        self.get_relative_position_bias(
            window_size
        )
    )

    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    pad_size = _compute_pad_size_3d(
        (t, h, w),
        tuple(window_size),
    )

    x = F.pad(
        x,
        (
            0, 0,
            0, pad_size[2],
            0, pad_size[1],
            0, pad_size[0],
        ),
    )

    _, tp, hp, wp, _ = x.shape

    padded_size = (
        tp,
        hp,
        wp,
    )

    # --------------------------------------------------------
    # Cyclic shift
    # --------------------------------------------------------

    if sum(shift_size) > 0:

        x = torch.roll(
            x,
            shifts=(
                -shift_size[0],
                -shift_size[1],
                -shift_size[2],
            ),
            dims=(
                1,
                2,
                3,
            ),
        )

    # --------------------------------------------------------
    # Window partition
    # --------------------------------------------------------

    num_windows = (
        (tp // window_size[0])
        * (hp // window_size[1])
        * (wp // window_size[2])
    )

    self.saved_num_windows = (
        num_windows
    )

    x = x.view(
        b,
        tp // window_size[0],
        window_size[0],
        hp // window_size[1],
        window_size[1],
        wp // window_size[2],
        window_size[2],
        c,
    )

    x = (
        x.permute(
            0,
            1,
            3,
            5,
            2,
            4,
            6,
            7,
        )
        .reshape(
            b * num_windows,
            window_size[0]
            * window_size[1]
            * window_size[2],
            c,
        )
    )

    # --------------------------------------------------------
    # QKV
    # --------------------------------------------------------

    qkv = F.linear(
        x,
        self.qkv.weight,
        self.qkv.bias,
    )

    qkv = qkv.reshape(
        x.size(0),
        x.size(1),
        3,
        self.num_heads,
        c // self.num_heads,
    )

    qkv = qkv.permute(
        2,
        0,
        3,
        1,
        4,
    )

    q = qkv[0]
    k = qkv[1]
    v = qkv[2]

    q = q * (
        c // self.num_heads
    ) ** -0.5

    # --------------------------------------------------------
    # Attention
    # --------------------------------------------------------

    attn = q.matmul(
        k.transpose(
            -2,
            -1,
        )
    )

    attn = (
        attn
        + relative_position_bias
    )

    # --------------------------------------------------------
    # Shift mask
    # --------------------------------------------------------

    if sum(shift_size) > 0:

        attn_mask = (
            _compute_attention_mask_3d(
                x,
                padded_size,
                tuple(window_size),
                tuple(shift_size),
            )
        )

        attn = attn.view(
            x.size(0) // num_windows,
            num_windows,
            self.num_heads,
            x.size(1),
            x.size(1),
        )

        attn = (
            attn
            + attn_mask
            .unsqueeze(1)
            .unsqueeze(0)
        )

        attn = attn.view(
            -1,
            self.num_heads,
            x.size(1),
            x.size(1),
        )

    attn = F.softmax(
        attn,
        dim=-1,
    )

    # Keep the real attention tensor.
    attn.retain_grad()

    self.saved_attention = (
        attn
    )

    # --------------------------------------------------------
    # Attention x V
    # --------------------------------------------------------

    x = (
        attn.matmul(v)
        .transpose(
            1,
            2,
        )
        .reshape(
            x.size(0),
            x.size(1),
            c,
        )
    )

    # --------------------------------------------------------
    # Projection
    # --------------------------------------------------------

    x = F.linear(
        x,
        self.proj.weight,
        self.proj.bias,
    )

    x = F.dropout(
        x,
        p=self.dropout,
        training=self.training,
    )

    # --------------------------------------------------------
    # Reverse windows
    # --------------------------------------------------------

    x = x.view(
        b,
        tp // window_size[0],
        hp // window_size[1],
        wp // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        c,
    )

    x = (
        x.permute(
            0,
            1,
            4,
            2,
            5,
            3,
            6,
            7,
        )
        .reshape(
            b,
            tp,
            hp,
            wp,
            c,
        )
    )

    # --------------------------------------------------------
    # Reverse shift
    # --------------------------------------------------------

    if sum(shift_size) > 0:

        x = torch.roll(
            x,
            shifts=(
                shift_size[0],
                shift_size[1],
                shift_size[2],
            ),
            dims=(
                1,
                2,
                3,
            ),
        )

    # Remove padding
    x = x[
        :,
        :t,
        :h,
        :w,
        :
    ].contiguous()

    return x


# ============================================================
# BINARY LABEL
# ============================================================

def quadrant_to_binary(
    quadrant,
):

    if quadrant in [
        "Q1",
        "Q2",
    ]:

        return 0

    return 1


# ============================================================
# CREATE MODEL
# ============================================================

def build_model(
    device,
):

    model = swin3d_t(
        weights=None
    )

    model.head = (
        torch.nn.Linear(
            model.head.in_features,
            2,
        )
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "model_state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        device
    )

    model.eval()

    # Final stage contains two blocks.
    block_1 = (
        model
        .features[6][0]
        .attn
    )

    block_2 = (
        model
        .features[6][1]
        .attn
    )

    block_1.forward = (
        types.MethodType(
            attention_forward_with_grad,
            block_1,
        )
    )

    block_2.forward = (
        types.MethodType(
            attention_forward_with_grad,
            block_2,
        )
    )

    return (
        model,
        block_1,
        block_2,
    )


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_input(
    clip,
    device,
):

    # [T,C,H,W]
    # ->
    # [1,C,T,H,W]

    x = (
        clip
        .permute(
            1,
            0,
            2,
            3,
        )
        .unsqueeze(0)
        .float()
    )

    mean = torch.tensor(
        [
            0.485,
            0.456,
            0.406,
        ],
        dtype=x.dtype,
    ).view(
        1,
        3,
        1,
        1,
        1,
    )

    std = torch.tensor(
        [
            0.229,
            0.224,
            0.225,
        ],
        dtype=x.dtype,
    ).view(
        1,
        3,
        1,
        1,
        1,
    )

    x = (
        x - mean
    ) / std

    return x.to(
        device
    )


# ============================================================
# COMPUTE FINAL-STAGE GRADIENT ATTENTION ROLLOUT
# ============================================================

def compute_rollout(
    model,
    block_1,
    block_2,
    x,
):

    model.zero_grad(
        set_to_none=True
    )

    logits = model(
        x
    )

    probabilities = (
        torch.softmax(
            logits,
            dim=1,
        )
    )

    predicted_class = (
        logits
        .argmax(
            dim=1
        )
        .item()
    )

    score = logits[
        0,
        predicted_class,
    ]

    score.backward()

    attention_1 = (
        block_1.saved_attention
    )

    attention_2 = (
        block_2.saved_attention
    )

    gradient_1 = (
        attention_1.grad
    )

    gradient_2 = (
        attention_2.grad
    )

    if gradient_1 is None:

        raise RuntimeError(
            "Block 1 attention gradient "
            "is missing."
        )

    if gradient_2 is None:

        raise RuntimeError(
            "Block 2 attention gradient "
            "is missing."
        )

    # --------------------------------------------------------
    # Gradient-weighted attention
    # --------------------------------------------------------

    G1 = torch.relu(
        attention_1
        * gradient_1
    ).mean(
        dim=1
    )

    G2 = torch.relu(
        attention_2
        * gradient_2
    ).mean(
        dim=1
    )

    # Shapes:
    #
    # [1,392,392]

    if (
        G1.shape[-1] != 392
        or
        G2.shape[-1] != 392
    ):

        raise RuntimeError(
            "Unexpected final-stage token count. "
            f"G1={G1.shape}, G2={G2.shape}"
        )

    # --------------------------------------------------------
    # Residual connections
    # --------------------------------------------------------

    num_tokens = (
        G1.shape[-1]
    )

    identity = torch.eye(
        num_tokens,
        device=G1.device,
        dtype=G1.dtype,
    ).unsqueeze(0)

    R1 = (
        G1
        + identity
    )

    R2 = (
        G2
        + identity
    )

    # --------------------------------------------------------
    # Row normalization
    # --------------------------------------------------------

    eps = 1e-8

    R1 = R1 / (
        R1.sum(
            dim=-1,
            keepdim=True,
        )
        + eps
    )

    R2 = R2 / (
        R2.sum(
            dim=-1,
            keepdim=True,
        )
        + eps
    )

    # --------------------------------------------------------
    # Rollout
    # --------------------------------------------------------

    rollout = torch.bmm(
        R2,
        R1,
    )

    # --------------------------------------------------------
    # VideoSwin uses global average pooling.
    #
    # No CLS token exists, so start from uniform relevance
    # over the final output tokens.
    # --------------------------------------------------------

    output_relevance = torch.ones(
        1,
        num_tokens,
        device=rollout.device,
        dtype=rollout.dtype,
    )

    output_relevance = (
        output_relevance
        / num_tokens
    )

    token_relevance = torch.bmm(
        output_relevance.unsqueeze(1),
        rollout,
    ).squeeze(1)

    # --------------------------------------------------------
    # 392 -> 8 x 7 x 7
    # --------------------------------------------------------

    token_relevance = (
        token_relevance
        .reshape(
            1,
            8,
            7,
            7,
        )
    )

    # Normalize each segment map independently.
    token_relevance = (
        token_relevance
        - token_relevance.min()
    )

    max_value = (
        token_relevance.max()
    )

    if max_value.item() > 0:

        token_relevance = (
            token_relevance
            / max_value
        )

    # --------------------------------------------------------
    # 8x7x7 -> 16x224x224
    # --------------------------------------------------------

    volume = F.interpolate(
        token_relevance.unsqueeze(1),
        size=(
            16,
            224,
            224,
        ),
        mode="trilinear",
        align_corners=False,
    )

    volume = (
        volume[
            0,
            0,
        ]
        .detach()
        .cpu()
        .numpy()
    )

    return (
        volume,
        probabilities.detach(),
        predicted_class,
        rollout.detach(),
    )


# ============================================================
# SAVE VISUALIZATIONS
# ============================================================

def save_segment_outputs(
    segment_id,
    clip,
    volume,
    true_name,
    predicted_name,
    probability_upper,
    probability_lower,
):

    segment_dir = (
        OUTPUT_ROOT
        / f"segment_{segment_id}"
    )

    segment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Raw 3D rollout
    np.save(
        segment_dir
        / "rollout_raw.npy",
        volume,
    )

    # --------------------------------------------------------
    # Original frames
    # --------------------------------------------------------

    frames = (
        clip
        .detach()
        .cpu()
        .numpy()
    )

    frames = np.transpose(
        frames,
        (
            0,
            2,
            3,
            1,
        ),
    )

    frames = np.clip(
        frames,
        0,
        1,
    )

    # --------------------------------------------------------
    # Save 16 overlays
    # --------------------------------------------------------

    for frame_idx in range(
        volume.shape[0]
    ):

        fig, ax = plt.subplots(
            figsize=(
                6,
                6,
            )
        )

        ax.imshow(
            frames[
                frame_idx
            ]
        )

        ax.imshow(
            volume[
                frame_idx
            ],
            cmap="jet",
            alpha=0.45,
            vmin=0,
            vmax=1,
        )

        ax.set_title(
            "Final-stage Gradient Attention Rollout\n"
            f"Segment {segment_id} | "
            f"Clip {CLIP_INDEX + 1} | "
            f"Frame {frame_idx + 1}\n"
            f"True: {true_name} | "
            f"Pred: {predicted_name}"
        )

        ax.axis(
            "off"
        )

        plt.tight_layout()

        plt.savefig(
            segment_dir
            / f"frame_{frame_idx + 1:02d}.png",
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    # --------------------------------------------------------
    # Temporal relevance
    # --------------------------------------------------------

    temporal = volume.mean(
        axis=(
            1,
            2,
        )
    )

    np.save(
        segment_dir
        / "temporal_relevance.npy",
        temporal,
    )

    fig, ax = plt.subplots(
        figsize=(
            8,
            4,
        )
    )

    ax.plot(
        np.arange(
            1,
            len(temporal) + 1,
        ),
        temporal,
        marker="o",
    )

    ax.set_xlabel(
        "Frame"
    )

    ax.set_ylabel(
        "Mean rollout relevance"
    )

    ax.set_title(
        "Temporal Final-stage Gradient Attention Rollout\n"
        f"Segment {segment_id} | "
        f"True: {true_name} | "
        f"Pred: {predicted_name}"
    )

    ax.set_xticks(
        np.arange(
            1,
            17,
        )
    )

    plt.tight_layout()

    plt.savefig(
        segment_dir
        / "temporal_relevance.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = pd.read_csv(
        METADATA
    )

    val_metadata = (
        metadata[
            metadata["split"]
            == "val"
        ]
        .reset_index(
            drop=True
        )
    )

    print(
        "Validation segments:",
        val_metadata[
            "segment_id"
        ].tolist(),
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = DentalQuadrantDataset(
        metadata_csv=str(
            METADATA
        ),
        split="val",
        frame_source="GT",
    )

    print(
        "Validation dataset size:",
        len(dataset),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, block_1, block_2 = (
        build_model(
            device
        )
    )

    print(
        "Checkpoint loaded."
    )

    print(
        "Final-stage attention blocks patched."
    )

    results = []

    # ========================================================
    # ALL VALIDATION SEGMENTS
    # ========================================================

    for dataset_index in range(
        len(dataset)
    ):

        row = val_metadata.iloc[
            dataset_index
        ]

        segment_id = int(
            row["segment_id"]
        )

        quadrant = row[
            "label"
        ]

        true_class = (
            quadrant_to_binary(
                quadrant
            )
        )

        true_name = (
            CLASS_NAMES[
                true_class
            ]
        )

        sample = dataset[
            dataset_index
        ]

        if not isinstance(
            sample,
            dict,
        ):

            raise TypeError(
                "Expected dataset sample "
                "to be a dictionary."
            )

        clips = sample[
            "clips"
        ]

        clip = clips[
            CLIP_INDEX
        ]

        x = prepare_input(
            clip,
            device,
        )

        print(
            "\n========================================"
        )

        print(
            f"Segment {segment_id}/13"
        )

        print(
            f"Original: {quadrant}"
        )

        print(
            f"True: {true_name}"
        )

        # ----------------------------------------------------
        # Rollout
        # ----------------------------------------------------

        (
            volume,
            probabilities,
            predicted_class,
            rollout,
        ) = compute_rollout(
            model,
            block_1,
            block_2,
            x,
        )

        predicted_name = (
            CLASS_NAMES[
                predicted_class
            ]
        )

        p_upper = (
            probabilities[
                0,
                0,
            ].item()
        )

        p_lower = (
            probabilities[
                0,
                1,
            ].item()
        )

        correct = (
            predicted_class
            == true_class
        )

        print(
            f"Predicted: "
            f"{predicted_name}"
        )

        print(
            f"P(Upper): "
            f"{p_upper:.4f}"
        )

        print(
            f"P(Lower): "
            f"{p_lower:.4f}"
        )

        print(
            f"Correct: "
            f"{correct}"
        )

        print(
            "Rollout volume:",
            volume.shape,
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_segment_outputs(
            segment_id=segment_id,
            clip=clip,
            volume=volume,
            true_name=true_name,
            predicted_name=predicted_name,
            probability_upper=p_upper,
            probability_lower=p_lower,
        )

        # ----------------------------------------------------
        # Temporal summary
        # ----------------------------------------------------

        temporal = volume.mean(
            axis=(
                1,
                2,
            )
        )

        peak_frame = int(
            np.argmax(
                temporal
            )
            + 1
        )

        peak_temporal_relevance = float(
            temporal.max()
        )

        mean_temporal_relevance = float(
            temporal.mean()
        )

        # ----------------------------------------------------
        # Summary row
        # ----------------------------------------------------

        results.append(
            {
                "segment_id":
                    segment_id,

                "original_label":
                    quadrant,

                "true_binary":
                    true_name,

                "predicted_binary":
                    predicted_name,

                "p_upper":
                    p_upper,

                "p_lower":
                    p_lower,

                "correct":
                    correct,

                "clip_index":
                    CLIP_INDEX + 1,

                "peak_frame":
                    peak_frame,

                "peak_temporal_relevance":
                    peak_temporal_relevance,

                "mean_temporal_relevance":
                    mean_temporal_relevance,
            }
        )

        # Release graph references before next segment.
        model.zero_grad(
            set_to_none=True
        )

        block_1.saved_attention = None
        block_2.saved_attention = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    summary_path = (
        OUTPUT_ROOT
        / "rollout_summary.csv"
    )

    results_df.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print(
        "\n\n========================================"
    )

    print(
        "ALL VALIDATION SEGMENTS COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        results_df[
            [
                "segment_id",
                "original_label",
                "true_binary",
                "predicted_binary",
                "p_upper",
                "p_lower",
                "correct",
                "peak_frame",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nCorrect:"
    )

    print(
        results_df[
            "correct"
        ].value_counts()
    )

    print(
        "\nSummary saved to:"
    )

    print(
        summary_path
    )

    print(
        "\nRollout images saved to:"
    )

    print(
        OUTPUT_ROOT
    )


if __name__ == "__main__":
    main()