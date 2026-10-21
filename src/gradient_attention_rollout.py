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

import pandas as pd
import torch
import torch.nn.functional as F

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


CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "videoswin_binary_clean_balanced_best.pt"
)

METADATA = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "quadrant_metadata.csv"
)

TARGET_SEGMENT = 4

# Python index 3 = Clip 4
CLIP_INDEX = 3

CLASS_NAMES = ["Upper", "Lower"]


# ============================================================
# MODIFIED VIDEO SWIN ATTENTION FORWARD
# ============================================================

def attention_forward_with_grad(self, x):
    """
    Same computation as torchvision ShiftedWindowAttention3d,
    but stores the REAL attention tensor used in the forward
    computation and retains its gradient.

    This allows us to calculate:

        d(class_score) / d(attention)

    for Gradient Attention Rollout.
    """

    b, t, h, w, c = x.shape

    size_dhw = [t, h, w]

    window_size = self.window_size.copy()
    shift_size = self.shift_size.copy()

    # Adjust window/shift if feature map is smaller
    window_size, shift_size = _get_window_and_shift_size(
        shift_size,
        size_dhw,
        window_size,
    )

    # Store effective values for debugging
    self.effective_window_size = window_size
    self.effective_shift_size = shift_size
    self.input_size_dhw = size_dhw

    relative_position_bias = (
        self.get_relative_position_bias(
            window_size
        )
    )

    # ========================================================
    # Padding
    # ========================================================

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

    # ========================================================
    # Cyclic shift
    # ========================================================

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

    # ========================================================
    # Partition into 3D windows
    # ========================================================

    num_windows = (
        (tp // window_size[0])
        * (hp // window_size[1])
        * (wp // window_size[2])
    )

    # Store for debugging
    self.saved_num_windows = num_windows

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

    # ========================================================
    # Q, K, V
    # ========================================================

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

    # ========================================================
    # REAL ATTENTION
    # ========================================================

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

    # ========================================================
    # Shifted-window attention mask
    # ========================================================

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

    # ========================================================
    # Softmax
    # ========================================================

    attn = F.softmax(
        attn,
        dim=-1,
    )

    # IMPORTANT:
    #
    # This is the REAL attention tensor that is subsequently
    # used for attention @ V.
    #
    # Therefore its gradient is connected to the final class
    # score.

    attn.retain_grad()

    self.saved_attention = attn

    # ========================================================
    # Attention x V
    # ========================================================

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

    # ========================================================
    # Output projection
    # ========================================================

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

    # ========================================================
    # Reverse windows
    # ========================================================

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

    # ========================================================
    # Reverse cyclic shift
    # ========================================================

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

    # ========================================================
    # Remove padding
    # ========================================================

    x = x[
        :,
        :t,
        :h,
        :w,
        :
    ].contiguous()

    return x


# ============================================================
# EXTRACT CLIPS FROM DATASET SAMPLE
# ============================================================

def extract_clips_and_label(sample):

    print(
        "\nSample type:",
        type(sample),
    )

    # --------------------------------------------------------
    # Dictionary output
    # --------------------------------------------------------

    if isinstance(
        sample,
        dict,
    ):

        print(
            "Sample keys:",
            list(sample.keys()),
        )

        for key, value in sample.items():

            if hasattr(
                value,
                "shape",
            ):

                print(
                    f"  {key}: "
                    f"shape={value.shape}"
                )

            else:

                print(
                    f"  {key}: "
                    f"{value}"
                )

        # We already know your dataset uses "clips".
        if "clips" in sample:

            clips = sample[
                "clips"
            ]

            print(
                "\nUsing video tensor key: clips"
            )

        else:

            # Backup search
            clips = None

            for key, value in sample.items():

                if (
                    torch.is_tensor(value)
                    and value.ndim == 5
                ):

                    clips = value

                    print(
                        "\nUsing video tensor key:",
                        key,
                    )

                    break

            if clips is None:

                raise KeyError(
                    "Could not find clip tensor "
                    "inside dataset sample."
                )

        label = sample.get(
            "label",
            None,
        )

        return (
            clips,
            label,
        )

    # --------------------------------------------------------
    # Tuple/list output
    # --------------------------------------------------------

    if isinstance(
        sample,
        (tuple, list),
    ):

        clips = sample[0]

        label = (
            sample[1]
            if len(sample) > 1
            else None
        )

        return (
            clips,
            label,
        )

    raise TypeError(
        "Unsupported dataset "
        f"sample type: {type(sample)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    # ========================================================
    # 2. LOAD VALIDATION METADATA
    # ========================================================

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

    # ========================================================
    # 3. BUILD VALIDATION DATASET
    # ========================================================

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

    # ========================================================
    # 4. FIND TARGET SEGMENT
    # ========================================================

    matches = (
        val_metadata.index[
            val_metadata[
                "segment_id"
            ]
            == TARGET_SEGMENT
        ]
        .tolist()
    )

    if len(matches) == 0:

        raise ValueError(
            f"Segment "
            f"{TARGET_SEGMENT} "
            f"not found."
        )

    dataset_index = matches[0]

    row = val_metadata.iloc[
        dataset_index
    ]

    print(
        "\nSelected segment:"
    )

    print(row)

    original_quadrant = (
        row["label"]
    )

    true_binary = (
        "Upper"
        if original_quadrant
        in ["Q1", "Q2"]
        else "Lower"
    )

    print(
        "\nOriginal quadrant:",
        original_quadrant,
    )

    print(
        "True binary class:",
        true_binary,
    )

    # ========================================================
    # 5. LOAD REAL SEGMENT
    # ========================================================

    sample = dataset[
        dataset_index
    ]

    clips, label = (
        extract_clips_and_label(
            sample
        )
    )

    print(
        "\nSegment tensor shape:",
        clips.shape,
    )

    # Expected:
    #
    # [5,16,3,224,224]

    if clips.ndim != 5:

        raise ValueError(
            "Expected clips tensor "
            "[num_clips,T,C,H,W], "
            f"got {clips.shape}"
        )

    # ========================================================
    # 6. SELECT CLIP 4
    # ========================================================

    clip = clips[
        CLIP_INDEX
    ]

    print(
        "\nSelected Clip 4 shape:",
        clip.shape,
    )

    # ========================================================
    # 7. CONVERT TO VIDEOSWIN INPUT
    # ========================================================

    # [T,C,H,W]
    #
    # ->
    #
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

    # ========================================================
    # 8. NORMALIZATION
    # ========================================================

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

    x = x.to(
        device
    )

    print(
        "\nModel input shape:",
        x.shape,
    )

    # ========================================================
    # 9. BUILD BINARY VIDEOSWIN
    # ========================================================

    model = swin3d_t(
        weights=None
    )

    model.head = (
        torch.nn.Linear(
            model.head.in_features,
            2,
        )
    )

    # ========================================================
    # 10. LOAD TRAINED CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    print(
        "\nCheckpoint keys:",
        checkpoint.keys()
        if isinstance(
            checkpoint,
            dict,
        )
        else "state_dict only",
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

    print(
        "Checkpoint loaded."
    )

    # ========================================================
    # 11. PATCH BOTH FINAL-STAGE ATTENTION BLOCKS
    # ========================================================

    attention_block_1 = (
        model
        .features[6][0]
        .attn
    )

    attention_block_2 = (
        model
        .features[6][1]
        .attn
    )

    attention_block_1.forward = (
        types.MethodType(
            attention_forward_with_grad,
            attention_block_1,
        )
    )

    attention_block_2.forward = (
        types.MethodType(
            attention_forward_with_grad,
            attention_block_2,
        )
    )

    print(
        "\nPatched attention blocks:"
    )

    print(
        "\nBlock 1:"
    )

    print(
        "  model.features[6][0].attn"
    )

    print(
        "  Configured window:",
        attention_block_1.window_size,
    )

    print(
        "  Configured shift:",
        attention_block_1.shift_size,
    )

    print(
        "  Heads:",
        attention_block_1.num_heads,
    )

    print(
        "\nBlock 2:"
    )

    print(
        "  model.features[6][1].attn"
    )

    print(
        "  Configured window:",
        attention_block_2.window_size,
    )

    print(
        "  Configured shift:",
        attention_block_2.shift_size,
    )

    print(
        "  Heads:",
        attention_block_2.num_heads,
    )

    # ========================================================
    # 12. FORWARD
    # ========================================================

    model.zero_grad()

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

    predicted_name = (
        CLASS_NAMES[
            predicted_class
        ]
    )

    print(
        "\nLogits:"
    )

    print(logits)

    print(
        "\nProbabilities:"
    )

    print(
        probabilities
    )

    print(
        "\nTrue class:",
        true_binary,
    )

    print(
        "Predicted class:",
        predicted_name,
    )

    print(
        "P(Upper):",
        probabilities[
            0,
            0,
        ].item(),
    )

    print(
        "P(Lower):",
        probabilities[
            0,
            1,
        ].item(),
    )

    print(
        "Correct:",
        predicted_name
        == true_binary,
    )

    # ========================================================
    # 13. BACKPROP PREDICTED CLASS
    # ========================================================

    score = logits[
        0,
        predicted_class,
    ]

    score.backward()

    # ========================================================
    # 14. EXTRACT BOTH ATTENTION MATRICES
    # ========================================================

    attention_1 = (
        attention_block_1
        .saved_attention
    )

    attention_2 = (
        attention_block_2
        .saved_attention
    )

    gradient_1 = (
        attention_1.grad
    )

    gradient_2 = (
        attention_2.grad
    )

    # ========================================================
    # BLOCK 1 INFO
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "BLOCK 1"
    )

    print(
        "========================================"
    )

    print(
        "Input feature grid:",
        attention_block_1.input_size_dhw,
    )

    print(
        "Effective window:",
        attention_block_1.effective_window_size,
    )

    print(
        "Effective shift:",
        attention_block_1.effective_shift_size,
    )

    print(
        "Number of windows:",
        attention_block_1.saved_num_windows,
    )

    print(
        "Attention shape:",
        attention_1.shape,
    )

    print(
        "Gradient exists:",
        gradient_1 is not None,
    )

    if gradient_1 is not None:

        print(
            "Gradient shape:",
            gradient_1.shape,
        )

        print(
            "Gradient |mean|:",
            gradient_1
            .abs()
            .mean()
            .item(),
        )

    # ========================================================
    # BLOCK 2 INFO
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "BLOCK 2"
    )

    print(
        "========================================"
    )

    print(
        "Input feature grid:",
        attention_block_2.input_size_dhw,
    )

    print(
        "Effective window:",
        attention_block_2.effective_window_size,
    )

    print(
        "Effective shift:",
        attention_block_2.effective_shift_size,
    )

    print(
        "Number of windows:",
        attention_block_2.saved_num_windows,
    )

    print(
        "Attention shape:",
        attention_2.shape,
    )

    print(
        "Gradient exists:",
        gradient_2 is not None,
    )

    if gradient_2 is not None:

        print(
            "Gradient shape:",
            gradient_2.shape,
        )

        print(
            "Gradient |mean|:",
            gradient_2
            .abs()
            .mean()
            .item(),
        )

    # ========================================================
    # 15. VALIDATE
    # ========================================================

    if gradient_1 is None:

        raise RuntimeError(
            "Gradient missing "
            "from Block 1."
        )

    if gradient_2 is None:

        raise RuntimeError(
            "Gradient missing "
            "from Block 2."
        )

        # ========================================================
    # 16. GRADIENT-WEIGHTED ATTENTION
    # ========================================================

    # Positive class-relevant attention:
    #
    # G = ReLU(A * d(score)/dA)

    grad_attention_1 = torch.relu(
        attention_1 * gradient_1
    )

    grad_attention_2 = torch.relu(
        attention_2 * gradient_2
    )

    # Average over the 24 attention heads:
    #
    # [1,24,392,392]
    # ->
    # [1,392,392]

    G1 = grad_attention_1.mean(dim=1)
    G2 = grad_attention_2.mean(dim=1)

    print("\nGradient-weighted attention:")

    print(
        "G1:",
        G1.shape,
    )

    print(
        "G2:",
        G2.shape,
    )

    # ========================================================
    # 17. ADD RESIDUAL CONNECTIONS
    # ========================================================

    num_tokens = G1.shape[-1]

    identity = torch.eye(
        num_tokens,
        device=device,
        dtype=G1.dtype,
    ).unsqueeze(0)

    R1 = G1 + identity
    R2 = G2 + identity

    print(
        "\nResidual connections added."
    )

    # ========================================================
    # 18. ROW NORMALIZATION
    # ========================================================

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

    print(
        "R1 row-sum mean:",
        R1.sum(dim=-1)
        .mean()
        .item(),
    )

    print(
        "R2 row-sum mean:",
        R2.sum(dim=-1)
        .mean()
        .item(),
    )

    # ========================================================
    # 19. ATTENTION ROLLOUT
    # ========================================================

    # Block 1 happens first,
    # then Block 2.
    #
    # Therefore:
    #
    # final -> Block 2 -> Block 1
    #
    # R = R2 @ R1

    rollout = torch.bmm(
        R2,
        R1,
    )

    print(
        "\nRollout matrix:",
        rollout.shape,
    )

    # Expected:
    #
    # [1,392,392]

    # ========================================================
    # 20. GLOBAL-AVERAGE OUTPUT RELEVANCE
    # ========================================================

    # VideoSwin has no CLS token.
    #
    # Its final representation is obtained by global
    # average pooling over the final spatiotemporal tokens.
    #
    # Therefore we start from uniform relevance over all
    # final query tokens and propagate it through rollout.

    output_relevance = torch.ones(
        1,
        num_tokens,
        device=device,
        dtype=rollout.dtype,
    )

    output_relevance = (
        output_relevance
        / num_tokens
    )

    # [1,392] @ [1,392,392]
    # ->
    # [1,392]

    token_relevance = torch.bmm(
        output_relevance.unsqueeze(1),
        rollout,
    ).squeeze(1)

    print(
        "Token relevance:",
        token_relevance.shape,
    )

    # ========================================================
    # 21. RESHAPE TOKENS INTO SPATIOTEMPORAL GRID
    # ========================================================

    # 392 = 8 * 7 * 7

    token_relevance = (
        token_relevance
        .reshape(
            1,
            8,
            7,
            7,
        )
    )

    print(
        "Coarse rollout volume:",
        token_relevance.shape,
    )

    # ========================================================
    # 22. NORMALIZE RELEVANCE
    # ========================================================

    token_relevance = (
        token_relevance
        - token_relevance.min()
    )

    max_value = (
        token_relevance.max()
    )

    if max_value > 0:

        token_relevance = (
            token_relevance
            / max_value
        )

    # ========================================================
    # 23. UPSAMPLE TO ORIGINAL CLIP RESOLUTION
    # ========================================================

    # [1,8,7,7]
    # ->
    # [1,1,8,7,7]

    rollout_volume = (
        token_relevance
        .unsqueeze(1)
    )

    rollout_volume = F.interpolate(
        rollout_volume,
        size=(
            16,
            224,
            224,
        ),
        mode="trilinear",
        align_corners=False,
    )

    # ->
    # [16,224,224]

    rollout_volume = (
        rollout_volume[
            0,
            0,
        ]
        .detach()
        .cpu()
        .numpy()
    )

    print(
        "Upsampled rollout volume:",
        rollout_volume.shape,
    )

    # ========================================================
    # 24. SAVE RAW ROLLOUT
    # ========================================================

    import numpy as np
    import matplotlib.pyplot as plt

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "gradient_attention_rollout"
        / f"segment_{TARGET_SEGMENT}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_dir
        / "rollout_raw.npy",
        rollout_volume,
    )

    print(
        "\nSaved raw rollout:"
    )

    print(
        output_dir
        / "rollout_raw.npy"
    )

    # ========================================================
    # 25. PREPARE ORIGINAL VIDEO FRAMES
    # ========================================================

    frames = (
        clip
        .detach()
        .cpu()
        .numpy()
    )

    # [T,C,H,W]
    # ->
    # [T,H,W,C]

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

    # ========================================================
    # 26. SAVE 16 ROLLOUT OVERLAYS
    # ========================================================

    for frame_idx in range(16):

        frame = frames[
            frame_idx
        ]

        heatmap = rollout_volume[
            frame_idx
        ]

        fig, ax = plt.subplots(
            figsize=(
                6,
                6,
            )
        )

        ax.imshow(
            frame
        )

        ax.imshow(
            heatmap,
            cmap="jet",
            alpha=0.45,
            vmin=0,
            vmax=1,
        )

        ax.set_title(
            f"Gradient Attention Rollout\n"
            f"Segment {TARGET_SEGMENT} | "
            f"Clip {CLIP_INDEX + 1} | "
            f"Frame {frame_idx + 1}\n"
            f"True: {true_binary} | "
            f"Pred: {predicted_name}"
        )

        ax.axis(
            "off"
        )

        save_path = (
            output_dir
            / f"frame_{frame_idx + 1:02d}.png"
        )

        plt.tight_layout()

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    # ========================================================
    # 27. TEMPORAL RELEVANCE
    # ========================================================

    # Average spatial relevance for every frame

    temporal_relevance = (
        rollout_volume
        .mean(
            axis=(
                1,
                2,
            )
        )
    )

    np.save(
        output_dir
        / "temporal_relevance.npy",
        temporal_relevance,
    )

    plt.figure(
        figsize=(
            8,
            4,
        )
    )

    plt.plot(
        range(
            1,
            17,
        ),
        temporal_relevance,
        marker="o",
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Mean rollout relevance"
    )

    plt.title(
        f"Temporal Gradient Attention Rollout\n"
        f"Segment {TARGET_SEGMENT} | "
        f"Clip {CLIP_INDEX + 1}"
    )

    plt.xticks(
        range(
            1,
            17,
        )
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "temporal_relevance.png",
        dpi=150,
    )

    plt.close()

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "GRADIENT ATTENTION ROLLOUT COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Segment: {TARGET_SEGMENT}"
    )

    print(
        f"Clip: {CLIP_INDEX + 1}"
    )

    print(
        f"True: {true_binary}"
    )

    print(
        f"Predicted: {predicted_name}"
    )

    print(
        "\nRollout matrix:",
        rollout.shape,
    )

    print(
        "Coarse volume: [8,7,7]"
    )

    print(
        "Final volume:",
        rollout_volume.shape,
    )

    print(
        "\nSaved 16 rollout overlays to:"
    )

    print(
        output_dir
    )

if __name__ == "__main__":
    main()