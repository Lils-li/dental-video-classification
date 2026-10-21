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
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "explainability"
    / "gradcam_style"
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
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Representative validation examples
# ============================================================

EXAMPLES = {
    "Q1_correct": 10,
    "Q2_correct": 5,
    "Q3_correct": 6,
    "Q4_wrong_to_Q1": 2,
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
    """
    Dataset:
        [T, C, H, W]

    VideoSwin:
        [B, C, T, H, W]
    """

    x = clip.to(DEVICE)

    x = (
        x
        .permute(1, 0, 2, 3)
        .unsqueeze(0)
    )

    x = (x - mean) / std

    return x


# ============================================================
# Capture final feature tensor
#
# VideoSwin model.features returns:
# [B, T, H, W, C]
# ============================================================

captured = {}


def feature_hook(
    module,
    inputs,
    output
):
    captured["features"] = output

    # During normal no_grad prediction:
    # requires_grad=False
    #
    # During CAM computation:
    # requires_grad=True
    if output.requires_grad:
        output.retain_grad()


hook = model.features.register_forward_hook(
    feature_hook
)


# ============================================================
# Grad-CAM-style class-specific attribution
# ============================================================

def compute_cam(
    model_input,
    target_class
):
    """
    Computes class-specific Grad-CAM-style attribution
    using the final VideoSwin feature tensor.

    This is not classical CNN Grad-CAM.
    """

    captured.clear()

    # Important:
    # backbone parameters are frozen, but we still need
    # an autograd graph through the backbone.
    model_input = (
        model_input
        .detach()
        .clone()
    )

    model_input.requires_grad_(True)

    model.zero_grad(
        set_to_none=True
    )

    logits = model(
        model_input
    )

    probs = torch.softmax(
        logits.detach(),
        dim=1
    )

    pred_id = int(
        probs.argmax(
            dim=1
        ).item()
    )

    target_score = logits[
        0,
        target_class
    ]

    target_score.backward()


    if "features" not in captured:
        raise RuntimeError(
            "Feature hook did not capture output."
        )


    features = captured[
        "features"
    ]

    gradients = features.grad


    if gradients is None:
        raise RuntimeError(
            "Feature gradients were not captured."
        )


    # ========================================================
    # Expected:
    #
    # features:
    # [B, T, H, W, C]
    #
    # gradients:
    # [B, T, H, W, C]
    # ========================================================


    # Channel weights:
    # average gradients over temporal + spatial dimensions
    weights = gradients.mean(
        dim=(1, 2, 3),
        keepdim=True
    )


    # Weighted combination of feature channels
    cam = (
        weights
        * features
    ).sum(
        dim=-1
    )


    # Positive evidence only
    cam = F.relu(
        cam
    )


    # Current:
    # [B, T, H, W]
    #
    # Interpolate back to:
    # [B, 16, 224, 224]
    cam = cam.unsqueeze(1)

    cam = F.interpolate(
        cam,
        size=(
            16,
            224,
            224
        ),
        mode="trilinear",
        align_corners=False
    )

    cam = cam.squeeze(1)


    # ========================================================
    # Normalize independently per clip
    # ========================================================

    minimum = cam.amin(
        dim=(1, 2, 3),
        keepdim=True
    )

    maximum = cam.amax(
        dim=(1, 2, 3),
        keepdim=True
    )

    cam = (
        cam - minimum
    ) / (
        maximum - minimum + 1e-8
    )


    return (
        cam.detach(),
        probs.cpu(),
        pred_id
    )


# ============================================================
# Visualization
# ============================================================

def save_figure(
    original_clip,
    cam,
    output_path,
    title
):
    """
    Shows middle frame of the 16-frame clip.
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
        cam[
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
    axes[0].imshow(
        frame
    )

    axes[0].set_title(
        "GT frame"
    )

    axes[0].axis(
        "off"
    )


    # Heatmap
    axes[1].imshow(
        heatmap,
        cmap="jet"
    )

    axes[1].set_title(
        "Grad-CAM-style attribution"
    )

    axes[1].axis(
        "off"
    )


    # Overlay
    axes[2].imshow(
        frame
    )

    axes[2].imshow(
        heatmap,
        cmap="jet",
        alpha=0.45
    )

    axes[2].set_title(
        "Overlay"
    )

    axes[2].axis(
        "off"
    )


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

    plt.close(
        fig
    )


# ============================================================
# Main loop
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

        clips = sample[
            "clips"
        ]

        true_id = int(
            sample[
                "label"
            ].item()
        )


        print(
            "True label:",
            CLASS_NAMES[
                true_id
            ]
        )

        print(
            "Segment:",
            sample[
                "segment_id"
            ]
        )

        print(
            "UUID:",
            sample[
                "uuid"
            ]
        )


        example_dir = (
            OUTPUT_ROOT
            / example_name
        )

        example_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        # ====================================================
        # Process all 5 clips
        # ====================================================

        for clip_idx in range(
            clips.shape[0]
        ):

            original_clip = clips[
                clip_idx
            ]

            model_input = (
                prepare_clip(
                    original_clip
                )
            )


            # ------------------------------------------------
            # Normal prediction only
            # ------------------------------------------------

            with torch.no_grad():

                normal_logits = model(
                    model_input
                )

                normal_probs = (
                    torch.softmax(
                        normal_logits,
                        dim=1
                    )
                )


            pred_id = int(
                normal_probs.argmax(
                    dim=1
                ).item()
            )

            pred_conf = float(
                normal_probs[
                    0,
                    pred_id
                ].item()
            )


            # =================================================
            # Q1 / Q2 / Q3 examples:
            # explain predicted class
            # =================================================

            if true_id != 3:

                cam, probs, _ = (
                    compute_cam(
                        model_input,
                        pred_id
                    )
                )

                confidence = float(
                    probs[
                        0,
                        pred_id
                    ].item()
                )

                output_path = (
                    example_dir
                    / (
                        f"clip_{clip_idx + 1}_"
                        f"target_{CLASS_NAMES[pred_id]}.png"
                    )
                )

                title = (
                    f"{example_name} | "
                    f"clip {clip_idx + 1} | "
                    f"true={CLASS_NAMES[true_id]} | "
                    f"pred={CLASS_NAMES[pred_id]} | "
                    f"CAM target={CLASS_NAMES[pred_id]} | "
                    f"conf={confidence:.3f}"
                )

                save_figure(
                    original_clip,
                    cam[0],
                    output_path,
                    title
                )


                print(
                    f"Clip {clip_idx + 1}: "
                    f"true={CLASS_NAMES[true_id]}, "
                    f"pred={CLASS_NAMES[pred_id]}, "
                    f"conf={confidence:.3f}"
                )

                print(
                    "  CAM target:",
                    CLASS_NAMES[
                        pred_id
                    ]
                )

                print(
                    "  Saved:",
                    output_path
                )


            # =================================================
            # Q4 failure:
            #
            # save two explanations per clip
            #
            # A) predicted-class evidence
            # B) true-Q4 evidence
            # =================================================

            else:

                # ---------------------------------------------
                # A. Predicted class
                # ---------------------------------------------

                cam_pred, probs_pred, _ = (
                    compute_cam(
                        model_input,
                        pred_id
                    )
                )

                predicted_confidence = float(
                    probs_pred[
                        0,
                        pred_id
                    ].item()
                )

                output_pred = (
                    example_dir
                    / (
                        f"clip_{clip_idx + 1}_"
                        f"target_pred_"
                        f"{CLASS_NAMES[pred_id]}.png"
                    )
                )

                title_pred = (
                    f"{example_name} | "
                    f"clip {clip_idx + 1} | "
                    f"true=Q4 | "
                    f"pred={CLASS_NAMES[pred_id]} | "
                    f"CAM target={CLASS_NAMES[pred_id]} | "
                    f"conf={predicted_confidence:.3f}"
                )

                save_figure(
                    original_clip,
                    cam_pred[0],
                    output_pred,
                    title_pred
                )


                # ---------------------------------------------
                # B. True Q4 class
                # ---------------------------------------------

                cam_true, probs_true, _ = (
                    compute_cam(
                        model_input,
                        true_id
                    )
                )

                q4_probability = float(
                    probs_true[
                        0,
                        true_id
                    ].item()
                )

                output_true = (
                    example_dir
                    / (
                        f"clip_{clip_idx + 1}_"
                        f"target_true_Q4.png"
                    )
                )

                title_true = (
                    f"{example_name} | "
                    f"clip {clip_idx + 1} | "
                    f"true=Q4 | "
                    f"pred={CLASS_NAMES[pred_id]} | "
                    f"CAM target=Q4 | "
                    f"P(Q4)={q4_probability:.3f}"
                )

                save_figure(
                    original_clip,
                    cam_true[0],
                    output_true,
                    title_true
                )


                print(
                    f"Clip {clip_idx + 1}: "
                    f"true=Q4, "
                    f"pred={CLASS_NAMES[pred_id]}, "
                    f"pred_conf={pred_conf:.3f}"
                )

                print(
                    "  Saved predicted-class map:",
                    output_pred
                )

                print(
                    "  Saved true-Q4 map:",
                    output_true
                )


finally:

    hook.remove()


# ============================================================
# Finished
# ============================================================

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
    "No test data was used."
)

print(
    "These are Grad-CAM-style "
    "class-specific spatio-temporal "
    "attributions on VideoSwin features."
)

print(
    "They are NOT classical CNN Grad-CAM."
)