# Source preservation: command-line execution resolves repository-relative inputs.
if __name__ == '__main__':
    import os as _preserve_os, sys as _preserve_sys
    from pathlib import Path as _PreservePath
    _preserve_root = _PreservePath(__file__).resolve().parents[0]
    _preserve_os.chdir(_preserve_root)
    _preserve_sys.path.insert(0, str(_preserve_root))
    _preserve_sys.path.insert(0, str(_preserve_root / 'src' / 'data'))

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "gradcam_comparison"
)

SUMMARY_CSV = INPUT_DIR / "comparison_summary.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "gradcam_validation_summary"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load prediction summary
# ============================================================

summary = pd.read_csv(SUMMARY_CSV)

print("Loaded segments:", len(summary))


# ============================================================
# Assign each segment to a prediction group
# ============================================================

def get_group(row):

    if row["true"] == "Upper" and row["predicted"] == "Upper":
        return "Correct Upper"

    if row["true"] == "Lower" and row["predicted"] == "Lower":
        return "Correct Lower"

    if row["true"] == "Lower" and row["predicted"] == "Upper":
        return "Lower -> Upper"

    if row["true"] == "Upper" and row["predicted"] == "Lower":
        return "Upper -> Lower"

    return "Other"


summary["group"] = summary.apply(
    get_group,
    axis=1,
)

print()
print(summary[
    [
        "segment_id",
        "original_label",
        "true",
        "predicted",
        "group",
    ]
].to_string(index=False))


# ============================================================
# Representative frame
# ============================================================

# We already generated 16 Grad-CAM frames per segment.
# Use the middle frame as a compact visual summary.
REPRESENTATIVE_FRAME = 8


def load_segment_image(segment_id):

    image_path = (
        INPUT_DIR
        / f"segment_{segment_id}"
        / f"frame_{REPRESENTATIVE_FRAME:02d}.png"
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Could not find: {image_path}"
        )

    return Image.open(image_path).convert("RGB")


# ============================================================
# Create one contact sheet for each group
# ============================================================

group_order = [
    "Lower -> Upper",
    "Correct Lower",
    "Correct Upper",
    "Upper -> Lower",
]


for group_name in group_order:

    group_df = summary[
        summary["group"] == group_name
    ].copy()

    if len(group_df) == 0:
        print(
            f"\nSkipping {group_name}: no segments."
        )
        continue

    n = len(group_df)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(5 * n, 5),
        squeeze=False,
    )

    axes = axes[0]

    for ax, (_, row) in zip(
        axes,
        group_df.iterrows(),
    ):

        segment_id = int(
            row["segment_id"]
        )

        image = load_segment_image(
            segment_id
        )

        ax.imshow(image)

        confidence = (
            row["p_upper"]
            if row["predicted"] == "Upper"
            else row["p_lower"]
        )

        ax.set_title(
            f"Segment {segment_id} ({row['original_label']})\n"
            f"True: {row['true']} | "
            f"Pred: {row['predicted']}\n"
            f"Confidence: {confidence:.3f}",
            fontsize=11,
        )

        ax.axis("off")

    fig.suptitle(
        f"{group_name} — Grad-CAM comparison",
        fontsize=16,
    )

    fig.tight_layout()

    safe_name = (
        group_name
        .lower()
        .replace(" ", "_")
        .replace("->", "to")
    )

    output_path = (
        OUTPUT_DIR
        / f"{safe_name}.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Q3 controlled comparison
# ============================================================

q3 = summary[
    summary["original_label"] == "Q3"
].copy()

q3_wrong = q3[
    q3["group"] == "Lower -> Upper"
]

q3_correct = q3[
    q3["group"] == "Correct Lower"
]

comparison_rows = []

for _, row in q3_wrong.iterrows():
    comparison_rows.append(
        ("Misclassified Q3 -> Upper", row)
    )

for _, row in q3_correct.iterrows():
    comparison_rows.append(
        ("Correct Q3 -> Lower", row)
    )


if comparison_rows:

    n = len(comparison_rows)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
    )

    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, (category, row) in zip(
        axes,
        comparison_rows,
    ):

        segment_id = int(
            row["segment_id"]
        )

        image = load_segment_image(
            segment_id
        )

        ax.imshow(image)
        ax.axis("off")

        ax.set_title(
            f"{category}\n"
            f"Segment {segment_id} | "
            f"P(Upper)={row['p_upper']:.3f}",
            fontsize=11,
        )

    fig.suptitle(
        "Q3 Controlled Comparison: "
        "Misclassified vs Correct Lower Segments",
        fontsize=16,
    )

    fig.tight_layout()

    q3_output = (
        OUTPUT_DIR
        / "q3_controlled_comparison.png"
    )

    fig.savefig(
        q3_output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {q3_output}"
    )


print()
print("Finished.")
print("Outputs saved to:")
print(OUTPUT_DIR)