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
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRADCAM_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "gradcam_comparison"
)

SUMMARY_CSV = (
    GRADCAM_DIR
    / "comparison_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "gradcam_temporal_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load summary
# ============================================================

summary = pd.read_csv(SUMMARY_CSV)


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


# ============================================================
# Load CAM and calculate temporal profile
# ============================================================

temporal_profiles = {}

rows = []

for _, row in summary.iterrows():

    segment_id = int(row["segment_id"])

    cam_path = (
        GRADCAM_DIR
        / f"segment_{segment_id}"
        / "gradcam_raw.npy"
    )

    cam = np.load(cam_path)

    print(
        f"Segment {segment_id}: "
        f"CAM shape = {cam.shape}"
    )

    # Mean spatial activation for every timestep.
    # [T,H,W] -> [T]
    temporal_profile = cam.mean(
        axis=(1, 2)
    )

    temporal_profiles[segment_id] = (
        temporal_profile
    )

    result = {
        "segment_id": segment_id,
        "original_label": row["original_label"],
        "true": row["true"],
        "predicted": row["predicted"],
        "group": row["group"],
    }

    for t, value in enumerate(
        temporal_profile,
        start=1,
    ):
        result[f"frame_{t:02d}"] = value

    rows.append(result)


temporal_df = pd.DataFrame(rows)

temporal_df.to_csv(
    OUTPUT_DIR / "temporal_profiles.csv",
    index=False,
)


# ============================================================
# Plot every segment
# ============================================================

fig, axes = plt.subplots(
    3,
    5,
    figsize=(18, 11),
)

axes = axes.flatten()

for ax in axes:
    ax.axis("off")


for ax, (_, row) in zip(
    axes,
    summary.iterrows(),
):

    segment_id = int(
        row["segment_id"]
    )

    profile = temporal_profiles[
        segment_id
    ]

    frames = np.arange(
        1,
        len(profile) + 1
    )

    ax.axis("on")

    ax.plot(
        frames,
        profile,
        marker="o",
    )

    ax.set_title(
        f"Segment {segment_id} ({row['original_label']})\n"
        f"{row['group']}"
    )

    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean CAM")

    ax.set_xticks(
        [1, 4, 8, 12, 16]
    )

    ax.grid(
        alpha=0.25
    )


fig.suptitle(
    "Temporal Grad-CAM Profiles",
    fontsize=18,
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "all_temporal_profiles.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Q3 controlled comparison
# ============================================================

q3 = summary[
    summary["original_label"] == "Q3"
].copy()

fig, ax = plt.subplots(
    figsize=(10, 6)
)


for _, row in q3.iterrows():

    segment_id = int(
        row["segment_id"]
    )

    profile = temporal_profiles[
        segment_id
    ]

    frames = np.arange(
        1,
        len(profile) + 1
    )

    linestyle = (
        "--"
        if row["group"] == "Lower -> Upper"
        else "-"
    )

    ax.plot(
        frames,
        profile,
        marker="o",
        linestyle=linestyle,
        label=(
            f"Seg {segment_id} | "
            f"{row['group']}"
        ),
    )


ax.set_title(
    "Q3 Temporal Grad-CAM: "
    "Misclassified vs Correct"
)

ax.set_xlabel(
    "Frame within Clip 4"
)

ax.set_ylabel(
    "Mean normalized CAM activation"
)

ax.set_xticks(
    range(1, 17)
)

ax.grid(
    alpha=0.25
)

ax.legend(
    fontsize=8
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "q3_temporal_comparison.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close(fig)


print()
print("Saved:")
print(
    OUTPUT_DIR
    / "temporal_profiles.csv"
)

print(
    OUTPUT_DIR
    / "all_temporal_profiles.png"
)

print(
    OUTPUT_DIR
    / "q3_temporal_comparison.png"
)