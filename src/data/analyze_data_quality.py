# Source preservation: command-line execution resolves repository-relative inputs.
if __name__ == '__main__':
    import os as _preserve_os, sys as _preserve_sys
    from pathlib import Path as _PreservePath
    _preserve_root = _PreservePath(__file__).resolve().parents[1]
    _preserve_os.chdir(_preserve_root)
    _preserve_sys.path.insert(0, str(_preserve_root))
    _preserve_sys.path.insert(0, str(_preserve_root / 'src' / 'data'))

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "data_quality"
    / "frame_quality_metrics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "data_quality"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "segment_quality_summary.csv"
)


def main():

    # --------------------------------------------------
    # 1. Load frame-level audit
    # --------------------------------------------------

    df = pd.read_csv(INPUT_PATH)

    print("Frame-level rows:", len(df))
    print("Segments:", df["segment_id"].nunique())

    # --------------------------------------------------
    # 2. Aggregate frames -> segments
    # --------------------------------------------------

    summary = (
        df.groupby(
            ["segment_id", "label", "uuid"]
        )
        .agg(
            num_frames=("frame_number", "count"),

            brightness_mean=("brightness", "mean"),
            brightness_median=("brightness", "median"),
            brightness_min=("brightness", "min"),
            brightness_max=("brightness", "max"),
            brightness_std=("brightness", "std"),

            contrast_mean=("contrast", "mean"),
            contrast_median=("contrast", "median"),
            contrast_min=("contrast", "min"),
            contrast_max=("contrast", "max"),
            contrast_std=("contrast", "std"),

            sharpness_mean=("sharpness", "mean"),
            sharpness_median=("sharpness", "median"),
            sharpness_min=("sharpness", "min"),
            sharpness_max=("sharpness", "max"),
            sharpness_std=("sharpness", "std"),
        )
        .reset_index()
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    print()
    print("Saved segment summary:")
    print(SUMMARY_PATH)

    # --------------------------------------------------
    # 3. Class-level summary
    # --------------------------------------------------

    # --------------------------------------------------
# 3. Class-level summary
# --------------------------------------------------

    class_summary = (
    summary
    .groupby("label")
    .agg(
        num_segments=("segment_id", "count"),
        total_frames=("num_frames", "sum"),
        median_segment_length=("num_frames", "median"),

        median_brightness=("brightness_median", "median"),
        median_contrast=("contrast_median", "median"),
        median_sharpness=("sharpness_median", "median"),
    )
    .round(2)
)

    print()
    print("Class-level dataset summary:")
    print(class_summary)

    print()
    print("Median segment-level values by class:")
    print(class_summary)

    # --------------------------------------------------
    # 4. Boxplots
    # --------------------------------------------------

    metrics = [
        ("brightness_median", "Brightness"),
        ("contrast_median", "Contrast"),
        ("sharpness_median", "Sharpness"),
        ("num_frames", "Segment length"),
    ]

    class_order = ["Q1", "Q2", "Q3", "Q4"]

    for column, title in metrics:

        data = [
            summary.loc[
                summary["label"] == label,
                column
            ].dropna()
            for label in class_order
        ]

        plt.figure(figsize=(8, 5))

        plt.boxplot(
            data,
            tick_labels=class_order
        )

        plt.xlabel("Quadrant")
        plt.ylabel(title)
        plt.title(f"{title} by quadrant")

        plt.tight_layout()

        output_path = (
            OUTPUT_DIR
            / f"{column}_by_class.png"
        )

        plt.savefig(
            output_path,
            dpi=200
        )

        plt.close()

        print("Saved:", output_path)


if __name__ == "__main__":
    main()