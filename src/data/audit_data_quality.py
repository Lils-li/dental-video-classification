# Source preservation: command-line execution resolves repository-relative inputs.
if __name__ == '__main__':
    import os as _preserve_os, sys as _preserve_sys
    from pathlib import Path as _PreservePath
    _preserve_root = _PreservePath(__file__).resolve().parents[1]
    _preserve_os.chdir(_preserve_root)
    _preserve_sys.path.insert(0, str(_preserve_root))
    _preserve_sys.path.insert(0, str(_preserve_root / 'src' / 'data'))

from pathlib import Path

import cv2
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "quadrant_metadata.csv"
)

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Vident-real"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "data_quality"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "frame_quality_metrics.csv"
)


# -------------------------
# Quality metrics
# -------------------------

def calculate_brightness(gray):
    return float(gray.mean())


def calculate_contrast(gray):
    return float(gray.std())


def calculate_sharpness(gray):
    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )
    return float(laplacian.var())


# -------------------------
# Main audit
# -------------------------

def main():

    metadata = pd.read_csv(METADATA_PATH)

    train_df = metadata[
        metadata["split"] == "train"
    ].copy()

    print(f"Train segments: {len(train_df)}")

    results = []

    for _, row in train_df.iterrows():

        segment_id = int(row["segment_id"])
        label = row["label"]
        uuid = row["uuid"]

        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])

        gt_dir = (
            DATA_ROOT
            / "train"
            / uuid
            / "GT"
        )

        print(
            f"Segment {segment_id:02d} | "
            f"{label} | "
            f"frames {start_frame}-{end_frame}"
        )

        for frame_number in range(
            start_frame,
            end_frame + 1
        ):

            frame_path = (
                gt_dir
                / f"{frame_number:05d}.png"
            )

            image = cv2.imread(str(frame_path))

            if image is None:
                print(
                    f"WARNING: could not read "
                    f"{frame_path}"
                )
                continue

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            results.append(
                {
                    "segment_id": segment_id,
                    "label": label,
                    "uuid": uuid,
                    "frame_number": frame_number,
                    "brightness": calculate_brightness(gray),
                    "contrast": calculate_contrast(gray),
                    "sharpness": calculate_sharpness(gray),
                }
            )

    results_df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print(f"Processed frames: {len(results_df)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()