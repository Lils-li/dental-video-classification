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
import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------

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
    / "contact_sheets"
)


# Representative temporal positions
POSITIONS = [0.10, 0.30, 0.50, 0.70, 0.90]


def get_frame_number(start_frame, end_frame, position):
    """
    Return the frame located at a relative temporal position
    inside the segment.
    """

    segment_length = end_frame - start_frame

    frame_number = round(
        start_frame + position * segment_length
    )

    return frame_number


def main():

    metadata = pd.read_csv(METADATA_PATH)

    train_df = metadata[
        metadata["split"] == "train"
    ].copy()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Train segments:", len(train_df))
    print("Output directory:", OUTPUT_DIR)
    print()

    for _, row in train_df.iterrows():

        segment_id = int(row["segment_id"])
        label = row["label"]
        uuid = row["uuid"]

        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])

        num_frames = (
            end_frame
            - start_frame
            + 1
        )

        gt_dir = (
            DATA_ROOT
            / "train"
            / uuid
            / "GT"
        )

        frame_numbers = [
            get_frame_number(
                start_frame,
                end_frame,
                position
            )
            for position in POSITIONS
        ]

        fig, axes = plt.subplots(
            1,
            5,
            figsize=(15, 4)
        )

        for ax, position, frame_number in zip(
            axes,
            POSITIONS,
            frame_numbers
        ):

            frame_path = (
                gt_dir
                / f"{frame_number:05d}.png"
            )

            image = cv2.imread(
                str(frame_path)
            )

            if image is None:

                ax.text(
                    0.5,
                    0.5,
                    "Frame not found",
                    ha="center",
                    va="center"
                )

                ax.axis("off")
                continue

            # OpenCV loads BGR.
            # Matplotlib expects RGB.
            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            ax.imshow(image_rgb)

            ax.set_title(
                f"{int(position * 100)}%\n"
                f"Frame {frame_number}"
            )

            ax.axis("off")

        fig.suptitle(
            f"Segment {segment_id} | "
            f"{label} | "
            f"{num_frames} frames",
            fontsize=16
        )

        plt.tight_layout()

        output_path = (
            OUTPUT_DIR
            / f"segment_{segment_id:03d}_{label}.jpg"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            f"Saved segment {segment_id:03d} "
            f"| {label} "
            f"| {num_frames} frames"
        )

    print()
    print("Finished.")
    print("Contact sheets saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()