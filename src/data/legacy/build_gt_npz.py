# LEGACY / UNVALIDATED: requires unavailable historical CSVs; processes test data too.
#еще не запускали. нет запука
from pathlib import Path
import io
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "Vident-real.zip"

# Put old metadata CSV files here:
META_DIR = PROJECT_ROOT / "data" / "labels"

TRAIN_VAL_META = META_DIR / "segments_with_uuid_92.csv"
TEST_META = META_DIR / "test_multiclip_5x16_metadata_27.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "clips_5x16_gt_npz"
OUTPUT_META_DIR = PROJECT_ROOT / "data" / "processed" / "metadata_gt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_META_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Clip settings
# Must match old INPUT preprocessing
# ============================================================

TEMPORAL_POSITIONS = [0.10, 0.30, 0.50, 0.70, 0.90]
N_FRAMES = 16
SIZE = 224


# ============================================================
# Frame sampling
# Same logic as old notebook
# ============================================================

def get_clip_indices(start_frame, end_frame, position, n_frames=16):

    start_frame = int(start_frame)
    end_frame = int(end_frame)

    length = end_frame - start_frame + 1

    # If segment is shorter than 16 frames
    if length < n_frames:
        return np.linspace(
            start_frame,
            end_frame,
            n_frames
        ).round().astype(int)

    center = round(
        start_frame
        + position * (end_frame - start_frame)
    )

    clip_start = center - n_frames // 2
    clip_end = clip_start + n_frames - 1

    if clip_start < start_frame:
        clip_start = start_frame
        clip_end = clip_start + n_frames - 1

    if clip_end > end_frame:
        clip_end = end_frame
        clip_start = clip_end - n_frames + 1

    return np.arange(
        clip_start,
        clip_end + 1
    )


# ============================================================
# Load metadata
# ============================================================

train_val = pd.read_csv(TRAIN_VAL_META)
test = pd.read_csv(TEST_META)

# test metadata may already contain npz_path etc.
# We only need the useful columns.
needed_cols = [
    "split",
    "segment_id",
    "video_id",
    "uuid",
    "start_frame",
    "end_frame",
    "label",
]

train_val = train_val[needed_cols].copy()
test = test[needed_cols].copy()

segments = pd.concat(
    [train_val, test],
    ignore_index=True
)

print("Total segments:", len(segments))
print()
print(segments["split"].value_counts())
print()
print(pd.crosstab(segments["split"], segments["label"]))


# ============================================================
# Create GT NPZ files
# ============================================================

metadata_rows = []

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    for _, row in tqdm(
        segments.iterrows(),
        total=len(segments),
        desc="Creating GT 5x16 NPZ"
    ):

        split = row["split"]

        # Dataset folder is called "val",
        # while our metadata says "validation"
        zip_split = "val" if split == "validation" else split

        uuid = row["uuid"]

        all_clips = []
        all_indices = []

        for position in TEMPORAL_POSITIONS:

            frame_indices = get_clip_indices(
                row["start_frame"],
                row["end_frame"],
                position,
                N_FRAMES
            )

            clip_frames = []

            for frame_no in frame_indices:

                # IMPORTANT:
                # old pipeline used input/*.jpg
                # new experiment uses GT/*.png
                img_path = (
                    f"Vident-real/{zip_split}/"
                    f"{uuid}/GT/"
                    f"{int(frame_no):05d}.png"
                )

                try:
                    with z.open(img_path) as f:
                        img = Image.open(
                            io.BytesIO(f.read())
                        ).convert("RGB")

                except KeyError:
                    raise FileNotFoundError(
                        f"Frame not found inside ZIP:\n{img_path}"
                    )

                img = img.resize(
                    (SIZE, SIZE),
                    Image.Resampling.BILINEAR
                )

                arr = np.asarray(
                    img,
                    dtype=np.uint8
                )

                clip_frames.append(arr)

            clip = np.stack(
                clip_frames,
                axis=0
            )

            assert clip.shape == (
                N_FRAMES,
                SIZE,
                SIZE,
                3
            )

            all_clips.append(clip)
            all_indices.append(frame_indices)

        clips_array = np.stack(
            all_clips,
            axis=0
        )

        assert clips_array.shape == (
            5,
            16,
            224,
            224,
            3
        )

        filename = (
            f"{split}_"
            f"seg{int(row['segment_id']):03d}_"
            f"video{int(row['video_id']):03d}_"
            f"{row['label']}.npz"
        )

        out_path = OUTPUT_DIR / filename

        np.savez_compressed(
            out_path,
            clips=clips_array,
            label=row["label"],
            split=split,
            segment_id=int(row["segment_id"]),
            video_id=int(row["video_id"]),
            uuid=uuid,
            source="GT",
            temporal_positions=np.array(
                TEMPORAL_POSITIONS
            ),
            frame_indices=np.stack(
                all_indices
            )
        )

        metadata_rows.append({
            "split": split,
            "segment_id": int(row["segment_id"]),
            "video_id": int(row["video_id"]),
            "uuid": uuid,
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "label": row["label"],
            "source": "GT",
            "npz_path": str(out_path),
        })


# ============================================================
# Save new metadata
# ============================================================

new_metadata = pd.DataFrame(metadata_rows)

META_OUT = (
    OUTPUT_META_DIR
    / "multiclip_5x16_gt_metadata.csv"
)

new_metadata.to_csv(
    META_OUT,
    index=False
)

print()
print("DONE")
print("GT NPZ files:", len(list(OUTPUT_DIR.glob("*.npz"))))
print("Metadata:", META_OUT)