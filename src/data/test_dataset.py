# Source preservation: command-line execution resolves repository-relative inputs.
if __name__ == '__main__':
    import os as _preserve_os, sys as _preserve_sys
    from pathlib import Path as _PreservePath
    _preserve_root = _PreservePath(__file__).resolve().parents[1]
    _preserve_os.chdir(_preserve_root)
    _preserve_sys.path.insert(0, str(_preserve_root))
    _preserve_sys.path.insert(0, str(_preserve_root / 'src' / 'data'))

from dataset import DentalQuadrantDataset


dataset = DentalQuadrantDataset(
    metadata_csv="data/labels/quadrant_metadata.csv",
    split="train",
    frame_source="GT"
)


print("Number of train segments:", len(dataset))
print()


for i in [0, 23, 51]:

    sample = dataset[i]

    print("=" * 70)
    print("Dataset index:", i)
    print("Segment ID:", sample["segment_id"])
    print("Video ID:", sample["video_id"])
    print("UUID:", sample["uuid"])
    print("Label:", sample["label_name"])
    print("Clips shape:", tuple(sample["clips"].shape))

    print()
    print("Frame indices:")

    for clip_id, indices in enumerate(
        sample["frame_indices"],
        start=1
    ):
        print(
            f"Clip {clip_id}:",
            indices.tolist()
        )

    print()