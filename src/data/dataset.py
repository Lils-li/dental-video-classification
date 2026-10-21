from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


CLASS_TO_ID = {
    "Q1": 0,
    "Q2": 1,
    "Q3": 2,
    "Q4": 3,
}


class DentalQuadrantDataset(Dataset):

    def __init__(
        self,
        metadata_csv,
        data_root="data/raw/Vident-real",
        split="train",
        frame_source="GT",
        num_clips=5,
        frames_per_clip=16,
        image_size=224,
    ):
        self.df = pd.read_csv(metadata_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.data_root = Path(data_root)
        self.split = split
        self.frame_source = frame_source

        self.num_clips = num_clips
        self.frames_per_clip = frames_per_clip
        self.image_size = image_size

        self.temporal_positions = np.linspace(
            0.1,
            0.9,
            num_clips
        )

    def __len__(self):
        return len(self.df)

    def _sample_clip_indices(
        self,
        start_frame,
        end_frame,
        position
    ):
        """
        One 16-frame clip around a temporal position
        such as 10%, 30%, 50%, 70%, 90%.

        If segment is shorter than 16 frames,
        indices are repeated.
        """

        segment_length = end_frame - start_frame + 1

        center = start_frame + int(
            round(position * (segment_length - 1))
        )

        half = self.frames_per_clip // 2

        clip_start = center - half

        max_start = end_frame - self.frames_per_clip + 1

        if segment_length >= self.frames_per_clip:

            clip_start = max(
                start_frame,
                min(clip_start, max_start)
            )

            indices = list(
                range(
                    clip_start,
                    clip_start + self.frames_per_clip
                )
            )

        else:
            # Short segment:
            # sample 16 positions across available range,
            # repeating frames when necessary.
            indices = np.linspace(
                start_frame,
                end_frame,
                self.frames_per_clip
            )

            indices = np.rint(indices).astype(int).tolist()

        return indices

    def _load_frame(self, folder, frame_number):

        if self.frame_source == "GT":
            path = folder / "GT" / f"{frame_number:05d}.png"

        elif self.frame_source == "input":
            path = folder / "input" / f"{frame_number:05d}.jpg"

        else:
            raise ValueError(
                f"Unknown frame_source: {self.frame_source}"
            )

        if not path.exists():
            raise FileNotFoundError(path)

        image = Image.open(path).convert("RGB")

        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        array = np.asarray(image, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(array).permute(2, 0, 1)

        return tensor

    def __getitem__(self, index):

        row = self.df.iloc[index]

        uuid = row["uuid"]

        folder = (
            self.data_root
            / self.split
            / uuid
        )

        all_indices = []
        clips = []

        for position in self.temporal_positions:

            frame_indices = self._sample_clip_indices(
                int(row["start_frame"]),
                int(row["end_frame"]),
                position
            )

            all_indices.append(frame_indices)

            frames = [
                self._load_frame(
                    folder,
                    frame_number
                )
                for frame_number in frame_indices
            ]

            clip = torch.stack(frames)

            clips.append(clip)

        clips = torch.stack(clips)

        label = CLASS_TO_ID[row["label"]]

        return {
            "clips": clips,
            "label": torch.tensor(label),
            "label_name": row["label"],
            "segment_id": int(row["segment_id"]),
            "video_id": int(row["video_id"]),
            "uuid": uuid,
            "frame_indices": torch.tensor(all_indices),
        }