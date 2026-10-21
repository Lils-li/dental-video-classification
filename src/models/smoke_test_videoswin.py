# Source preservation: command-line execution resolves repository-relative inputs.
if __name__ == '__main__':
    import os as _preserve_os, sys as _preserve_sys
    from pathlib import Path as _PreservePath
    _preserve_root = _PreservePath(__file__).resolve().parents[1]
    _preserve_os.chdir(_preserve_root)
    _preserve_sys.path.insert(0, str(_preserve_root))
    _preserve_sys.path.insert(0, str(_preserve_root / 'src' / 'data'))

import torch
from torchvision.models.video import swin3d_t, Swin3D_T_Weights

from pathlib import Path
import sys

# чтобы импортировать dataset.py
sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))

from dataset import DentalQuadrantDataset


# ============================================================
# 1. Dataset
# ============================================================

dataset = DentalQuadrantDataset(
    metadata_csv="data/labels/quadrant_metadata.csv",
    split="train",
    frame_source="GT",
)

sample = dataset[0]

print("Label:", sample["label_name"])
print("All clips:", tuple(sample["clips"].shape))


# ============================================================
# 2. Take ONE of the 5 clips
# ============================================================

clip = sample["clips"][0]

print("Single clip before reshape:", tuple(clip.shape))

# dataset gives:
# [T, C, H, W]
#
# VideoSwin expects:
# [B, C, T, H, W]

clip = clip.permute(1, 0, 2, 3).unsqueeze(0)

print("VideoSwin input:", tuple(clip.shape))


# ============================================================
# 3. Pretrained VideoSwin Tiny
# ============================================================

weights = Swin3D_T_Weights.DEFAULT

model = swin3d_t(weights=weights)

model.eval()


# ============================================================
# 4. Normalize
# ============================================================

mean = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1, 1)

clip = (clip - mean) / std


# ============================================================
# 5. Forward pass
# ============================================================

with torch.no_grad():
    output = model(clip)

print()
print("Forward pass successful.")
print("Output shape:", tuple(output.shape))