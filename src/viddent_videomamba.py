"""Optional VideoMamba-Ti adapter and short, isolated CUDA feasibility probe."""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn

from data.dataset import DentalQuadrantDataset
from viddent_protocol import FRAMES, OVERNIGHT, forward_segment, gpu_lock, seed_all, sha

CHECKPOINT = "videomamba_t16_k400_f16_res224.pth"
CHECKPOINT_SHA256 = "a335d728ae4dbe4f49a435022f95c6cf98108d20fe084120db1f18cb73e84f4a"
STRATEGIES = ("linear", "partial", "full")


def preprocess_frame(image):
    image = image.convert("RGB")
    short = min(image.size)
    size = tuple(int(side * 224 / short) for side in image.size)
    image = image.resize(size, Image.Resampling.BILINEAR)
    left, top = (round((side - 224) / 2) for side in size)
    image = image.crop((left, top, left + 224, top + 224))
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1)


class VideoMambaDataset(DentalQuadrantDataset):
    def __init__(self, metadata, *, data_root, split):
        if split not in ("train", "val"):
            raise ValueError("VideoMamba supports only VidDent train and validation")
        super().__init__(metadata, data_root=data_root, split=split)

    def _load_frame(self, folder, frame_number):
        with Image.open(folder / "GT" / f"{frame_number:05d}.png") as image:
            return preprocess_frame(image)


def configure_binary_head(model, strategy):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    model.requires_grad_(strategy == "full")
    model.head = nn.Linear(model.head.in_features, 2)
    model.num_classes = 2
    if strategy == "partial":
        model.layers[-1].requires_grad_(True)
    return model.eval()


def build_model(source, checkpoint_path, strategy):
    if sha(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("Official VideoMamba checkpoint SHA-256 mismatch")
    path = source / "videomamba/video_sm/models/videomamba.py"
    spec = importlib.util.spec_from_file_location("viddent_official_videomamba", path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    model = upstream.videomamba_tiny(pretrained=False, num_frames=16, img_size=224,
                                    kernel_size=1, num_classes=400)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key in ("model", "module", "state_dict"):
        if key in state:
            state = state[key]
            break
    model.load_state_dict(state, strict=True)
    return configure_binary_head(model, strategy)


def probe_samples(real):
    if not real:
        generator = torch.Generator().manual_seed(42)
        clips = torch.rand((5, 16, 3, 224, 224), generator=generator)
        sample = {"segment_id": 0, "video_id": 0, "uuid": "synthetic", "label_name": "Q1", "clips": clips}
        return [sample, {**sample, "segment_id": 1, "label_name": "Q3"}], sample
    from viddent_matrix import FIELDS, fixed_subsets, validation_tables

    original, _, val = validation_tables()
    fixed_subsets(val)
    train = VideoMambaDataset(OVERNIGHT / "train_manifest.csv", data_root=FRAMES, split="train")
    validation = VideoMambaDataset(OVERNIGHT / "validation_manifest.csv", data_root=FRAMES, split="val")
    if not train.df[FIELDS].equals(original[FIELDS]):
        raise ValueError("Historical train manifest differs from current Original training rows")
    samples = [train[int(train.df.index[train.df.label.isin(labels)][0])]
               for labels in (("Q1", "Q2"), ("Q3", "Q4"))]
    return samples, validation[0]


def probe(source, checkpoint_path, strategy, updates, real):
    if sys.platform != "linux" or not torch.cuda.is_available():
        raise RuntimeError("This probe requires Linux with a working NVIDIA CUDA environment")
    seed_all(42)
    train, val = probe_samples(real)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(source, checkpoint_path, strategy).cuda()
    params = dict(model.named_parameters())
    trainable = {name for name, p in params.items() if p.requires_grad}
    head = [p for name, p in params.items() if name.startswith("head.")]
    backbone = [p for name, p in params.items() if p.requires_grad and not name.startswith("head.")]
    groups = [{"params": head, "lr": .001}]
    if backbone:
        groups.append({"params": backbone, "lr": .00001})
    optimizer = torch.optim.AdamW(groups, weight_decay=.0001, foreach=False)

    before = {name: p.detach().cpu().clone() for name, p in params.items()}
    buffers = {name: b.detach().cpu().clone() for name, b in model.named_buffers()}
    duration = []
    for step in range(updates):
        start = time.perf_counter()
        _, _, loss, grad_norms = forward_segment(model, train[step % len(train)], True, optimizer)
        torch.cuda.synchronize()
        duration.append(time.perf_counter() - start)

    changed = {name for name, p in params.items() if not torch.equal(before[name], p.detach().cpu())}
    if any(not params[name].requires_grad for name in changed):
        raise RuntimeError("Frozen parameter changed")
    if any(not torch.equal(buffers[name], b.detach().cpu()) for name, b in model.named_buffers()):
        raise RuntimeError("Model buffer changed")
    head_changes = changed & {name for name in params if name.startswith("head.")}
    backbone_changes = changed - head_changes
    if not head_changes or bool(backbone_changes) != (strategy != "linear"):
        raise RuntimeError("Expected parameter group failed update verification")
    if strategy == "full" and not any(name.startswith("patch_embed.") for name in backbone_changes):
        raise RuntimeError("Early full-backbone parameters failed update verification")

    start = time.perf_counter()
    row, clips, _, _ = forward_segment(model, val)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start
    allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    checkpoint = io.BytesIO()
    torch.save({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}, checkpoint)
    checkpoint.seek(0)
    del model, optimizer, params, head, backbone, groups
    torch.cuda.empty_cache()
    reloaded = build_model(source, checkpoint_path, strategy)
    reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    reloaded.cuda()
    row_again, clips_again, _, _ = forward_segment(reloaded, val)
    if row["predicted_id"] != row_again["predicted_id"]:
        raise RuntimeError("Reload prediction differs")
    for first, second in zip(clips, clips_again):
        np.testing.assert_allclose([first["p_upper"], first["p_lower"]],
                                   [second["p_upper"], second["p_lower"]], rtol=1e-5, atol=1e-6)
    return {"status": "PASS", "strategy": strategy, "inputs": "VidDent train and validation" if real else "synthetic",
            "checkpoint_sha256": CHECKPOINT_SHA256, "gpu": torch.cuda.get_device_name(),
            "compute_capability": torch.cuda.get_device_capability(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "free_vram_bytes": free, "total_vram_bytes": total,
            "trainable_parameters": sum(t.numel() for name, t in before.items() if name in trainable),
            "gradient_norms": grad_norms, "head_tensors_changed": len(head_changes),
            "backbone_tensors_changed": len(backbone_changes), "frozen_parameters_unchanged": True,
            "reload_consistent": True, "loss": loss, "seconds_per_update": duration,
            "seconds_per_validation_segment": inference_seconds, "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", choices=STRATEGIES)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--updates", type=int, choices=range(1, 6), default=2)
    parser.add_argument("--viddent", action="store_true", help="Use verified local train/val frames instead of synthetic clips")
    args = parser.parse_args()
    with gpu_lock():
        print(json.dumps(probe(args.source, args.checkpoint, args.strategy, args.updates, args.viddent), indent=2))
