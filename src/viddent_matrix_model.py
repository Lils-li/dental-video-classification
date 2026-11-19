import argparse
import gc
import json
import time

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.models.video import S3D_Weights, s3d

from viddent_protocol import (
    FRAMES, OVERNIGHT, ROOT, S3D_SHA, WEIGHT_SHA, forward_segment, gpu_lock, make_model,
    seed_all, sha, weight_directory,
)
from data.dataset import DentalQuadrantDataset

GIB = 1024**3


class S3DClipAverage(nn.Module):
    def forward(self, x):
        if x.shape[-3:] != (2, 7, 7):
            raise RuntimeError(f"Unexpected S3D feature dimensions: {x.shape[-3:]}")
        return x.mean(dim=(-3, -2, -1), keepdim=True)


class CheckpointedBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        if torch.is_grad_enabled():
            return checkpoint(self.block, x, use_reentrant=False)
        return self.block(x)


def build_model(architecture, strategy, recompute=False):
    torch.set_num_threads(4)
    seed_all(42)
    if architecture == "swin":
        weights = weight_directory() / "swin3d_t-7615ae03.pth"
        if sha(weights) != WEIGHT_SHA:
            raise RuntimeError("VideoSwin weight checksum mismatch")
        model = make_model(weights, strategy == "partial")
        head = model.head
    else:
        weights = S3D_Weights.KINETICS400_V1
        weight_path = weight_directory() / weights.url.rsplit("/", 1)[-1]
        if not weight_path.is_file() or sha(weight_path) != S3D_SHA:
            raise RuntimeError("S3D Kinetics-400 weight cache is missing or invalid")
        model = s3d(weights=None)
        model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
        model.avgpool = S3DClipAverage()
        model.requires_grad_(False)
        head = nn.Conv3d(1024, 2, kernel_size=1)
        model.classifier[1] = head
        if strategy == "partial":
            model.features[-1].requires_grad_(True)
    if strategy == "full":
        model.requires_grad_(True)
    if recompute:
        for index, stage in enumerate(model.features):
            if isinstance(stage, nn.Sequential):
                for block_index, block in enumerate(stage):
                    stage[block_index] = CheckpointedBlock(block)
            else:
                model.features[index] = CheckpointedBlock(stage)
    model.eval().cuda()
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": list(head.parameters()), "lr": .001}]
    if backbone:
        groups.append({"params": backbone, "lr": .00001})
    optimizer = torch.optim.AdamW(groups, weight_decay=.0001, foreach=False)
    return model, optimizer


def train_segment(model, sample, optimizer, architecture, strategy):
    # CUDA max-pool backward in the early S3D backbone can be nondeterministic.
    if (architecture, strategy) != ("s3d", "full"):
        return forward_segment(model, sample, True, optimizer)
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        return forward_segment(model, sample, True, optimizer)
    finally:
        torch.use_deterministic_algorithms(True)


def check_resources(directory):
    import shutil
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    free, total = torch.cuda.mem_get_info()
    disk = shutil.disk_usage(directory).free
    if free < 2 * GIB or disk < 2 * GIB:
        raise RuntimeError(f"Insufficient resources: {free/GIB:.2f} GiB VRAM, {disk/GIB:.2f} GiB disk free")
    return {"free_vram": free, "total_vram": total, "disk_free": disk,
            "gpu": torch.cuda.get_device_name()}


def benchmark(architecture, strategy, updates):
    from viddent_matrix import validation_tables

    original, _, _ = validation_tables()
    base = check_resources(ROOT)
    train = DentalQuadrantDataset(OVERNIGHT / "train_manifest.csv",
                                 data_root=FRAMES, split="train")
    if not train.df[original.columns].equals(original):
        raise RuntimeError("Benchmark training manifest differs from current Original training rows")
    val = DentalQuadrantDataset(OVERNIGHT / "validation_manifest.csv",
                               data_root=FRAMES, split="val")
    attempts = [False, True] if architecture == "swin" and strategy == "full" else [False]
    for recompute in attempts:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model, optimizer = build_model(architecture, strategy, recompute)
            named = dict(model.named_parameters())
            before = {name: p.detach().cpu().clone() for name, p in named.items()}
            durations = []
            for i in range(updates):
                start = time.perf_counter()
                _, _, loss, _ = train_segment(model, train[i], optimizer, architecture, strategy)
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - start)
                print(f"Update {i+1}/{updates}: {durations[-1]:.2f} s", flush=True)
            changed = {name: not torch.equal(p.detach().cpu(), before[name])
                       for name, p in named.items()}
            backbone_changed = sum(changed[name] for name in named
                                   if not name.startswith(("head.", "classifier.")))
            if sum(changed.values()) == backbone_changed:
                raise RuntimeError("Classifier parameters did not update")
            gradient_norms = {
                "head": sum(p.grad.square().sum().item() for name, p in named.items()
                            if name.startswith(("head.", "classifier."))) ** .5,
                "backbone": sum(p.grad.square().sum().item() for name, p in named.items()
                                if not name.startswith(("head.", "classifier."))
                                and p.grad is not None) ** .5,
            }
            if (strategy != "linear" and not backbone_changed) or any(
                    changed[name] and not p.requires_grad for name, p in named.items()):
                raise RuntimeError("Parameter update verification failed")
            val_start = time.perf_counter()
            forward_segment(model, val[0])
            torch.cuda.synchronize()
            val_seconds = time.perf_counter() - val_start
            return {**base, "architecture": architecture, "strategy": strategy,
                    "recompute": recompute, "seconds_per_update": durations,
                    "validation_segment_seconds": val_seconds,
                    "allocated_peak": torch.cuda.max_memory_allocated(),
                    "reserved_peak": torch.cuda.max_memory_reserved(),
                    "parameters": sum(p.numel() for p in named.values()),
                    "trainable": sum(p.numel() for p in named.values() if p.requires_grad),
                    "gradient_norms": gradient_norms, "loss": loss,
                    "changed_tensors": sum(changed.values()), "backbone_changed_tensors": backbone_changed}
        except torch.cuda.OutOfMemoryError:
            print(json.dumps({"status": "OOM", "recompute": recompute,
                              "allocated_peak": torch.cuda.max_memory_allocated(),
                              "reserved_peak": torch.cuda.max_memory_reserved()}), flush=True)
            if recompute == attempts[-1]:
                raise
        finally:
            model = optimizer = named = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture", choices=["swin", "s3d"])
    parser.add_argument("strategy", choices=["linear", "partial", "full"])
    parser.add_argument("--updates", type=int, default=3, choices=range(1, 6))
    args = parser.parse_args()
    with gpu_lock():
        print(json.dumps(benchmark(args.architecture, args.strategy, args.updates), indent=2))
