"""Isolated binary VideoSwin comparison; never invokes legacy entry points."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torchvision.models.video import swin3d_t, Swin3D_T_Weights

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data"))
from dataset import DentalQuadrantDataset

NAMES = ["Upper", "Lower"]
MAPPING = {"Q1": 0, "Q2": 0, "Q3": 1, "Q4": 1}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def summarize(rows, losses):
    truth = [r["true_id"] for r in rows]
    pred = [r["predicted_id"] for r in rows]
    return {
        "segments": len(rows), "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=[0, 1], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, pred, labels=[0, 1], average="weighted", zero_division=0)),
        "precision": precision_score(truth, pred, labels=[0, 1], average=None, zero_division=0).tolist(),
        "recall": recall_score(truth, pred, labels=[0, 1], average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(truth, pred, labels=[0, 1]).tolist(),
        "class_order": NAMES, "matrix_rows": "true", "matrix_columns": "predicted",
    }


def save_predictions(run, prefix, rows, metrics):
    pd.DataFrame(rows).to_csv(run / f"{prefix}_predictions.csv", index=False)
    pd.DataFrame(metrics["confusion_matrix"], index=NAMES, columns=NAMES).rename_axis("true_label").to_csv(run / f"{prefix}_confusion_matrix.csv")
    write_json(run / f"{prefix}_metrics.json", metrics)


def execute(config, run, smoke):
    if sys.executable.lower() != str(ROOT / ".venv" / "Scripts" / "python.exe").lower():
        raise RuntimeError("Use the repository .venv interpreter explicitly.")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires the existing CUDA GPU; no silent CPU fallback.")
    torch.set_num_threads(config["cpu_threads"])
    seed_all(config["seed"])
    device = torch.device("cuda")
    weights = Swin3D_T_Weights.KINETICS400_V1
    cache = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    if not cache.is_file():
        raise RuntimeError("Pretrained weights are not cached. Refusing an implicit download.")
    weight_sha = sha(cache)
    if not weight_sha.startswith("7615ae03"):
        raise RuntimeError("Cached pretrained weight hash does not match the official filename.")
    train = DentalQuadrantDataset(ROOT / config["train_metadata"], data_root=ROOT / config["data_root"], split="train", frame_source="GT", num_clips=5, frames_per_clip=16, image_size=224)
    val = DentalQuadrantDataset(ROOT / config["val_metadata"], data_root=ROOT / config["data_root"], split="val", frame_source="GT", num_clips=5, frames_per_clip=16, image_size=224)
    assert len(train) == config["expected_train_segments"] and len(val) == 13
    assert not (set(train.df.uuid) & set(val.df.uuid))
    assert not train.df.duplicated(["split", "segment_id"]).any()
    train_labels = train.df.label.map(MAPPING).to_numpy()
    val_labels = val.df.label.map(MAPPING).to_numpy()
    assert not pd.isna(train_labels).any() and not pd.isna(val_labels).any()
    if "binary_label" in train.df:
        assert (train.df.binary_label == train.df.label.map({k: NAMES[v] for k, v in MAPPING.items()})).all()
    upper = np.flatnonzero(train_labels == 0)
    lower = np.flatnonzero(train_labels == 1)
    assert len(upper) and len(lower)
    # Check every planned TRAIN/VAL frame path, without opening any test frame.
    for dataset in (train, val):
        for row in dataset.df.itertuples():
            assert 1 <= row.start_frame <= row.end_frame <= row.total_frames
            for pos in dataset.temporal_positions:
                for frame in dataset._sample_clip_indices(row.start_frame, row.end_frame, pos):
                    path = dataset.data_root / dataset.split / row.uuid / "GT" / f"{frame:05d}.png"
                    if not path.is_file():
                        raise FileNotFoundError(path)
    provenance = {
        "python": sys.version, "interpreter": sys.executable,
        "gpu": torch.cuda.get_device_name(0), "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cuda": torch.version.cuda, "weights_url": weights.url, "pretrained_sha256": weight_sha,
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "torchvision", "numpy", "pandas", "Pillow", "scikit-learn"]},
        "input_sha256": {p: sha(ROOT / p) for p in [config["train_metadata"], config["val_metadata"], "src/data/dataset.py", "src/run_binary_comparison.py", "src/train_videoswin_binary_clean.py", "src/analyze_binary_predictions.py", "outputs/data_quality/manual_segment_audit_colored.xlsx"]},
        "train_counts": {"Upper": int(len(upper)), "Lower": int(len(lower))},
        "val_counts": {name: int((val_labels == i).sum()) for i, name in enumerate(NAMES)},
        "training_updates_per_full_epoch": int(2 * len(upper)),
        "smoke_only": smoke, "test_used": False,
    }
    write_json(run / "config.json", config | {"smoke_only": smoke, "executed_epochs": 1 if smoke else config["epochs"]})
    train.df.to_csv(run / "train_manifest.csv", index=False)
    val.df.to_csv(run / "validation_manifest.csv", index=False)
    start = time.perf_counter()
    model = swin3d_t(weights=None)
    model.load_state_dict(torch.load(cache, map_location="cpu", weights_only=True))
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.head = nn.Linear(model.head.in_features, 2)
    provenance["initial_head_sha256"] = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.head.parameters())).hexdigest()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    mean = torch.tensor(config["mean"], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(config["std"], device=device).view(1, 3, 1, 1, 1)
    torch.cuda.synchronize()
    provenance["model_setup_seconds"] = time.perf_counter() - start
    write_json(run / "provenance.json", provenance)
    timing = {"train_segment_seconds": [], "val_segment_seconds": []}
    best = -1.0
    epoch_metrics = []
    # Legacy policy retained: frozen backbone remains in train mode, including stochastic depth.
    for epoch in range(1, (1 if smoke else config["epochs"]) + 1):
        rng = np.random.default_rng(config["seed"] + epoch)
        indices = np.concatenate([upper, rng.choice(lower, size=len(upper), replace=True)])
        rng.shuffle(indices)
        if smoke:
            indices = np.array([upper[0], lower[0]])
        pd.DataFrame({"dataset_index": indices, "segment_id": train.df.iloc[indices].segment_id.to_numpy(), "binary_id": train_labels[indices]}).to_csv(run / f"epoch_{epoch:02d}_sampling.csv", index=False)
        outputs = {}
        epoch_start = time.perf_counter()
        for phase, dataset, order in [("train", train, indices), ("val", val, [int(np.flatnonzero(val_labels == 0)[0]), int(np.flatnonzero(val_labels == 1)[0])] if smoke else range(len(val)))]:
            model.train(phase == "train")
            rows, losses = [], []
            for index in order:
                torch.cuda.synchronize()
                sample_start = time.perf_counter()
                sample = dataset[int(index)]
                assert sample["clips"].shape == (5, 16, 3, 224, 224)
                assert torch.isfinite(sample["clips"]).all()
                target_id = MAPPING[sample["label_name"]]
                target = torch.tensor([target_id], dtype=torch.long, device=device)
                if phase == "train":
                    optimizer.zero_grad(set_to_none=True)
                    head_before = model.head.weight.detach().clone() if smoke else None
                probabilities, loss_sum = [], 0.0
                with torch.set_grad_enabled(phase == "train"):
                    for clip in sample["clips"]:
                        x = (clip.to(device).permute(1, 0, 2, 3).unsqueeze(0) - mean) / std
                        logits = model(x)
                        loss = criterion(logits, target) / 5
                        assert torch.isfinite(loss) and torch.isfinite(logits).all()
                        if phase == "train":
                            loss.backward()
                        loss_sum += loss.item()
                        probabilities.append(torch.softmax(logits.detach(), dim=1).cpu())
                if phase == "train":
                    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith("head."))
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters())
                    optimizer.step()
                    if smoke:
                        assert not torch.equal(head_before, model.head.weight.detach())
                p = torch.cat(probabilities).mean(0)
                pred = int(p.argmax())
                assert abs(float(p.sum()) - 1) < 1e-5
                rows.append({"segment_id": sample["segment_id"], "original_label": sample["label_name"], "true_id": target_id, "predicted_id": pred, "true_binary": NAMES[target_id], "predicted_binary": NAMES[pred], "p_upper": float(p[0]), "p_lower": float(p[1]), "correct": pred == target_id})
                losses.append(loss_sum)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - sample_start
                timing[f"{phase}_segment_seconds"].append(elapsed)
                print(f"epoch={epoch} phase={phase} segment={sample['segment_id']} seconds={elapsed:.3f}", flush=True)
            metrics = summarize(rows, losses)
            save_predictions(run, f"epoch_{epoch:02d}_{phase}", rows, metrics)
            outputs[phase] = metrics
            if phase == "val" and metrics["macro_f1"] > best:
                best = metrics["macro_f1"]
                # Unique epoch checkpoint names; no previous checkpoint is replaced.
                checkpoint_path = run / f"checkpoint_best_epoch_{epoch:02d}.pt"
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "class_names": NAMES, "val_macro_f1": best, "config": config, "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all(), "numpy_rng_state": np.random.get_state()[1].tolist(), "python_rng_state": random.getstate()}, checkpoint_path)
                write_json(run / "best_checkpoint.json", {"epoch": epoch, "path": checkpoint_path.name, "val_macro_f1": best, "smoke_only": smoke})
                save_predictions(run, f"best_epoch_{epoch:02d}_val", rows, metrics)
        epoch_metrics.append({"epoch": epoch, "train": outputs["train"], "val": outputs["val"], "seconds": time.perf_counter() - epoch_start})
        write_json(run / "metrics.json", epoch_metrics)
        pd.DataFrame([{ "epoch": m["epoch"], "seconds": m["seconds"], **{f"{phase}_{key}": m[phase][key] for phase in ("train", "val") for key in ("loss", "accuracy", "balanced_accuracy", "macro_f1")} } for m in epoch_metrics]).to_csv(run / "metrics.csv", index=False)
    timing["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
    timing["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
    write_json(run / "timing.json", timing)
    write_json(run / "completed.json", {"smoke_only": smoke, "epochs": len(epoch_metrics), "test_used": False})
    print(json.dumps({"run": str(run), "smoke_only": smoke, "timing": timing}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--approve-full-training", action="store_true")
    args = parser.parse_args()
    if not args.smoke and not args.approve_full_training:
        parser.error("Full training requires --approve-full-training after runtime approval.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run = args.run_dir.resolve()
    allowed_root = (ROOT / "outputs" / "runs").resolve()
    if not run.is_relative_to(allowed_root):
        raise RuntimeError("Run directory must be inside outputs/runs.")
    run.mkdir(parents=True, exist_ok=False)
    try:
        execute(config, run, args.smoke)
    except Exception as error:
        write_json(run / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
