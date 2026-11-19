"""Stable training primitives from the verified November 12 VideoSwin protocol."""
import contextlib
import hashlib
import os
from pathlib import Path
import tempfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import random
import torch
from torch import nn
from torchvision.models.video import swin3d_t
from torchvision.ops import StochasticDepth
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
INPUTS = Path(os.environ.get("VIDDENT_INPUTS", ROOT)).resolve()
RUNS = Path(os.environ.get("VIDDENT_RUNS", ROOT / "outputs/runs")).resolve()
FRAMES = INPUTS / "data/raw/Vident-real"
EVIDENCE = INPUTS / "outputs/runs"
OVERNIGHT = EVIDENCE / "overnight_frozen_partial_2026-11-12_202404"
NAMES = ["Upper", "Lower"]
MAPPING = {"Q1": 0, "Q2": 0, "Q3": 1, "Q4": 1}
BLOCK = "features.6.1."
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
WEIGHT_SHA = "7615ae035996b65eb38dad437ae533d2dfcd36f9f89d28c0f0fa7bfb8e6b3130"
S3D_SHA = "d76dad2f4b2a6701ad7f64e85b294b2d1989f1ea0d3f8b240446ed2152850bbb"


def weight_directory():
    return Path(os.environ.get("VIDDENT_WEIGHTS", Path(torch.hub.get_dir()) / "checkpoints")).resolve()


@contextlib.contextmanager
def gpu_lock():
    # The lock is shared by matrix runs and probes, even with different output roots.
    user = str(os.getuid()) if hasattr(os, "getuid") else os.environ.get("USERNAME", "local")
    path = Path(tempfile.gettempdir()) / f"viddent-matrix-{user}.lock"
    with path.open("a+b") as lock:
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another VidDent GPU run or probe holds the lock") from exc
        yield


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def prepare(clip, device):
    require(tuple(clip.shape) == (16,3,224,224), 'Incorrect clip tensor shape')
    require(torch.isfinite(clip).all().item() and clip.min() >= 0 and clip.max() <= 1, 'Invalid RGB tensor')
    x = clip.to(device).permute(1,0,2,3).unsqueeze(0)
    return (x - torch.tensor(MEAN, device=device).view(1,3,1,1,1)) / torch.tensor(STD, device=device).view(1,3,1,1,1)


def metrics(rows, losses):
    truth = [r["true_id"] for r in rows]
    guess = [r["predicted_id"] for r in rows]
    recall = recall_score(truth, guess, labels=[0, 1], average=None, zero_division=0)
    return {"loss": float(np.mean(losses)), "accuracy": float(accuracy_score(truth, guess)),
            "macro_f1": float(f1_score(truth, guess, labels=[0, 1], average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, guess)),
            "upper_recall": float(recall[0]), "lower_recall": float(recall[1]),
            "confusion_matrix": confusion_matrix(truth, guess, labels=[0, 1]).tolist(),
            "segments_or_occurrences": len(rows), "class_order": NAMES}


def forward_segment(model, sample, training=False, optimizer=None):
    require(tuple(sample["clips"].shape) == (5, 16, 3, 224, 224), "Input shape changed")
    target_id = MAPPING[sample["label_name"]]
    target = torch.tensor([target_id], device="cuda")
    probabilities = []
    loss_sum = 0.0
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for clip in sample["clips"]:
            logits = model(prepare(clip, "cuda"))
            loss = nn.functional.cross_entropy(logits, target) / 5
            require(bool(torch.isfinite(loss)), "Nonfinite loss")
            if training:
                loss.backward()
            loss_sum += loss.item()
            probabilities.append(logits.detach().softmax(1)[0].cpu())
    grads = {}
    if training:
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
                        "Missing/nonfinite gradient: " + name)
                group = "head" if name.startswith("head.") else "block"
                grads[group] = grads.get(group, 0.0) + parameter.grad.square().sum().item()
            else:
                require(parameter.grad is None, "Frozen parameter has gradient: " + name)
        require(all(v > 0 for v in grads.values()), "Zero gradient group")
        optimizer.step()
    probs = torch.stack(probabilities)
    mean = probs.mean(0)
    pred = int(mean.argmax())
    row = {k: sample[k] for k in ["segment_id", "video_id", "uuid", "label_name"]}
    row.update(true_id=target_id, predicted_id=pred, p_upper=float(mean[0]),
               p_lower=float(mean[1]), confidence=float(mean.max()), correct=pred == target_id)
    clips = [{"segment_id": sample["segment_id"], "video_id": sample["video_id"],
              "uuid": sample["uuid"], "clip": i + 1, "true_id": target_id,
              "p_upper": float(p[0]), "p_lower": float(p[1]), "predicted_id": int(p.argmax())}
             for i, p in enumerate(probs)]
    return row, clips, loss_sum, {k: v ** .5 for k, v in grads.items()}


def evaluate(model, dataset):
    model.eval()
    rows, clips, losses = [], [], []
    for i in range(len(dataset)):
        row, cr, loss, _ = forward_segment(model, dataset[i])
        rows.append(row); clips.extend(cr); losses.append(loss)
    return metrics(rows, losses), rows, clips


def make_model(weight_path, partial):
    seed_all(42)
    model = swin3d_t(weights=None)
    model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True), strict=True)
    model.requires_grad_(False)
    model.head = nn.Linear(768, 2)
    if partial:
        for p in model.features[6][-1].parameters():
            p.requires_grad = True
    model.eval()
    for module in model.modules():
        if isinstance(module, StochasticDepth):
            module.p = 0.0
    return model
