"""Fixed validation diagnostics and resumable sequential VidDent training."""
import argparse
import contextlib
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import pandas as pd
import torch
import torchvision

from data.dataset import DentalQuadrantDataset
from viddent_protocol import EVIDENCE, FRAMES, INPUTS, MAPPING, OVERNIGHT, ROOT, RUNS, Tee, evaluate, gpu_lock, metrics, sha
from viddent_matrix_model import build_model, check_resources, train_segment

CACHED = EVIDENCE / "cached_equal_updates_2026-11-04_01" / "cleaned_balanced_seed42"
REFERENCE = OVERNIGHT / "frozen/best_val_reloaded_predictions.csv"
REFERENCE_SHA = "0afe926bd03bc1aa5675e6985c822860ec0595f264b9a5542542ab0445cfb500"
KEY = ["segment_id", "video_id", "uuid"]
FIELDS = ["split", *KEY, "label", "start_frame", "end_frame", "total_frames"]
MANIFEST = INPUTS / "data/labels/quadrant_metadata.csv"
CLEANED = INPUTS / "data/labels/binary_clean_metadata.csv"
COMPLETED = {"original-swin-linear", "original-swin-partial", "cleaned-swin-linear"}
CASES = [f"{data}-{arch}-{strategy}" for data in ("original", "cleaned")
         for arch in ("swin", "s3d") for strategy in ("linear", "partial", "full")
         if f"{data}-{arch}-{strategy}" not in COMPLETED]


def validation_tables():
    with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        source = pd.DataFrame(row for row in csv.DictReader(stream)
                              if row["split"] in {"train", "val"})
    for column in ["segment_id", "video_id", "start_frame", "end_frame", "total_frames"]:
        source[column] = source[column].astype(int)
    original = source[source.split.eq("train")].reset_index(drop=True)
    val = source[source.split.eq("val")].reset_index(drop=True)
    cleaned = pd.read_csv(CLEANED)
    for table, count in [(original, {0: 63, 1: 18}), (cleaned, {0: 46, 1: 7}), (val, {0: 4, 1: 9})]:
        if (table.label.map(MAPPING).value_counts(dropna=False).to_dict() != count
                or table.segment_id.duplicated().any()
                or table.groupby("video_id").uuid.nunique().max() != 1):
            raise RuntimeError("Unexpected class counts or segment/video identifiers")
    for table in (original, cleaned):
        if set(table.uuid) & set(val.uuid):
            raise RuntimeError("Train/validation UUID overlap")
    if (len(cleaned.merge(original, on=FIELDS)) != len(cleaned)
            or set(cleaned.identifiability) != {"Clear", "Partial"}
            or not val[FIELDS].equals(pd.read_csv(OVERNIGHT / "validation_manifest.csv")[FIELDS])):
        raise RuntimeError("Current rows disagree with the expected manifests")
    return original, cleaned, val


def join_predictions(val, preds):
    preds = preds.rename(columns={"p_Upper": "p_upper", "p_Lower": "p_lower"}).copy()
    if len(preds) != len(val) or preds.duplicated(KEY).any():
        raise ValueError("Expected one prediction per fixed validation segment")
    rows = val.merge(preds[KEY + ["true_id", "predicted_id", "p_upper", "p_lower"]],
                     on=KEY, validate="one_to_one")
    probabilities = rows[["p_upper", "p_lower"]].to_numpy()
    if (len(rows) != len(val) or not rows.true_id.eq(rows.label.map(MAPPING)).all()
            or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1)
            or not np.array_equal(rows.predicted_id, probabilities.argmax(axis=1))):
        raise ValueError("Validation identities, labels, or probabilities disagree")
    rows["confidence"] = probabilities.max(axis=1)
    rows["correct"] = rows.true_id.eq(rows.predicted_id)
    return rows


def fixed_subsets(val):
    if sha(REFERENCE) != REFERENCE_SHA:
        raise ValueError("Original VideoSwin diagnostic prediction checksum changed")
    reference = join_predictions(val, pd.read_csv(REFERENCE))
    hard = reference.loc[~reference.correct]
    if sorted(hard.segment_id) != [1, 2, 3, 4, 5] or not hard.true_id.eq(1).all():
        raise ValueError("Pinned Original VideoSwin reference disagrees with recorded diagnostic membership")
    return reference[KEY].assign(subset=np.where(reference.correct, "easy", "hard"))


def atomic_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def atomic_checkpoint(path, state):
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def save_diagnostics(out, val, preds):
    rows = join_predictions(val, preds).merge(fixed_subsets(val), on=KEY, validate="one_to_one")
    result = metrics(rows.to_dict("records"), [0.0])
    del result["loss"]
    for label, group in rows.groupby("subset"):
        result[label] = {"correct": int(group.correct.sum()), "total": len(group),
                         "accuracy": float(group.correct.mean())}
    rows.to_csv(out / "segment_predictions.csv", index=False)
    pd.DataFrame(result["confusion_matrix"], index=["Upper", "Lower"],
                 columns=["Upper", "Lower"]).to_csv(out / "confusion_matrix.csv", index_label="true_label")
    atomic_json(out / "metrics.json", result)


def new_output(prefix):
    out = RUNS / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    RUNS.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    return out


def audit(out, val, extra):
    memberships = val.merge(fixed_subsets(val), on=KEY, validate="one_to_one")
    for name, rows in memberships.groupby("subset"):
        rows.to_csv(out / f"{name}_manifest.csv", index=False)
    atomic_json(out / "reference.json", {"predictions": REFERENCE.relative_to(INPUTS).as_posix(),
                                        "sha256": REFERENCE_SHA,
                                        "checkpoint": "frozen/checkpoint_best_epoch_01.pt",
                                        "checkpoint_sha256": "fc7e240d7611ae448fc0195077b46358940dbf4551ac1960ed074fd4e090a65d"})
    best = json.loads((CACHED / "best_checkpoint.json").read_text())
    inputs = {"original-swin-linear": REFERENCE,
              "original-swin-partial": OVERNIGHT / "partial/best_val_reloaded_predictions.csv",
              "cleaned-swin-linear": CACHED / f"epoch_{best['epoch']:02d}_val_predictions.csv"}
    cached_original = CACHED.with_name("original_balanced_seed42")
    epoch = json.loads((cached_original / "best_checkpoint.json").read_text())["epoch"]
    inputs["original-swin-linear-matched-cache"] = cached_original / f"epoch_{epoch:02d}_val_predictions.csv"
    if extra:
        inputs["external"] = extra
    for case, path in inputs.items():
        target = out / case
        target.mkdir()
        save_diagnostics(target, val, pd.read_csv(path))
        atomic_json(target / "source.json", {"path": str(path), "sha256": sha(path)})


def run_case(out, case, tables, swin_recompute):
    original, cleaned, val = tables
    dataset, architecture, strategy = case.split("-")
    train = original if dataset == "original" else cleaned
    out.mkdir(exist_ok=True)
    recompute = architecture == "swin" and strategy == "full" and swin_recompute
    if (out / "completed.json").exists():
        print(f"Skipping completed {case}", flush=True)
        return
    config = {"case": case, "epochs": 15, "seed": 42, "recompute": recompute,
              "updates_per_epoch": 2 * int(train.label.map(MAPPING).eq(0).sum()),
              "manifests": [table[FIELDS].to_csv(index=False) for table in (train, val)], "reference": sha(REFERENCE),
              "environment": {
                  "os": platform.system(), "gpu": torch.cuda.get_device_name(),
                  "pytorch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                  "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              },
              "determinism": "warn_only_during_update" if case.endswith("s3d-full") else "strict",
              "code": {path.name: sha(path) for path in [
                  Path(__file__), ROOT / "src/viddent_matrix_model.py",
                  ROOT / "src/viddent_protocol.py", ROOT / "src/data/dataset.py"]}}
    config_path = out / "protocol.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Run protocol differs from saved protocol")
    for marker in RUNS.glob(f"viddent_matrix_*/{case}/completed.json"):
        prior = marker.parent
        if (prior / "protocol.json").is_file() and json.loads((prior / "protocol.json").read_text()) == config:
            print(f"Skipping prior completed {case} at {prior}", flush=True)
            return
    atomic_json(config_path, config)
    atomic_json(out / "resources.json", check_resources(out))
    train.to_csv(out / "train_manifest.csv", index=False)
    val.to_csv(out / "validation_manifest.csv", index=False)
    train_data = DentalQuadrantDataset(out / "train_manifest.csv", data_root=FRAMES, split="train")
    val_data = DentalQuadrantDataset(out / "validation_manifest.csv", data_root=FRAMES, split="val")
    labels = train.label.map(MAPPING).to_numpy()
    upper, lower = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    model, optimizer = build_model(architecture, strategy, recompute)
    latest = out / "latest.pt"
    first_epoch, best_epoch, best_score = 1, 0, -math.inf
    if latest.exists():
        state = torch.load(latest, weights_only=True, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        first_epoch = state["epoch"] + 1
        best_epoch, best_score = state["best_epoch"], state["best_score"]
        del state
    for epoch in range(first_epoch, 16):
        if shutil.disk_usage(out).free < 2 * 1024**3:
            raise RuntimeError("Less than 2 GiB free disk before epoch")
        rng = np.random.default_rng(42 + epoch)
        order = np.concatenate([upper, rng.choice(lower, len(upper), replace=True)])
        rng.shuffle(order)
        pd.DataFrame({"segment_id": train.iloc[order].segment_id}).to_csv(out / f"epoch_{epoch:02d}_sampling.csv", index=False)
        torch.cuda.reset_peak_memory_stats()
        train_rows, losses = [], []
        for step, index in enumerate(order, 1):
            row, _, loss, _ = train_segment(model, train_data[int(index)], optimizer, architecture, strategy)
            train_rows.append(row)
            losses.append(loss)
            if step == 1 or step % 10 == 0:
                print(f"{case} epoch {epoch}/15 update {step}/{len(order)} loss {loss:.4f}", flush=True)
        result, rows, _ = evaluate(model, val_data)
        score = result["macro_f1"]
        if score > best_score:
            best_epoch, best_score = epoch, score
            atomic_checkpoint(out / f"best_epoch_{epoch:02d}.pt", model.state_dict())
        pd.DataFrame(rows).to_csv(out / f"epoch_{epoch:02d}_predictions.csv", index=False)
        atomic_json(out / f"epoch_{epoch:02d}_metrics.json",
                    {"validation": result, "training": metrics(train_rows, losses),
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                     "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        atomic_checkpoint(latest, {"epoch": epoch, "best_epoch": best_epoch, "best_score": best_score,
                                   "model": model.state_dict(), "optimizer": optimizer.state_dict()})
        for path in out.glob("best_epoch_*.pt"):
            if path.name != f"best_epoch_{best_epoch:02d}.pt":
                path.unlink()
        print(f"{case} epoch {epoch} validation macro-F1 {score:.4f}", flush=True)
    for label, epoch in [("best", best_epoch), ("final", 15)]:
        target = out / label
        target.mkdir(exist_ok=True)
        save_diagnostics(target, val, pd.read_csv(out / f"epoch_{epoch:02d}_predictions.csv"))
    atomic_json(out / "completed.json", {"case": case, "best_epoch": best_epoch, "epochs": 15})


def main():
    p = argparse.ArgumentParser()
    modes = p.add_subparsers(dest="mode", required=True)
    audit_parser = modes.add_parser("audit")
    audit_parser.add_argument("--predictions", type=Path, help="Additional validation prediction CSV")
    train_parser = modes.add_parser("train")
    choice = train_parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--cases", nargs="+", choices=CASES)
    choice.add_argument("--resume", type=Path)
    train_parser.add_argument("--swin-recompute", action="store_true")
    args = p.parse_args()
    tables = validation_tables()
    if args.mode == "audit":
        out = new_output("viddent_diagnostics")
        audit(out, tables[-1], args.predictions)
        print(out)
        return
    fixed_subsets(tables[-1])
    with gpu_lock():
        if args.resume:
            out = args.resume.resolve()
            if out.parent != RUNS or not out.name.startswith("viddent_matrix_"):
                p.error("Resume requires a VidDent matrix run directory")
            plan = json.loads((out / "plan.json").read_text())
        else:
            out = new_output("viddent_matrix")
            plan = {"cases": list(dict.fromkeys(args.cases)), "swin_recompute": args.swin_recompute}
            atomic_json(out / "plan.json", plan)
        print(f"Run directory: {out}", flush=True)
        with (out / f"attempt_{datetime.now():%Y%m%d_%H%M%S_%f}.log").open("x", encoding="utf-8") as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                execute_plan(out, plan, tables)


def execute_plan(out, plan, tables):
    for case in plan["cases"]:
        if case not in CASES:
            raise ValueError(f"Unexpected or already completed configuration: {case}")
        try:
            run_case(out / case, case, tables, plan["swin_recompute"])
        except Exception as exc:
            atomic_json(out / "last_failure.json", {"case": case, "error": repr(exc)})
            print(f"{case} stopped: {exc!r}", file=sys.stderr, flush=True)
            raise
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
