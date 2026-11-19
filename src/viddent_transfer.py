"""Build and verify a private, allowlisted VidDent train/validation archive."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import uuid

import pandas as pd

from data.dataset import DentalQuadrantDataset
from viddent_matrix import CACHED, CLEANED, FIELDS, MANIFEST, REFERENCE, fixed_subsets, validation_tables
from viddent_protocol import FRAMES, INPUTS, OVERNIGHT, sha

INVENTORY = "SHA256SUMS"


def evidence_paths():
    yield OVERNIGHT / "train_manifest.csv"
    yield OVERNIGHT / "validation_manifest.csv"
    yield REFERENCE
    yield OVERNIGHT / "partial/best_val_reloaded_predictions.csv"
    for cached in (CACHED, CACHED.with_name("original_balanced_seed42")):
        best = cached / "best_checkpoint.json"
        epoch = int(json.loads(best.read_text())["epoch"])
        if not 1 <= epoch <= 15:
            raise ValueError("Unexpected cached checkpoint epoch")
        yield best
        yield cached / f"epoch_{epoch:02d}_val_predictions.csv"


def sampled_frame_paths():
    paths = set()
    for split, manifest in (("train", OVERNIGHT / "train_manifest.csv"),
                            ("val", OVERNIGHT / "validation_manifest.csv")):
        dataset = DentalQuadrantDataset(manifest, data_root=FRAMES, split=split)
        for row in dataset.df.itertuples():
            if str(uuid.UUID(row.uuid)) != row.uuid or not 1 <= row.start_frame <= row.end_frame <= row.total_frames:
                raise ValueError("Invalid UUID or frame boundaries")
            folder = FRAMES / split / row.uuid / "GT"
            for position in dataset.temporal_positions:
                indices = dataset._sample_clip_indices(row.start_frame, row.end_frame, position)
                paths.update(folder / f"{number:05d}.png" for number in indices)
    return sorted(paths)


def private_inputs():
    original, cleaned, val = validation_tables()
    fixed_subsets(val)
    historical = pd.read_csv(OVERNIGHT / "train_manifest.csv")
    if not historical[FIELDS].equals(original[FIELDS]):
        raise ValueError("Original training differs from the historical frame manifest")
    return original, cleaned, val, sampled_frame_paths()


def checksummed_sources(frames):
    archived = pd.read_csv(OVERNIGHT / "fixed_frames.csv", usecols=["path", "sha256"])
    historical = dict(zip(archived.path.str.replace(chr(92), "/", regex=False), archived.sha256))
    relative_frames = {p.relative_to(INPUTS).as_posix() for p in frames}
    if set(historical) != relative_frames:
        raise ValueError("Sampled frame list differs from historical provenance")
    for source in [*evidence_paths(), *frames]:
        relative = source.relative_to(INPUTS).as_posix()
        if source.resolve() != source or not source.is_file():
            raise ValueError(f"Expected a regular local input: {source}")
        checksum = sha(source)
        if relative in historical and historical[relative] != checksum:
            raise ValueError(f"Frame checksum differs from historical provenance: {source}")
        yield relative, source, checksum


def archive_bytes(archive, name, content):
    item = tarfile.TarInfo(name)
    item.size = len(content)
    item.mode = 0o600
    archive.addfile(item, io.BytesIO(content))
    return hashlib.sha256(content).hexdigest()


def create_archive(destination):
    original, cleaned, val, frames = private_inputs()
    lines = []
    with tarfile.open(destination, "x") as archive:
        for path, table in [(MANIFEST, pd.concat([original, val])), (CLEANED, cleaned)]:
            name = path.relative_to(INPUTS).as_posix()
            checksum = archive_bytes(archive, name, table.to_csv(index=False).encode("utf-8"))
            lines.append(f"{checksum}  {name}")
        for name, source, checksum in checksummed_sources(frames):
            with source.open("rb") as stream:
                item = tarfile.TarInfo(name)
                item.size = source.stat().st_size
                item.mode = 0o600
                archive.addfile(item, stream)
            lines.append(f"{checksum}  {name}")
        archive_bytes(archive, INVENTORY, ("\n".join(lines) + "\n").encode("utf-8"))
    print(f"Packed {len(frames)} unique train/validation GT PNGs. Archive SHA-256: {sha(destination)}")


def verify_inputs():
    _, _, _, frames = private_inputs()
    with MANIFEST.open(encoding="utf-8-sig") as stream:
        if any(row["split"] not in {"train", "val"} for row in csv.DictReader(stream)):
            raise ValueError("Private Original manifest must contain only train/validation rows")
    expected = {path.relative_to(INPUTS).as_posix()
                for path in [MANIFEST, CLEANED, *evidence_paths(), *frames]}
    checksums = {}
    for line in (INPUTS / INVENTORY).read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        if name in checksums:
            raise ValueError(f"Duplicate inventory entry: {name}")
        checksums[name] = digest
    if set(checksums) != expected:
        raise ValueError("Transfer inventory differs from the train/validation allowlist")
    for name, checksum in checksums.items():
        path = INPUTS / name
        if path.resolve() != path or sha(path) != checksum:
            raise ValueError(f"Transfer checksum mismatch: {name}")
    print(f"Verified {len(frames)} unique GT PNGs and {len(expected) - len(frames)} supporting files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--pack", type=Path, metavar="PRIVATE_TAR")
    modes.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.pack:
        create_archive(args.pack.resolve())
    else:
        verify_inputs()
