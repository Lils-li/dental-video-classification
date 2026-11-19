# VidDent-Real Linux migration

The matrix runner uses the existing 13-segment validation split and checksum-pinned historical predictions for the two diagnostic subsets. It reads RGB GT PNGs from train and val. The private transfer includes the exact five-by-sixteen clip inputs used by the study. Original and Cleaned have 81 and 53 train segments, respectively. They share validation and use disjoint train/validation UUID groups.

## Source and private inputs

On Windows, run from the repository root. Use a new filename if the private archive already exists. The source archive comes from the reviewed Git index and the data archive comes from the explicit train/validation allowlist.

```powershell
$sourceTree = git write-tree
if ($LASTEXITCODE -ne 0) { throw 'Unable to write staged tree' }
git archive --format=tar --output=outputs/viddent_linux_source.tar $sourceTree
if ($LASTEXITCODE -ne 0) { throw 'Unable to archive staged source' }
.\.venv\Scripts\python.exe -B src/viddent_transfer.py --pack outputs/viddent_private_trainval_20261119.tar
Get-FileHash -Algorithm SHA256 outputs/viddent_linux_source.tar, outputs/viddent_private_trainval_20261119.tar
```

Use the existing archive if the pack command reports that the destination exists. Verify its SHA-256 before copying. Transfer the two archives over the lab's approved private channel. For a private SSH destination, set the actual account, host, and private destination first.

```powershell
$lab = 'LAB_USER@LAB_HOST'
$remotePrivate = '/absolute/path/to/private/incoming'
scp outputs/viddent_linux_source.tar outputs/viddent_private_trainval_20261119.tar "${lab}:${remotePrivate}/"
```

The private archive contains a train/val-only copy of the Original manifest, the Cleaned training manifest, sampled train and val GT PNGs, two prior run manifests, and validation CSV/JSON evidence for the reused VideoSwin results. It includes no raw video, annotation workbook, evaluation checkpoint, or environment. The source archive also excludes the Windows virtual environment. Download the two public pretrained checkpoints on Linux using the commands below.

## Linux setup and verification

Start in the private incoming directory. Compare both printed archive hashes with the Windows values before extraction. Choose fresh destination directories.

```bash
set -e
umask 077
sha256sum viddent_linux_source.tar viddent_private_trainval_20261119.tar
mkdir "$HOME/viddent-code" "$HOME/viddent-private" "$HOME/viddent-runs"
tar -xf viddent_linux_source.tar -C "$HOME/viddent-code"
tar -xf viddent_private_trainval_20261119.tar -C "$HOME/viddent-private"
cd "$HOME/viddent-code"
export VIDDENT_INPUTS="$HOME/viddent-private"
export VIDDENT_RUNS="$HOME/viddent-runs"
export TORCH_HOME="$HOME/.cache/viddent-torch"
export CUDA_VISIBLE_DEVICES=0

nvidia-smi
df -h "$VIDDENT_INPUTS" "$VIDDENT_RUNS"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-viddent.txt
python -m pip check
python -B -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(), torch.cuda.mem_get_info()); print(torch.ones(1, device="cuda"))'
python -B src/viddent_transfer.py --verify
python -B src/viddent_matrix.py audit
python -B -c 'from torchvision.models.video import Swin3D_T_Weights, S3D_Weights; [w.get_state_dict(progress=True, check_hash=True) for w in (Swin3D_T_Weights.KINETICS400_V1, S3D_Weights.KINETICS400_V1)]'
```

Use an NVIDIA driver that supports the installed CUDA wheel and GPU architecture. A CUDA Toolkit installation is unnecessary for these prebuilt wheels. If the Linux GPU requires a different PyTorch CUDA wheel, record the wheel and driver versions and repeat the probes before training. The model code checks the full SHA-256 of each official weight file. To use an existing private weight cache, set `VIDDENT_WEIGHTS` to the directory containing `swin3d_t-7615ae03.pth` and `s3d-d76dad2f.pth`. The default is the Torch Hub checkpoints directory. [PyTorch wheel installation](https://pytorch.org/get-started/previous-versions/) ? [NVIDIA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

## GPU probes and approved training

Run probes from the initialized shell with the selected GPU idle. The probes record trainable parameters, actual gradients and AdamW updates, timing, peak allocated/reserved GPU memory, free disk, and free VRAM. Record the GPU/driver from `nvidia-smi`. Swin full must demonstrate an actual backbone update with the existing FP32 16-frame 224-pixel clips. The probe can retry with activation recomputation following a CUDA OOM. Carry that setting into training only if the recomputation probe passed. If both fail, use a larger GPU and retain the existing input protocol.

```bash
set -o pipefail
python -B src/viddent_matrix_model.py swin full --updates 5 2>&1 | tee "$VIDDENT_RUNS/swin-full-preflight.log"
python -B src/viddent_matrix_model.py s3d linear --updates 3 2>&1 | tee "$VIDDENT_RUNS/s3d-linear-preflight.log"
python -B src/viddent_matrix_model.py s3d partial --updates 3 2>&1 | tee "$VIDDENT_RUNS/s3d-partial-preflight.log"
python -B src/viddent_matrix_model.py s3d full --updates 3 2>&1 | tee "$VIDDENT_RUNS/s3d-full-preflight.log"
```

For each case, estimate duration as measured steady seconds per update times 1,890 Original or 1,380 Cleaned updates, plus 195 validation segment evaluations and checkpoint/I/O overhead. Use the Linux measurements for the overnight queue. The Windows P2000 established that the full Swin path fits at about 2.04 GiB reserved. Three measured full-S3D updates on Windows used about 0.71 GiB reserved. Full S3D keeps strict determinism outside training updates and uses warn-only determinism during updates because CUDA max-pool backward can be nondeterministic. S3D keeps its BatchNorm running statistics frozen for the study's one-clip microbatches.

After approval, give each host distinct cases. Prioritize the two full VideoSwin cases on Linux and the four S3D linear/partial cases on Windows. The same user account can run only one matrix runner or benchmark per host at a time. Inspect system GPU activity for jobs outside this runner. Continue in order of available time and reassess after the first epoch.

```bash
# Linux priority. Add --swin-recompute only if that was the successful probe setting.
python -B src/viddent_matrix.py train --cases original-swin-full cleaned-swin-full

# Linux follow-up if time remains and the S3D determinism exception is approved.
python -B src/viddent_matrix.py train --cases original-s3d-full cleaned-s3d-full

# Completes the remaining VideoSwin partial case if it has no assignment elsewhere.
python -B src/viddent_matrix.py train --cases cleaned-swin-partial

# Alternative for Linux if these four cases have no Windows assignment.
python -B src/viddent_matrix.py train --cases original-s3d-linear cleaned-s3d-linear original-s3d-partial cleaned-s3d-partial

# Resume the interrupted queue on the same host.
python -B src/viddent_matrix.py train --resume "$VIDDENT_RUNS/viddent_matrix_REPLACE_WITH_RUN_TIMESTAMP"
```

Windows priority, from the existing repository:

```powershell
.\.venv\Scripts\python.exe -B src/viddent_matrix.py train --cases original-s3d-linear cleaned-s3d-linear original-s3d-partial cleaned-s3d-partial
```

Use separate timestamped output roots per host. Copy selected run artifacts back over the private channel. The runner skips the three verified historical cells and previously completed runs with matching protocol in its output root. A failure stops the queue. Resume restores optimizer and model state at the last completed epoch. Full validation macro-F1 determines checkpoint selection. Easy and Hard are reported only after selection.
