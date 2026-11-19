# VideoMamba-Tiny feasibility for VidDent-Real

Status: **NEEDS LINUX GPU TEST**. This is a short compatibility probe for the official ECCV 2024 architecture. The existing matrix runner and optional S3D baseline remain available.

Use the official **VideoMamba-Ti** checkpoint **videomamba_t16_k400_f16_res224.pth**. It is ImageNet-1K initialized and Kinetics-400 fine-tuned. The checkpoint has a 400-by-192 classifier, temporal positions for 16 frames, and 1-frame tubelets with 16-by-16 patches. The 28.3 MB public file was downloaded and verified with SHA-256 and a CPU weights-only load. Tiny is the lowest-capacity variant with a 16-frame Kinetics-400 checkpoint. The source and published model card list Apache-2.0. Keep upstream notices with upstream code and weights. [Official model zoo](https://github.com/OpenGVLab/VideoMamba/blob/main/videomamba/video_sm/MODEL_ZOO.md) | [Checkpoint and SHA-256](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_t16_k400_f16_res224.pth) | [License](https://github.com/OpenGVLab/VideoMamba/blob/main/LICENSE)

## Study protocol

The adapter strictly loads the 400-class checkpoint and replaces its classifier with a 2-output linear head, Upper=0 and Lower=1. Linear trains the head alone. Partial trains the last of 24 backbone blocks and the head. Full trains every parameter. The initial probe uses FP32, evaluation mode with autograd enabled for training, one clip at a time, five losses averaged per optimizer update, and strict PyTorch determinism. The upstream custom kernels still need a runtime reproducibility check.

Both architectures take five 16-frame clips at 224 by 224, with the same RGB and ImageNet normalization. VideoMamba uses aspect-preserving bilinear short-side resizing to 224 and a single center crop for both train and validation. VideoSwin uses a square resize. Both study models use the existing VidDent temporal windows at 10%, 30%, 50%, 70%, and 90% of each annotated segment, with the same short-segment fallback. This retains the same private frame allowlist and differs from the upstream Kinetics sparse whole-video sampling and multiple crops. Keep the spatial transform difference visible when interpreting the architecture comparison. The three datasets used by the adapter are Original train, Cleaned train, and fixed validation via the shared project manifests. The real-data probe reads two Original training segments and one validation segment. [Official 16-frame recipe](https://github.com/OpenGVLab/VideoMamba/blob/main/videomamba/video_sm/exp/k400/videomamba_tiny/run_f16x224.sh) | [Official preprocessing](https://github.com/OpenGVLab/VideoMamba/blob/main/videomamba/video_sm/datasets/kinetics_sparse.py)

The eventual runner change is a lazy model and dataset dispatch for VideoMamba and six new case IDs, plus provenance for the upstream commit, checkpoint digest, preprocessing, and kernel environment. Keep the existing full-validation macro-F1 selection and earliest tie, five-clip mean probabilities, accuracy, balanced accuracy, recalls, confusion matrices, and segment predictions. The Easy and Hard manifests stay fixed and are scored after checkpoint selection.

| Training data | Architecture | Linear | Partial | Full |
|---|---|---|---|---|
| Original | VideoSwin-T | Completed | Completed | Missing |
| Cleaned | VideoSwin-T | Completed, cached protocol | Missing | Missing |
| Original | VideoMamba-Ti | Needs Linux GPU test | Needs Linux GPU test | Needs Linux GPU test |
| Cleaned | VideoMamba-Ti | Needs Linux GPU test | Needs Linux GPU test | Needs Linux GPU test |

## Linux installation and probe

Run from the VidDent source directory. The commands create a separate virtual environment. Use the lab CUDA 12.6 development toolkit with nvcc and a supported host compiler. Check nvidia-smi, nvcc, and device compute capability before building. The published dependency file references torch 2.1.1 / torchvision 0.16.1 with CUDA 11.8. The following commands test the study's torch 2.14.0 / torchvision 0.29.0 CUDA 12.6 stack. That pairing has not been validated by this work.

The repository vendors a BiMamba ABI that the model calls with bimamba=True. Build its own causal-conv1d and mamba_ssm and use the Triton version selected by PyTorch. The old fused RMSNorm can also be sensitive to Triton version changes. Both extension setup files explicitly generate SM70, SM80, and SM90 code with CUDA 11.8 or newer. A newer GPU family such as Blackwell needs a separate build compatibility check. If the host GPU or ABI is unsupported, keep the current matrix and capture the first error. [Bundled Mamba setup](https://github.com/OpenGVLab/VideoMamba/blob/main/mamba/setup.py) | [Bundled causal-conv1d setup](https://github.com/OpenGVLab/VideoMamba/blob/main/causal-conv1d/setup.py)

```bash
set -euo pipefail
umask 077
nvidia-smi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
c++ --version

python3.11 -m venv "$HOME/.venvs/viddent-videomamba-cu126"
source "$HOME/.venvs/viddent-videomamba-cu126/bin/activate"
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-viddent.txt setuptools==68.2.2 wheel==0.42.0 packaging==24.0 ninja timm==0.4.12 einops==0.7.0 transformers==4.36.1
export CUDA_VISIBLE_DEVICES=0
python -c 'import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__); print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(), torch.cuda.mem_get_info()); print(torch.ones(1, device="cuda"))'

vm_work=$(mktemp -d "$HOME/viddent-videomamba.XXXXXX")
vm_source="$vm_work/OpenGVLab-VideoMamba"
git clone https://github.com/OpenGVLab/VideoMamba.git "$vm_source"
git -C "$vm_source" checkout --detach 37355c26d0ae99ca2459f6d4044a5f509031a79f
export MAX_JOBS=2
CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip install --no-build-isolation --no-deps "$vm_source/causal-conv1d"
MAMBA_FORCE_BUILD=TRUE python -m pip install --no-build-isolation --no-deps "$vm_source/mamba"
python -m pip check
python -c 'import causal_conv1d_cuda, selective_scan_cuda; from mamba_ssm.ops.triton.layernorm import RMSNorm; from mamba_ssm.modules.mamba_simple import Mamba; import inspect; assert "bimamba" in inspect.signature(Mamba).parameters'

vm_weights="$vm_work/videomamba_t16_k400_f16_res224.pth"
curl --fail --location --output "$vm_weights" https://huggingface.co/OpenGVLab/VideoMamba/resolve/main/videomamba_t16_k400_f16_res224.pth
printf '%s  %s\n' a335d728ae4dbe4f49a435022f95c6cf98108d20fe084120db1f18cb73e84f4a "$vm_weights" | sha256sum -c -

for strategy in linear partial full; do
    python -B src/viddent_videomamba.py "$strategy" --source "$vm_source" --checkpoint "$vm_weights" --updates 2 2>&1 | tee "$vm_work/$strategy-synthetic.log"
done

# Point to the already-verified, private Linux transfer directory.
export VIDDENT_INPUTS="$HOME/viddent-private"
python -B src/viddent_transfer.py --verify
for strategy in linear partial full; do
    python -B src/viddent_videomamba.py "$strategy" --source "$vm_source" --checkpoint "$vm_weights" --updates 3 --viddent 2>&1 | tee "$vm_work/$strategy-viddent.log"
done
```

Each probe checks finite gradients, head and expected backbone updates, frozen parameter and buffer integrity, and prediction agreement after a strict checkpoint reload. It logs actual update times and peak allocated and reserved GPU memory. Exact VRAM remains unknown. Roughly 7 million parameters imply around 0.1 GiB for FP32 parameters, gradients, and AdamW moments in full tuning. Saved activations across 24 blocks with about 3,137 tokens and CUDA workspaces account for the larger, unmeasured allocation. Reserve at least 8 GiB free for the first probe as a scheduling allowance and use the measured peak to decide whether to proceed. Stop on import, build, kernel, gradient, reload, or memory errors.

Budget each eventual case from steady update time times 1,890 Original or 1,380 Cleaned updates, plus 195 validation segment evaluations and checkpoint/I/O overhead. Tonight is suitable for installation and probes. Major training needs the three passing real-input probe logs and the runner registration review.
