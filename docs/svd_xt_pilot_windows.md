# SVD-XT dental pilot on PC 2

Prepared for Windows, RTX 5090 32 GB and Python 3.13.13. This is a proposed pinned environment, syntax-validated on PC 1; installation and GPU generation have not been validated on PC 2. Do not install into the existing Python/VideoSwin environment. Codex is not required on PC 2.

## Setup (PC 2 only)

Open PowerShell at the cloned repository root. Run these commands only after the pilot files have been approved, committed and pushed from PC 1. Stop on any nonzero native-command exit code; do not proceed after a failed install. The environment must be new; if it already exists, inspect it or choose another name rather than overwriting it.

```powershell
git pull --ff-only origin main
python --version
nvidia-smi
$SvdEnv = Join-Path $env:USERPROFILE '.venvs\dental_svd_xt'
if (Test-Path -LiteralPath $SvdEnv) { throw 'Choose a new environment directory' }
python -m venv $SvdEnv
if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
$SvdPython = Join-Path $SvdEnv 'Scripts\python.exe'
& $SvdPython -m pip install pip==25.2
if ($LASTEXITCODE -ne 0) { throw 'pip setup failed' }
& $SvdPython -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $SvdPython -m pip install -r .\requirements-svd-xt.txt
if ($LASTEXITCODE -ne 0) { throw 'Pilot dependency installation failed' }
& $SvdPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
& $SvdPython -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())"
if ($LASTEXITCODE -ne 0) { throw 'CUDA check failed' }
```

Use a current NVIDIA driver compatible with CUDA 12.8. The cu128 wheel supplies its runtime; this workflow does not require installing a separate CUDA toolkit. Verify the actual device and compiled architectures; the pilot also runs a tiny CUDA matrix multiplication before model loading. No torchvision, xformers, FlashAttention, torch.compile or custom CUDA build is required. The default is fp16 with UNet forward chunking and five-frame VAE decoding. Free-VRAM checks are a planning gate, not a fit guarantee. If an OOM occurs, retain the failed run, close other GPU programs and choose a new output directory with --cpu-offload (requires system RAM) and/or --decode-chunk-size 2. Smaller decode chunks can affect temporal consistency. Do not change precision/parameters silently between experiments.

The dependency file pins direct packages, not the complete transitive resolution. Save pip freeze with each setup/run; matching seeds do not guarantee identical results across GPUs, drivers or library builds. Do not copy the existing PC 1 virtual environment.

## Model access, inputs and first run (PC 2 only)

Review the model license/access terms on its Hugging Face page. If authentication is required, accept access in a browser and run the command below; enter the token interactively and never put it in source, metadata or command-line arguments:

```powershell
& $SvdPython -c "from huggingface_hub import login; login(add_to_git_credential=False)"
```

Place one authorized JPG at data\mendeley_pilot\input.jpg on PC 2 (create that local directory first). No PC 1 manifest, archive, UUID mapping, checkpoint or absolute path is needed. The source image and all generated outputs remain ignored. Use a de-identified photograph; source licensing and privacy remain the researcher's responsibility. Approximately 305 GB free disk should leave room for one pilot, but confirm the actual cache/output drives. The script requires 30 GiB free before online model download and 2 GiB on the output drive; these are conservative planning floors, not measured usage limits. Also allow space for the dedicated Python environment and system RAM/pagefile.

The next command queries model metadata only and records the current immutable revision. For repeats, reuse the saved 40-character revision rather than resolving main again. The generation script downloads fp16 safetensors/config files into the ignored cache on PC 2. It does not download the redundant original .safetensors/.ckpt or fp32 model variants.

```powershell
$SvdRevision = (& $SvdPython -c "from huggingface_hub import HfApi; print(HfApi().model_info('stabilityai/stable-video-diffusion-img2vid-xt').sha)").Trim()
if ($LASTEXITCODE -ne 0 -or $SvdRevision -notmatch '^[0-9a-f]{40}$') { throw 'Could not resolve model revision' }
& $SvdPython .\src\synthetic\generate_svd_xt.py --input .\data\mendeley_pilot\input.jpg --output-dir .\outputs\runs\svd_xt_pilot_01 --revision $SvdRevision --check-only
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed' }
& $SvdPython .\src\synthetic\generate_svd_xt.py --input .\data\mendeley_pilot\input.jpg --output-dir .\outputs\runs\svd_xt_pilot_01 --revision $SvdRevision --seed 42 --steps 25 --motion-bucket-id 127 --noise-aug-strength 0.02 --conditioning-fps 7 --fps 6 --decode-chunk-size 5 --cache-dir .\outputs\model_cache\svd_xt
if ($LASTEXITCODE -ne 0) { throw 'Generation failed; retain partial output and inspect traceback' }
& $SvdPython -m pip freeze | Set-Content -LiteralPath .\outputs\runs\svd_xt_pilot_01\environment_freeze.txt -Encoding UTF8
```

--check-only validates the JPG, padding and CUDA/memory (including one tiny CUDA kernel), without loading/downloading the model, generating video, or creating output files. --help imports only the Python standard library. Use --local-files-only for repeats after all required model files are cached. Every generation requires a nonexistent output directory; failures are retained without automatic retries. Paths resolve from your working directory, so start at the repository root. No existing source/data/output is overwritten.

## Output and motion settings

The clip is exactly 25 frames at 1024x576. At 6 playback FPS its container duration is 25/6 = 4.17 seconds; the last frame starts at 4.0 seconds. Model conditioning FPS is separate (7 by default; Diffusers internally applies its trained FPS offset). Increasing motion-bucket-id generally increases motion; increasing noise-aug-strength can reduce resemblance to the source. Neither parameter guarantees realistic dental camera motion or preserved anatomy. There is no text prompt control in SVD-XT.

The image is EXIF-oriented, converted to RGB, resized with preserved aspect ratio and centered on a neutral gray 1024x576 canvas. No source pixels are deliberately cropped. The conditioning PNG and exact content box document padding. Padding may itself influence generation, and the model can alter anatomy even inside the source content region. Extremely wide/tall inputs leave less effective dental resolution. Inspect conditioning.png before accepting the video.

Outputs: svd_xt.mp4 (H.264, 25 decoded/verified frames), contact_sheet.png (all 25 frames), conditioning.png, request.json, and generation_metadata.json. Metadata records source/script/model-file SHA-256 values, immutable model revision, parameters, package versions, hardware, timing, peak VRAM and artifact hashes. The contact sheet uses pre-encoding frames; inspect the actual MP4 too because encoding is lossy. Failed runs retain request/partial artifacts and failure.json. No training label is assigned automatically.

## Visual audit before VideoSwin use

Generated teeth, restorations, gingiva, occlusion, tools and camera motion may be hallucinated or change between frames. SVD-XT is a general image-to-video model, not a validated dental simulator. Check all frames/full-resolution playback against the photograph for tooth count/identity, anatomical topology, view-label preservation, temporal flicker, distortion, unwanted padding motion and clinically misleading content. Record accept/reject and reviewer notes separately; unreviewed or rejected clips must not enter training.

One photograph cannot establish benefit to VideoSwin. Split by original source image/patient before generation; keep all seeds/clips/derivatives of that source in the same split. Do not contaminate real validation/test sets with related synthetic material. Assess downstream benefit in a separately designed training experiment with real held-out evaluation. This pilot runs generation only.

## Primary references

- https://huggingface.co/docs/diffusers/v0.35.1/using-diffusers/svd
- https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
- https://pytorch.org/get-started/previous-versions/ (2.11.0 Windows CUDA 12.8 wheels)
