"""CPU-only MD01 homography animation; never downloads or imports training code."""
import csv, hashlib, io, json, math, sys, time, zipfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
MANIFEST=ROOT/'outputs/runs/mendeley_i2v_feasibility_2026-10-20_01/source_images.csv'
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
def main():
    global OUT
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    global MANIFEST
    args = parser.parse_args()
    MANIFEST = args.manifest.resolve()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    assert not list(OUT.glob('*.mp4')) and not (OUT/'method_a_metrics.json').exists(),'Refusing overwrite'
    index_before=sha(ROOT/'.git/index')
    cv2.setNumThreads(4)
    started=time.perf_counter()
    rows=list(csv.DictReader(MANIFEST.open(encoding='utf-8',newline='')))
    row=next(r for r in rows if r['source_id']=='MD01')
    assert row['class_label']=='Lower Front' and row['binary_label']=='Lower'
    with zipfile.ZipFile(args.archive.resolve()) as z:raw=z.read(row['archive_member'])
    assert hashlib.sha256(raw).hexdigest()==row['sha256']
    with Image.open(io.BytesIO(raw)) as image:
        rgb=np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
    image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    h,w=image.shape[:2]
    assert (w,h)==(1280,550)
    corners=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    mask=np.full((h,w),255,np.uint8)
    frames=[];poses=[];changes=[];coverage=[]
    render_start=time.perf_counter()
    for i in range(16):
        phase=2*math.pi*i/15
        dx=3*math.sin(phase);dy=1.5*math.sin(2*phase)
        angle=.35*math.sin(phase)
        zoom=1.035+.005*math.sin(2*phase)
        affine=cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),angle,zoom)
        affine[:,2]+=np.array([dx,dy])
        affine=np.vstack([affine,[0,0,1]])
        # Subpixel corner offsets simulate a mild planar perspective change only.
        q=.6*math.sin(phase);v=.4*math.sin(2*phase)
        offsets=np.float32([[-q,-v],[q,v],[-q,v],[q,-v]])
        projective=cv2.getPerspectiveTransform(corners,corners+offsets)
        H=projective@affine
        # Require the output view to stay inside the original image; no reflected/inpainted borders.
        source_corners=cv2.perspectiveTransform(corners[None],np.linalg.inv(H))[0]
        assert source_corners[:,0].min()>=1 and source_corners[:,0].max()<=w-2
        assert source_corners[:,1].min()>=1 and source_corners[:,1].max()<=h-2
        valid=cv2.warpPerspective(mask,H,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
        assert valid.min()==255,'Uncovered output pixel; stop rather than synthesize border'
        frame=cv2.warpPerspective(image,H,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        frames.append(frame)
        coverage.append(float(np.mean(valid==255)))
        poses.append(dict(frame_number=i+1,time_seconds=i/8,translation_x_px=dx,translation_y_px=dy,rotation_degrees=angle,zoom=zoom,perspective_corner_offsets_px=offsets.tolist(),homography_source_to_output=H.tolist(),inverse_output_corners=source_corners.tolist()))
        if i:changes.append(float(np.abs(frame.astype(np.float32)-frames[i-1].astype(np.float32)).mean()))
    render_seconds=time.perf_counter()-render_start
    video=OUT/'MD01_method_A_geometric.mp4'
    encode_start=time.perf_counter()
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),8,(w,h))
    assert writer.isOpened(),'MP4 codec unavailable; no package installs/fallback formats'
    try:
        for frame in frames:writer.write(frame)
    finally:writer.release()
    encode_seconds=time.perf_counter()-encode_start
    cap=cv2.VideoCapture(str(video));assert cap.isOpened()
    metadata_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    decoded=[];codec=int(cap.get(cv2.CAP_PROP_FOURCC))
    try:
        while True:
            ok,f=cap.read()
            if not ok:break
            assert f.shape==(h,w,3);decoded.append(f)
    finally:cap.release()
    assert len(decoded)==metadata_count==16 and abs(fps-8)<1e-6
    assert np.abs(decoded[0].astype(float)-decoded[8].astype(float)).mean()>.1
    compression_mae=float(np.mean([np.abs(a.astype(np.float32)-b.astype(np.float32)).mean() for a,b in zip(frames,decoded)]))
    elapsed=time.perf_counter()-started
    config=dict(method='2D planar homography, no depth and no generative model',source=dict(row),width=w,height=h,frames=16,fps=8,duration_seconds=2,seed=None,deterministic=True,trajectory='one smooth sinusoidal loop, phase=2*pi*i/15',translation_x_amplitude_px=3,translation_y_amplitude_px=1.5,rotation_amplitude_degrees=.35,zoom_base=1.035,zoom_amplitude=.005,perspective_corner_amplitudes_px=[.6,.4],resampling='OpenCV INTER_LINEAR',border='constant; never sampled, verified full valid coverage',codec_requested='mp4v',frame_label='Lower Front / Lower, inherited from source, not independently clinically verified')
    save('method_a_config.json',config|{'poses':poses})
    metrics=dict(status='PASS',source_sha256=row['sha256'],manifest_sha256=sha(MANIFEST),script_sha256=sha(__file__),video_sha256=sha(video),video_bytes=video.stat().st_size,decoded_frame_count=len(decoded),metadata_frame_count=metadata_count,decoded_fps=fps,width=w,height=h,duration_seconds=len(decoded)/fps,codec_fourcc=''.join(chr((codec>>(8*k))&255) for k in range(4)),full_source_pixel_coverage_each_frame=coverage,mean_consecutive_frame_absolute_differences_0_255=changes,encoding_mean_absolute_error_0_255=compression_mae,render_seconds=render_seconds,encode_seconds=encode_seconds,total_load_render_encode_verify_seconds=elapsed,python=sys.version,opencv=cv2.__version__,numpy=np.__version__,gpu_used=False,git_index_unchanged=(sha(ROOT/'.git/index')==index_before),method_b_status='PENDING approval for real depth checkpoint/dependencies')
    assert metrics['git_index_unchanged']
    save('method_a_metrics.json',metrics)
    report=f"""# MD01 synthetic dental video pilot — Method A completed
Source: MD01, LF 0464.jpg; original class Lower Front, derived binary Lower.
Source SHA-256: {row['sha256']}.
Source bytes were read from the existing ZIP and verified against the feasibility manifest. No image, label, dataset or prior result was changed.

## Method A
Output: MD01_method_A_geometric.mp4 — verified by full decoding: **16 frames, 1280 x 550, 8 FPS, 2.0 seconds**. MPEG-4 Part 2 (mp4v) encoding; no audio.
Each frame is a single small homography of the source. Smooth loop: translation at most 3 px horizontally / 1.5 px vertically, rotation at most 0.35 degrees, scale between 1.030 and 1.040, perspective corner offsets at most 0.6/0.4 px. Baseline zoom limits approximate per-axis edge loss to about 3–4%; it is not an aggressive crop. Frame 16 returns to frame 1's pose for a continuous loop.
All 16 output masks are 100% source-covered and inverse-mapped corners lie within the input; no empty borders, reflection padding, inpainting or invented pixels were used. The tiny projective transform preserves topology but is not physical 3D camera motion. View-label preservation is intended and inherited, not independently established by clinical assessment.
Anatomical content comes from the original photograph: no teeth are generated or removed. Interpolation and lossy encoding can soften fine details, and a planar warp moves all depths together without genuine parallax. No claim of clinical fidelity or new anatomical viewpoints is made.
Decoded-versus-rendered mean absolute pixel error: {compression_mae:.4f}/255. Consecutive-frame changes are nonzero; this is not a repeated still-frame video.

## Runtime
CPU only, four OpenCV threads. Rendering: {render_seconds:.3f} s; encoding: {encode_seconds:.3f} s; load + rendering + encoding + full decode verification: {elapsed:.3f} s. No GPU inference, downloads or installations.

## Method B — pending approval
Recommended actual estimator: Depth Anything V2 Small (24.8M parameters), official HF conversion depth-anything/Depth-Anything-V2-Small-hf, Apache-2.0.
Checkpoint: model.safetensors, 99,173,660 bytes (99.2 MB decimal / about 94.6 MiB), plus small config.json and preprocessor_config.json. Official model listing: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/tree/main
Model SHA-256 published in the official weight pointer: 3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70.
The checkpoint was not found in the inspected local Torch/HF caches.
Use a separately approved isolated environment with the existing compatible Torch, NumPy and Pillow plus Transformers and its resolved dependencies (notably safetensors, huggingface-hub and tokenizers). These additional libraries are absent from the current environment; exact pinned install plan and total dependency size must be checked before installation. Accelerate is not required if we explicitly load and move the model to CPU, rather than use device_map='auto'. No xformers/timm installation is needed for the Transformers route.
Recommend one CPU float32 depth inference initially; this avoids P2000 kernel/VRAM uncertainty. The small model is a plausible 5 GB GPU candidate, but neither runtime nor peak memory has been benchmarked here.
Estimate relative inverse depth from this same source, verify finite/nonconstant output, and save the real raw estimate and one visualization. Depth scale is ambiguous and teeth/reflections can fool a general-domain estimator. Converting estimated inverse depth into bounded relative geometry is necessary, but an arbitrary hand-made gradient must never replace the estimate.
Then render a conservative image-based 2.5D camera path (roughly 1–3 px differential parallax), with visibility/coverage checks, bounded hole handling and logged hole fractions. It cannot reveal genuinely unseen surfaces or guarantee metric anatomy.
Required final Method B settings: exactly 16 frames, 1280 x 550, 8 FPS. Save only one Method B MP4, one depth visualization and one final A/B contact sheet of corresponding frame numbers in this run directory. Do not recreate Method A or make extra MP4 variants. Assess stretching, disocclusion, tooth contours and label preservation before claiming improvement.
No Method B video, depth visualization or A/B comparison sheet is produced yet; fabricating these would misrepresent a method that has not run.

## Reproduction and files
run_method_a.py (this script), method_a_config.json (all 16 matrices and settings), method_a_metrics.json (checksums/verification/runtime), MD01_method_A_geometric.mp4, report.md.
Run with the repository .venv Python explicitly. The script refuses reruns over its existing artifacts. The directory is ignored by Git; the existing staged baseline is unchanged.
"""
    with (OUT/'report.md').open('x',encoding='utf-8') as f:f.write(report)
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
