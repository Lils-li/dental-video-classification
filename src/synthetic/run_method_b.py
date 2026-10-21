"""Actual Depth Anything V2 Small inference + conservative 2.5D splatting for MD01 only."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import csv,hashlib,io,json,math,sys,time,threading,subprocess,importlib.metadata
from pathlib import Path
import numpy as np
import cv2,psutil,torch
from PIL import Image,ImageOps,ImageDraw
from transformers import DPTImageProcessor,AutoModelForDepthEstimation
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def decode(p):
    c=cv2.VideoCapture(str(p));assert c.isOpened()
    fps=c.get(cv2.CAP_PROP_FPS);frames=[]
    try:
        while True:
            ok,f=c.read()
            if not ok:break
            assert f.shape==(550,1280,3)
            frames.append(f)
    finally:c.release()
    assert len(frames)==16 and abs(fps-8)<1e-6
    return frames
def project(z,u,v,fx,cx,cy,phase):
    yaw=math.radians(.08)*math.sin(phase);roll=math.radians(.15)*math.sin(phase)
    ry=np.array([[math.cos(yaw),0,math.sin(yaw)],[0,1,0],[-math.sin(yaw),0,math.cos(yaw)]],np.float32)
    rz=np.array([[math.cos(roll),-math.sin(roll),0],[math.sin(roll),math.cos(roll),0],[0,0,1]],np.float32)
    R=rz@ry
    xyz=np.stack([(u-cx)*z/fx,(v-cy)*z/fx,z],axis=0).reshape(3,-1)
    moved=R@xyz
    moved[0]+=.0035*math.sin(phase);moved[1]+=.0007*math.sin(2*phase)
    zz=moved[2].reshape(z.shape)
    zoom=1.035+.005*math.sin(2*phase)
    pu=(fx*moved[0].reshape(z.shape)/zz)*zoom+cx
    pv=(fx*moved[1].reshape(z.shape)/zz)*zoom+cy
    return pu,pv,zz,dict(yaw_degrees=math.degrees(yaw),roll_degrees=math.degrees(roll),translation_relative_units=[.0035*math.sin(phase),.0007*math.sin(2*phase),0],zoom=zoom,rotation_matrix=R.tolist())
def splat(rgb,pu,pv,z):
    h,w=z.shape
    u0=np.floor(pu).astype(np.int32);v0=np.floor(pv).astype(np.int32)
    fu=pu-u0;fv=pv-v0
    contributions=[]
    for dx,dy,weight in [(0,0,(1-fu)*(1-fv)),(1,0,fu*(1-fv)),(0,1,(1-fu)*fv),(1,1,fu*fv)]:
        xx=u0+dx;yy=v0+dy
        valid=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)&(weight>1e-6)&(z>0)
        idx=(yy[valid]*w+xx[valid]).ravel()
        contributions.append((idx,weight[valid].ravel(),z[valid].ravel(),rgb[valid].astype(np.float32)))
    nearest=np.full(h*w,np.inf,np.float32)
    for idx,wt,zz,col in contributions:np.minimum.at(nearest,idx,zz)
    sumw=np.zeros(h*w,np.float64);sumc=np.zeros((h*w,3),np.float64)
    for idx,wt,zz,col in contributions:
        # Soft z-buffer: reject samples more than 2% behind the closest surface.
        keep=zz<=nearest[idx]*1.02
        idx=idx[keep];wt=wt[keep];col=col[keep]
        sumw+=np.bincount(idx,weights=wt,minlength=h*w)
        for channel in range(3):sumc[:,channel]+=np.bincount(idx,weights=wt*col[:,channel],minlength=h*w)
    holes=(sumw<1e-6).reshape(h,w)
    fraction=float(holes.mean())
    assert fraction<=.005,f'Excess disocclusion: {fraction:.6%}; refusing distorted output'
    frame=np.clip(sumc/np.maximum(sumw[:,None],1e-6),0,255).astype(np.uint8).reshape(h,w,3)
    if holes.any():frame=cv2.inpaint(frame,holes.astype(np.uint8)*255,2,cv2.INPAINT_TELEA)
    return frame,fraction,int(holes.sum())
def main():
    global OUT
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--method-a-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--download-record', type=Path, required=True)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    assert not (OUT/'MD01_method_B_depth.mp4').exists() and not (OUT/'method_b_metrics.json').exists(),'Refusing overwrite'
    original={p:sha(p) for p in args.method_a_dir.resolve().iterdir() if p.is_file()}
    for p in [args.manifest.resolve(), args.download_record.resolve()]:
        original[p]=sha(p)
    index_sha=sha(ROOT/'.git/index')
    torch.set_num_threads(4);torch.manual_seed(42);cv2.setNumThreads(4)
    torch.use_deterministic_algorithms(True)
    monitor_stop=threading.Event();proc=psutil.Process();peak=[proc.memory_info().rss]
    def monitor():
        while not monitor_stop.wait(.05):peak[0]=max(peak[0],proc.memory_info().rss)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    started=time.perf_counter()
    manifest=args.manifest.resolve()
    row=next(r for r in csv.DictReader(manifest.open(encoding='utf-8',newline='')) if r['source_id']=='MD01')
    import zipfile
    with zipfile.ZipFile(args.archive.resolve()) as z:raw=z.read(row['archive_member'])
    assert hashlib.sha256(raw).hexdigest()==row['sha256']
    with Image.open(io.BytesIO(raw)) as p:image=ImageOps.exif_transpose(p).convert('RGB')
    assert image.size==(1280,550)
    rgb=np.array(image);bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    modelpath=args.model_dir.resolve()
    download=json.loads(args.download_record.resolve().read_text())
    for name,info in download['files'].items():assert sha(modelpath/name)==info['sha256']
    t=time.perf_counter()
    processor=DPTImageProcessor.from_pretrained(modelpath,local_files_only=True)
    # Aspect-preserving reduced depth-input size on CPU; all subsequent warps remain full resolution.
    processor.size={'height':224,'width':518}
    processor.keep_aspect_ratio=True
    model,loading=AutoModelForDepthEstimation.from_pretrained(modelpath,local_files_only=True,use_safetensors=True,output_loading_info=True)
    assert not any(loading[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),loading
    model.to('cpu').eval().requires_grad_(False)
    inputs=processor(images=image,return_tensors='pt')
    load_seconds=time.perf_counter()-t
    t=time.perf_counter()
    with torch.inference_mode():
        prediction=model(**inputs).predicted_depth
        raw_depth=torch.nn.functional.interpolate(prediction.unsqueeze(1),size=(550,1280),mode='bicubic',align_corners=False)[0,0].numpy()
    inference_seconds=time.perf_counter()-t
    assert np.isfinite(raw_depth).all() and float(raw_depth.std())>1e-5
    low,high=np.percentile(raw_depth,[2,98]);assert high>low
    disparity=np.clip((raw_depth-low)/(high-low),0,1).astype(np.float32)
    disparity=cv2.bilateralFilter(disparity,5,.05,3)
    # Relative inverse depth inferred by the network, bounded for conservative parallax; not metric geometry.
    z=1/(.8+.4*disparity)
    np.savez_compressed(OUT/'MD01_depth_raw.npz',predicted_relative_inverse_depth=raw_depth,normalized_smoothed_inverse_depth=disparity,relative_z=z)
    heat=cv2.cvtColor(cv2.applyColorMap(np.uint8(disparity*255),cv2.COLORMAP_INFERNO),cv2.COLOR_BGR2RGB)
    vis=Image.new('RGB',(1280,600),'white');vis.paste(Image.fromarray(heat),(0,50))
    draw=ImageDraw.Draw(vis);draw.text((15,8),'MD01 | Depth Anything V2 Small | brighter = larger relative inverse depth (nearer)',fill='black')
    draw.text((15,28),'Relative, scale-ambiguous estimate; NOT calibrated dental 3D geometry',fill='black')
    vis.save(OUT/'MD01_depth_visualization.png')
    del model,inputs,prediction
    u,v=np.meshgrid(np.arange(1280,dtype=np.float32),np.arange(550,dtype=np.float32))
    frames=[];stats=[];t=time.perf_counter()
    for i in range(16):
        phase=2*math.pi*i/15
        pu,pv,zz,pose=project(z,u,v,1280,639.5,274.5,phase)
        flat_u,flat_v,_,_=project(np.ones_like(z),u,v,1280,639.5,274.5,phase)
        parallax=np.hypot(pu-flat_u,pv-flat_v)
        du_dy,du_dx=np.gradient(pu);dv_dy,dv_dx=np.gradient(pv)
        det=du_dx*dv_dy-du_dy*dv_dx
        folds=float(np.mean(det<=0))
        assert folds<=.001,f'Folded geometry at frame {i+1}: {folds}'
        frame,hole_fraction,hole_pixels=splat(bgr,pu,pv,zz)
        frames.append(frame)
        stats.append(dict(frame_number=i+1,time_seconds=i/8,pose=pose,hole_fraction_before_inpainting=hole_fraction,hole_pixels_before_inpainting=hole_pixels,inpainted=bool(hole_pixels),foldover_fraction=folds,depth_vs_flat_parallax_p95_pixels=float(np.percentile(parallax,95)),max_depth_vs_flat_parallax_pixels=float(parallax.max())))
        print(f'Frame {i+1}/16: holes={hole_fraction:.6%}, folds={folds:.6%}',flush=True)
    render_seconds=time.perf_counter()-t
    assert max(a['max_depth_vs_flat_parallax_pixels'] for a in stats)>.1,'No actual depth dependence'
    video=OUT/'MD01_method_B_depth.mp4';t=time.perf_counter()
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),8,(1280,550));assert writer.isOpened()
    try:
        for f in frames:writer.write(f)
    finally:writer.release()
    encode_seconds=time.perf_counter()-t
    decoded=decode(video);a_frames=decode(args.method_a_dir.resolve()/'MD01_method_A_geometric.mp4')
    compression_mae=float(np.mean([np.abs(a.astype(np.float32)-b.astype(np.float32)).mean() for a,b in zip(frames,decoded)]))
    sheet=Image.new('RGB',(2560,722),'white');draw=ImageDraw.Draw(sheet)
    draw.text((10,8),'MD01 / LF 0464.jpg | True source label: Lower Front / Lower | A: planar homography | B: estimated-depth 2.5D reprojection',fill='black')
    draw.text((10,26),'All 16 corresponding frames; small motion is intentional. Neither method reconstructs unseen anatomy.',fill='black')
    for i,(af,bf) in enumerate(zip(a_frames,decoded)):
        base_x=(i%4)*640;base_y=50+(i//4)*168
        for j,(f,label) in enumerate([(af,'A geometric'),(bf,'B depth-based')]):
            draw.text((base_x+j*320+3,base_y),f'Frame {i+1} | {label}',fill='black')
            thumb=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)).resize((320,138),Image.Resampling.LANCZOS)
            sheet.paste(thumb,(base_x+j*320,base_y+20))
    sheet.save(OUT/'AB_contact_sheet.png')
    elapsed=time.perf_counter()-started
    monitor_stop.set();thread.join()
    memory=proc.memory_info();peak_rss=max(peak[0],getattr(memory,'peak_wset',memory.rss))
    config=dict(source=row,model_repo=download['repo'],model_revision=download['revision'],checkpoint_sha256=download['files']['model.safetensors']['sha256'],strict_loading_checked=loading,device='cpu',dtype='float32',depth_processor_input_shape='recorded in metrics',depth_processor_size_override={'height':224,'width':518},keep_aspect_ratio=True,depth_postprocessing='bicubic to source size; robust 2/98 percentile normalization; bilateral d=5 sigmaColor=.05 sigmaSpace=3',relative_geometry='Z=1/(0.8+0.4*normalized learned inverse depth); unknown true scale and camera calibration',focal_length_px=1280,principal_point_px=[639.5,274.5],frames=16,fps=8,width=1280,height=550,depth_fixed_across_frames=True,render='forward bilinear splatting with 2% relative z-buffer tolerance',inpainting='OpenCV Telea radius 2; only uncovered pixels; stop if >0.5% holes',foldover_stop_fraction=.001,poses=stats)
    save('method_b_config.json',config)
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    save('method_b_environment.json',dict(python=sys.version,executable=sys.executable,packages=packages))
    metrics=dict(status='PASS',source_sha256=row['sha256'],actual_model_inference=True,model_load_seconds=load_seconds,depth_inference_seconds=inference_seconds,depth_input_shape=list(processor(images=image,return_tensors='pt')['pixel_values'].shape),depth_std=float(raw_depth.std()),depth_percentiles_2_98=[float(low),float(high)],render_seconds=render_seconds,encode_seconds=encode_seconds,total_generation_verification_seconds=elapsed,peak_process_working_set_gib=peak_rss/2**30,gpu_memory_used_bytes=0,device='cpu',verified_frame_count=len(decoded),fps=8,width=1280,height=550,duration_seconds=2,max_hole_fraction=max(s['hole_fraction_before_inpainting'] for s in stats),max_foldover_fraction=max(s['foldover_fraction'] for s in stats),max_depth_vs_flat_parallax_pixels=max(s['max_depth_vs_flat_parallax_pixels'] for s in stats),compression_mae_0_255=compression_mae,all_existing_run_files_unchanged=all(sha(p)==h for p,h in original.items()),git_index_unchanged=sha(ROOT/'.git/index')==index_sha,script_sha256=sha(__file__),artifacts={p.name:sha(p) for p in [video,OUT/'MD01_depth_raw.npz',OUT/'MD01_depth_visualization.png',OUT/'AB_contact_sheet.png']})
    assert metrics['all_existing_run_files_unchanged'] and metrics['git_index_unchanged']
    assert len(list(OUT.glob('*.mp4')))==1  # Method A remains in its separate input directory.
    save('method_b_metrics.json',metrics)
    a_metrics=json.loads((args.method_a_dir.resolve()/'method_a_metrics.json').read_text())
    report=f"""# MD01 two-method pilot — numerical results
Only MD01 / LF 0464.jpg was used. Original label Lower Front / Lower is retained.
Method A is read from --method-a-dir; Method B is written to --output-dir.
Both fully decode to 16 RGB frames at 1280 x 550, 8 FPS, 2.0 seconds. No audio.

## Real depth and geometry
Depth Anything V2 Small, official HF revision {download['revision']}; checkpoint SHA-256 {download['files']['model.safetensors']['sha256']}. Checkpoint and all local model-file hashes verified; missing/unexpected/mismatched keys all empty. Actual CPU float32 network inference was performed once; no synthetic gradient substituted.
Depth tensor shape: {metrics['depth_input_shape']}; relative depth was upsampled to the source resolution. Reduced resolution is an explicit CPU-memory choice. Robust normalization and light bilateral smoothing preserve learned spatial structure, but small tooth boundaries can be inaccurate.
Monocular output is scale-ambiguous inverse depth, not calibrated millimeters. Z=1/(0.8+0.4*d) bounds relative surface depth; the focal length is assumed, not measured. A fixed depth surface is reused for all frames, avoiding frame-to-frame depth-network instability.
Camera motion: horizontal relative translation +/-0.0035, vertical +/-0.0007, yaw +/-0.08 degrees, roll +/-0.15 degrees, scale 1.030–1.040. A smooth loop returns to its starting pose. Real depth-dependent reprojection differs from a flat plane by up to {metrics['max_depth_vs_flat_parallax_pixels']:.3f} pixels. Perspective changes in A and camera parameters in B differ: this is a small illustrative pilot, not an equal-motion causal ablation.
Forward bilinear splatting uses a soft z-buffer. Maximum pre-fill holes: {metrics['max_hole_fraction']:.6%}; all such pixels, if any, use radius-2 Telea filling. Maximum numerical foldover fraction: {metrics['max_foldover_fraction']:.6%}. Hole filling does not recover genuinely hidden anatomy. Stop thresholds were 0.5% holes and 0.1% foldovers; no retries or alternate videos were generated.
Interpolation, splatting and MP4 compression can soften edges. Model-derived depth can mistake specular reflections/soft tissue for geometry. No clinical anatomy or view-label guarantee follows from coverage/foldover checks.

## Runtime and memory
| Stage | Seconds |
|---|---:|
| Model load + preprocessing | {load_seconds:.3f} |
| Actual depth-network inference | {inference_seconds:.3f} |
| 16-frame depth reprojection/render | {render_seconds:.3f} |
| MP4 encoding | {encode_seconds:.3f} |
| Total load/generate/verify/contact sheet | {elapsed:.3f} |
Method A's recorded CPU generation/verification time: {a_metrics['total_load_render_encode_verify_seconds']:.3f} s (different library versions and timing scope; not a controlled performance benchmark).
Peak process working set: {metrics['peak_process_working_set_gib']:.3f} GiB. CPU only, zero GPU allocations; four Torch/OpenCV CPU threads. Download/install time is excluded.
The original successful run used CPU Torch 2.5.1 and Transformers 4.46.3; this portable copy records the active environment. Existing VideoSwin .venv was not changed. An earlier nested .depth_env installation hit the Windows path-length limit and is preserved; the successful environment uses the shorter user-profile path without system-setting changes. Full package versions are in method_b_environment.json.

## Outputs
- MD01_method_B_depth.mp4: new Method B video.
- MD01_depth_visualization.png: brighter means nearer (larger estimated inverse depth), no metric scale.
- MD01_depth_raw.npz: raw estimate plus normalized/smoothed inverse depth and bounded relative Z.
- AB_contact_sheet.png: all 16 corresponding decoded A/B frames.
- method_b_config.json: camera poses, preprocessing, hole/foldover/parallax metrics.
- method_b_metrics.json and method_b_environment.json: numerical verification, hashes, versions and resources.
The Method A MP4/configuration/report and all existing run files were verified unchanged, as was the Git index. No further image was processed; no training, commit or push occurred.
This report records numerical findings. A separate visual-inspection note will document review of the saved depth map, contact sheet and full-resolution decoded frame samples.
"""
    with (OUT/'method_b_report.md').open('x',encoding='utf-8') as f:f.write(report)
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
