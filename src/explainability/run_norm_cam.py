"""Inference-only, post-norm linear-head CAM for three audited validation clips."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, hashlib, traceback, platform
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src' / 'data'))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.video import swin3d_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from dataset import DentalQuadrantDataset
OUT = Path(__file__).resolve().parent
AUDIT = OUT.parent
NAMES = ['Upper', 'Lower']
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def save_json(name, obj):
    with (OUT/name).open('x', encoding='utf-8') as f: json.dump(obj, f, indent=2, allow_nan=False)
def main():
    global OUT, AUDIT
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    AUDIT = args.audit_dir.resolve()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    assert not list(OUT.glob('*.png')) and not (OUT/'verification.json').exists(), 'Refusing artifact overwrite'
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    assert torch.cuda.is_available()
    evidence = json.loads((AUDIT/'audit_evidence.json').read_text())['models']['original81']
    checkpoint_path = args.checkpoint.resolve()
    assert sha(checkpoint_path) == evidence['checkpoint_sha256']
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert ck['epoch'] == 1
    config = ck['config']
    assert config['num_clips']==5 and config['frames_per_clip']==16 and config['frame_source']=='GT'
    dataset = DentalQuadrantDataset(ROOT/config['val_metadata'], data_root=ROOT/config['data_root'], split='val', frame_source='GT', num_clips=5, frames_per_clip=16, image_size=224)
    assert len(dataset)==13
    model = swin3d_t(weights=None)
    model.head = nn.Linear(model.head.in_features, 2)
    model.load_state_dict(ck['model_state_dict'], strict=True)
    model.requires_grad_(False).to('cuda').eval()
    del ck
    mean = torch.tensor(config['mean'],device='cuda').view(1,3,1,1,1)
    std = torch.tensor(config['std'],device='cuda').view(1,3,1,1,1)
    reference = pd.read_csv(AUDIT/'per_clip_predictions.csv')
    captured = {}
    handle = model.norm.register_forward_hook(lambda m,i,o: captured.update(activation=o.detach()))
    cases, checks = [], []
    # All prediction/reconstruction gates must pass before saving any map or image.
    with torch.inference_mode():
        for sid, start in [(1,1),(7,44),(2,47)]:
            idxs = np.flatnonzero(dataset.df.segment_id.to_numpy()==sid)
            assert len(idxs)==1
            sample = dataset[int(idxs[0])]
            frames = sample['frame_indices'][0].tolist()
            assert frames==list(range(start,start+16))
            ref = reference[(reference.model=='original81') & (reference.segment_id==sid) & (reference.clip_number==1)]
            assert len(ref)==1
            ref = ref.iloc[0]
            assert ref.checkpoint_sha256==evidence['checkpoint_sha256']
            assert frames == [int(n) for n in str(ref.frame_indices).split(';')]
            assert sample['uuid']==ref.source_uuid and sample['label_name']==ref.ground_truth_quadrant
            rgb = sample['clips'][0]
            x = (rgb.to('cuda').permute(1,0,2,3).unsqueeze(0)-mean)/std
            logits = model(x)
            probs = logits.softmax(1)[0]
            a = captured.pop('activation')
            assert tuple(a.shape)==(1,8,7,7,768), tuple(a.shape)
            # BTHWC activations -> KTHW signed evidence; no channel-first assumption.
            signed = torch.einsum('bthwc,kc->bkthw',a,model.head.weight)[0]
            reconstructed = signed.mean((1,2,3)) + model.head.bias
            delta = (reconstructed-logits[0]).abs()
            pdelta = max(abs(probs[0].item()-float(ref.p_upper)),abs(probs[1].item()-float(ref.p_lower)))
            assert pdelta<=1e-6, f'Segment {sid}: probability discrepancy {pdelta}'
            assert NAMES[int(probs.argmax())]==ref.predicted_binary
            assert torch.allclose(reconstructed,logits[0],atol=1e-5,rtol=0), f'Logit reconstruction failed: {delta}'
            positive = signed.clamp_min(0)
            # Preserve signed evidence and positive CAM separately; upsample only for display.
            up_signed = F.interpolate(signed[None],size=(16,224,224),mode='trilinear',align_corners=False)[0].cpu().numpy()
            up_positive = F.interpolate(positive[None],size=(16,224,224),mode='trilinear',align_corners=False)[0].cpu().numpy()
            checks.append(dict(segment_id=sid, clip_number=1, quadrant=sample['label_name'], uuid=sample['uuid'], frames=frames, probabilities=probs.cpu().tolist(), reference_probabilities=[float(ref.p_upper),float(ref.p_lower)], max_probability_error=pdelta, logits=logits[0].cpu().tolist(), reconstructed_logits=reconstructed.cpu().tolist(), absolute_logit_errors=delta.cpu().tolist(), predicted=NAMES[int(probs.argmax())], segment_predicted=str(ref.segment_predicted_binary)))
            cases.append(dict(sid=sid,rgb=rgb.permute(0,2,3,1).numpy(),signed=signed.cpu().numpy(),positive=positive.cpu().numpy(),up_signed=up_signed,up_positive=up_positive,check=checks[-1]))
    handle.remove()
    max_signed=max(float(np.abs(c['signed']).max()) for c in cases)
    max_positive=max(float(c['positive'].max()) for c in cases)
    signed_norm=TwoSlopeNorm(vmin=-max_signed,vcenter=0,vmax=max_signed)
    pos_norm=Normalize(vmin=0,vmax=max_positive)
    for c in cases:
        sid=c['sid']; info=c['check']
        np.savez_compressed(OUT/f'segment_{sid:02d}_clip_1_signed_native.npz',signed=c['signed'],frame_indices=info['frames'],class_names=NAMES)
        np.savez_compressed(OUT/f'segment_{sid:02d}_clip_1_positive_native.npz',positive=c['positive'],frame_indices=info['frames'],class_names=NAMES)
        # Four chronological columns x four time blocks; original/Upper/Lower rows in each block.
        for kind,norm,cmap in [('signed',signed_norm,'RdBu_r'),('positive',pos_norm,'inferno')]:
            fig,axes=plt.subplots(12,4,figsize=(13,30),layout='constrained')
            for frame in range(16):
                block,col=divmod(frame,4)
                for typ in range(3):
                    ax=axes[block*3+typ,col]
                    ax.imshow(c['rgb'][frame])
                    if typ:
                        ax.imshow(c['up_'+kind][typ-1,frame],cmap=cmap,norm=norm,alpha=.5)
                    ax.set_title(f"Frame {info['frames'][frame]} | {['Original','Upper','Lower'][typ]}",fontsize=10)
                    ax.axis('off')
            fig.suptitle(f"Segment {sid}, clip 1 | True {info['quadrant']} / Lower | Predicted {info['predicted']}\nP(Upper)={info['probabilities'][0]:.6f}, P(Lower)={info['probabilities'][1]:.6f} | {kind} class evidence\nNative 8 x 7 x 7; interpolated overlays are NOT pixel-level localization",fontsize=13)
            fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes.ravel().tolist(),shrink=.35,label='Class-logit contribution per token (shared scale across cases/classes)')
            fig.savefig(OUT/f'segment_{sid:02d}_clip_1_{kind}_all_frames.png',dpi=110)
            plt.close(fig)
    verification=dict(checkpoint=str(checkpoint_path),checkpoint_sha256=evidence['checkpoint_sha256'],epoch=1,layer='model.norm',native_shape=[8,7,7],formula='E_k(t,h,w) = sum_c weight[k,c] * A[t,h,w,c]; logit_k = mean(E_k) + bias[k]',probability_tolerance=1e-6,logit_absolute_tolerance=1e-5,cases=checks,color_scales=dict(signed=[-max_signed,max_signed],positive=[0,max_positive]),environment=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,gpu=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated()),input_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),AUDIT/'per_clip_predictions.csv',AUDIT/'audit_evidence.json',ROOT/'src/data/dataset.py',ROOT/config['val_metadata']]},preprocessing=config, test_accessed=False,backward_used=False)
    save_json('verification.json',verification)
    lines=['# Original-81 epoch-1 class-evidence analysis','', 'All three clip predictions match the existing audit within 1e-6. Both class logits reconstruct from signed token means plus classifier bias within absolute 1e-5. All gates passed before any visualizations were generated.','', '| Segment | True quadrant | Clip prediction | P(Lower) | Probability error | Max logit error |','|---|---|---|---:|---:|---:|']
    for c in cases:
        a=c['check']; lines.append(f"| {a['segment_id']} | {a['quadrant']} | {a['predicted']} | {a['probabilities'][1]:.8f} | {a['max_probability_error']:.3g} | {max(a['absolute_logit_errors']):.3g} |")
    lines += ['', '## Method and interpretation','', 'The frozen model runs in eval mode, float32, one clip at a time under inference_mode. The exact dataset loader supplies the five evaluation clips; only clip 1 of the three selected validation segments is inferred. No test frames are read. A forward hook captures channels-last [1,8,7,7,768] model.norm activations. Each class uses its own linear classifier weights, including negative evidence. Positive CAM is ReLU of the native signed evidence, before interpolation. This is an exact linear-head evidence decomposition, not a causal explanation or attention map.', '', 'Segment 1 favors Upper incorrectly. Segment 7 favors Lower correctly. Segment 2 has only a small Lower clip-level advantage; its overall five-clip segment decision remains Upper. These three clips alone cannot explain that five-clip decision. Spatial patterns must be interpreted from the figures, without claiming anatomical causality.', '', 'Native attribution resolution is **8 x 7 x 7** (time x height x width). Trilinear upsampling to 16 x 224 x 224 is for display only and does not provide pixel-level localization. Temporally and spatially mixed transformer tokens cannot be assigned to precise teeth or individual input pixels. RGB panels are the exact resized model-input frames before normalization. Overlays have fixed alpha 0.5; signed diverging and positive scales are shared across all cases and both classes, with no per-frame normalization. Red indicates positive signed evidence; blue negative evidence. Bias is not spatially distributed. Positive maps omit opposing evidence and do not reconstruct logits.', '', '## Reproduction and outputs','', 'Run from the repository root using `.\\.venv\\Scripts\\python.exe -B outputs/runs/prediction_agreement_audit_2026-10-15_01/explainability_original81_epoch01_norm_cam/run_norm_cam.py`. The script refuses reruns into populated artifacts; preserve this folder and use a separately approved destination for any rerun. No dependency installation, gradients, retraining, or checkpoint changes are required.', '', '`verification.json` contains full numerical checks, hashes, versions, preprocessing and shared scales. `segment_XX_clip_1_signed_native.npz` and `segment_XX_clip_1_positive_native.npz` preserve native maps, frame indices and class order. Each segment has signed and positive `*_all_frames.png` figures showing all 16 frames with original/Upper/Lower rows. `artifact_sha256.json` records output checksums.']
    with (OUT/'report.md').open('x',encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')
    save_json('artifact_sha256.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
    print(json.dumps({'status':'PASS','cases':checks,'output':str(OUT)},indent=2))
if __name__=='__main__':
    try: main()
    except Exception:
        message=traceback.format_exc()
        print(message,file=sys.stderr)
        if not (OUT/'failure.txt').exists():
            with (OUT/'failure.txt').open('x',encoding='utf-8') as f: f.write(message)
        raise