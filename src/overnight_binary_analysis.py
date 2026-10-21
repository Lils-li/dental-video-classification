"""Fixed overnight postprocessing; validation only, no adaptive research choices."""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['Upper', 'Lower']
MAP = {'Q1':'Upper', 'Q2':'Upper', 'Q3':'Lower', 'Q4':'Lower'}
POLICY = {
    'exploratory': True, 'models': ['original', 'cleaned'],
    'selection': 'lowest segment_id correct and lowest segment_id incorrect per model; absent strata omitted',
    'selection_unit': 'five-clip segment prediction at best validation Macro-F1 epoch',
    'clip_index': 3, 'frames': 16, 'image_size': 224,
    'regions': {'top_left':[0,112,0,112], 'top_right':[0,112,112,224], 'bottom_left':[112,224,0,112], 'bottom_right':[112,224,112,224]},
    'mask_fill': 'RGB training mean; zero after normalization; same region in every frame',
    'minimum_free_gpu_gib': 1.5, 'budget_minutes': 10, 'optional_window_hours': 8,
    'max_selected_clips': 4, 'max_forward_passes': 20,
    'test_used': False,
}


def save_json(path, obj):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def csv_out(path, rows):
    if not rows: raise ValueError('Cannot write empty table without a schema.')
    with Path(path).open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def inputs(args):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,recall_score
    validation=pd.read_csv(ROOT/'data/labels/quadrant_metadata.csv')
    validation=validation[validation.split=='val'].copy()
    if len(validation)!=13 or validation.segment_id.duplicated().any():raise ValueError('Invalid validation manifest.')
    arms={}
    for arm,path in [('original',args.original),('cleaned',args.cleaned)]:
        path=path.resolve()
        complete=read_json(path/'completed.json')
        if complete!=dict(smoke_only=False,epochs=15,test_used=False):raise ValueError('Incomplete full run: '+str(path))
        config=read_json(path/'config.json');provenance=read_json(path/'provenance.json')
        if digest(ROOT/'data/labels/quadrant_metadata.csv')!=provenance['input_sha256']['data/labels/quadrant_metadata.csv']:raise ValueError('Validation metadata changed since training.')
        best=read_json(path/'best_checkpoint.json')
        allmetrics=read_json(path/'metrics.json')
        if [x['epoch'] for x in allmetrics]!=list(range(1,16)):raise ValueError('Epoch metrics incomplete.')
        expected_best=max(allmetrics,key=lambda x:x['val']['macro_f1'])['epoch']
        if best['smoke_only'] or best['epoch']!=expected_best:raise ValueError('Invalid best-epoch selection.')
        if Path(best['path']).name!=best['path']:raise ValueError('Unsafe checkpoint filename.')
        tables={};metrics={}
        for scope,epoch in [('best',best['epoch']),('final',15)]:
            table=pd.read_csv(path/f'epoch_{epoch:02d}_val_predictions.csv')
            if len(table)!=13 or table.segment_id.duplicated().any():raise ValueError('Invalid prediction row count.')
            joined=table.merge(validation[['segment_id','label']],on='segment_id',validate='one_to_one',how='outer',indicator=True)
            if not (joined['_merge']=='both').all() or not (joined.original_label==joined.label).all():raise ValueError('Prediction labels do not match original validation annotations.')
            if not (joined.true_binary==joined.label.map(MAP)).all():raise ValueError('Binary labels inconsistent.')
            truth=joined.true_binary.map({'Upper':0,'Lower':1});pred=joined.predicted_binary.map({'Upper':0,'Lower':1})
            probs=joined[['p_upper','p_lower']].to_numpy()
            if pred.isna().any() or not np.isfinite(probs).all() or (probs<0).any() or (probs>1).any() or not np.allclose(probs.sum(1),1,atol=1e-5):raise ValueError('Invalid probabilities.')
            if not np.array_equal(probs.argmax(1),pred.to_numpy()):raise ValueError('Predictions disagree with probabilities.')
            joined['is_correct']=joined.true_binary==joined.predicted_binary
            metric={'accuracy':float(accuracy_score(truth,pred)), 'balanced_accuracy':float(balanced_accuracy_score(truth,pred)), 'macro_f1':float(f1_score(truth,pred,labels=[0,1],average='macro',zero_division=0)), 'recall':recall_score(truth,pred,labels=[0,1],average=None,zero_division=0).tolist(), 'confusion_matrix':confusion_matrix(truth,pred,labels=[0,1]).tolist()}
            saved=allmetrics[epoch-1]['val']
            for k in ['accuracy','balanced_accuracy','macro_f1']:
                if not np.isclose(metric[k],saved[k],atol=1e-10):raise ValueError('Saved metrics disagree with predictions.')
            if metric['confusion_matrix']!=saved['confusion_matrix']:raise ValueError('Confusion matrix mismatch.')
            tables[scope]=joined.sort_values('segment_id');metrics[scope]=metric|{'epoch':epoch}
        arms[arm]={'path':path,'config':config,'provenance':provenance,'best':best,'tables':tables,'metrics':metrics}
    for k in ['initial_head_sha256','pretrained_sha256']:
        if arms['original']['provenance'][k]!=arms['cleaned']['provenance'][k]:raise ValueError('Initialization mismatch.')
    allowed={'arm','train_metadata','expected_train_segments'}
    a,b=arms['original']['config'],arms['cleaned']['config']
    if set(a)!=set(b) or any(a[k]!=b[k] for k in a if k not in allowed):raise ValueError('Controlled configurations differ unexpectedly.')
    return arms


def analyze(args, out):
    import pandas as pd
    arms=inputs(args);rows=[];result={};plt=figures()
    for arm,data in arms.items():
        result[arm]={'run':str(data['path']),'metrics':data['metrics']}
        for scope,table in data['tables'].items():
            for quadrant in ['Q1','Q2','Q3','Q4']:
                subset=table[table.label==quadrant];correct=int(subset.is_correct.sum())
                rows.append({'arm':arm,'selection':scope,'epoch':data['metrics'][scope]['epoch'],'quadrant':quadrant,'binary_truth':MAP[quadrant],'count':len(subset),'correct':correct,'errors':len(subset)-correct,'accuracy':correct/len(subset),'error_segment_ids':';'.join(map(str,subset.loc[~subset.is_correct,'segment_id'].tolist()))})
            table[['segment_id','label','true_binary','predicted_binary','p_upper','p_lower','is_correct']].to_csv(out/f'{arm}_{scope}_validation.csv',index=False)
            matrix=data['metrics'][scope]['confusion_matrix']
            pd.DataFrame(matrix,index=NAMES,columns=NAMES).rename_axis('true_label').to_csv(out/f'{arm}_{scope}_confusion_matrix.csv')
            fig,ax=plt.subplots(figsize=(4,4));ax.imshow(matrix,cmap='Blues',vmin=0)
            for i in range(2):
                for j in range(2):ax.text(j,i,str(matrix[i][j]),ha='center',va='center')
            ax.set(xticks=[0,1],yticks=[0,1],xticklabels=NAMES,yticklabels=NAMES,xlabel='Predicted',ylabel='True',title=f'{arm} / {scope} / epoch {data["metrics"][scope]["epoch"]}')
            fig.tight_layout();fig.savefig(out/f'{arm}_{scope}_confusion_matrix.png',dpi=160);plt.close(fig)
    csv_out(out/'per_quadrant.csv',rows)
    save_json(out/'summary.json',{'arms':result,'test_used':False,'interpretation':'Binary classification accuracy grouped by original quadrant; not four-class accuracy.'})
    print('Per-quadrant analysis saved:',out,flush=True)


def occlude(args, out):
    save_json(out/'policy.json',POLICY)
    if args.remaining_minutes<POLICY['budget_minutes']:
        save_json(out/'status.json',{'status':'skipped','reason':'Fewer than 10 minutes remain in the predefined optional-stage window.','test_used':False});return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback.')
    free,total=torch.cuda.mem_get_info()
    if free<1.5*2**30:
        save_json(out/'status.json',{'status':'skipped','reason':'Less than 1.5 GiB free GPU memory.','free_bytes':free,'test_used':False});return
    torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    import pandas as pd
    from torch import nn
    from torchvision.models.video import swin3d_t
    sys.path.insert(0,str(ROOT/'src/data'))
    from dataset import DentalQuadrantDataset
    arms=inputs(args);plt=figures();rows=[];selected=[];start=time.monotonic()
    dataset=DentalQuadrantDataset(ROOT/'data/labels/quadrant_metadata.csv',data_root=ROOT/'data/raw/Vident-real',split='val',frame_source='GT')
    assert set(dataset.df.split)=={'val'}
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1,1)
    std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1,1)
    for arm,data in arms.items():
        selection=[]
        for correct in [True,False]:
            candidates=data['tables']['best'][data['tables']['best'].is_correct==correct]
            if len(candidates):selection.append((int(candidates.iloc[0].segment_id),correct))
        if not selection:continue
        checkpoint_path=data['path']/data['best']['path']
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        model=swin3d_t(weights=None);model.head=nn.Linear(model.head.in_features,2)
        model.load_state_dict(checkpoint['model_state_dict']);model.to('cuda').eval()
        del checkpoint
        for segment_id,correct in selection:
            if time.monotonic()-start>600:raise TimeoutError('Occlusion exceeded its predefined 10-minute budget; no parameter fallback.')
            index=dataset.df.index[dataset.df.segment_id==segment_id].tolist()
            if len(index)!=1:raise ValueError('Selected validation ID is not unique.')
            sample=dataset[index[0]];clip=sample['clips'][3]
            raw=clip.to('cuda').permute(1,0,2,3).unsqueeze(0)
            target=0 if MAP[sample['label_name']]=='Upper' else 1
            with torch.no_grad():
                baseline=model((raw-mean)/std).softmax(1)[0].cpu().numpy()
                if not torch.isfinite(torch.tensor(baseline)).all():raise ValueError('Nonfinite baseline.')
                fig,axes=plt.subplots(1,5,figsize=(15,3));middle=7
                axes[0].imshow(clip[middle].permute(1,2,0).numpy());axes[0].set_title(f'Unmasked\nP(true)={baseline[target]:.3f}')
                for panel,(region,(y0,y1,x0,x1)) in enumerate(POLICY['regions'].items(),1):
                    if time.monotonic()-start>600:raise TimeoutError('Occlusion time budget exceeded.')
                    masked=raw.clone();masked[:,:,:,y0:y1,x0:x1]=mean
                    probs=model((masked-mean)/std).softmax(1)[0].cpu().numpy()
                    if not torch.isfinite(torch.tensor(probs)).all():raise ValueError('Nonfinite occlusion prediction.')
                    rows.append({'arm':arm,'epoch':data['best']['epoch'],'segment_id':segment_id,'quadrant':sample['label_name'],'segment_was_correct':correct,'clip_number':4,'region':region,'unmasked_clip_prediction':NAMES[int(baseline.argmax())],'masked_clip_prediction':NAMES[int(probs.argmax())],'baseline_p_upper':float(baseline[0]),'baseline_p_lower':float(baseline[1]),'masked_p_upper':float(probs[0]),'masked_p_lower':float(probs[1]),'delta_p_upper':float(probs[0]-baseline[0]),'delta_p_lower':float(probs[1]-baseline[1]),'delta_p_true':float(probs[target]-baseline[target])})
                    axes[panel].imshow(masked[0,:,middle].permute(1,2,0).cpu().numpy());axes[panel].set_title(f'{region}\nDelta P(true)={probs[target]-baseline[target]:+.3f}')
                for ax in axes:ax.axis('off')
                fig.suptitle(f'Exploratory occlusion: {arm}, segment {segment_id}, {sample["label_name"]}; selection based on segment, probabilities based on clip 4')
                fig.tight_layout();fig.savefig(out/f'{arm}_segment_{segment_id}_clip4.png',dpi=150);plt.close(fig)
            selected.append({'arm':arm,'segment_id':segment_id,'segment_was_correct':correct,'checkpoint_sha256':digest(checkpoint_path)})
            print('Exploratory occlusion completed:',arm,segment_id,flush=True)
        del model
        torch.cuda.empty_cache()
    if rows:csv_out(out/'probability_changes.csv',rows)
    save_json(out/'status.json',{'status':'completed','selected':selected,'mask_evaluations':len(rows),'seconds':time.monotonic()-start,'exploratory':True,'test_used':False,'omitted_strata':'A missing correct or incorrect stratum is omitted without replacement.'})


def report(args, out):
    arms=inputs(args);analysis=read_json(args.analysis/'summary.json');occlusion=read_json(args.occlusion/'status.json')
    import pandas as pd
    quadrants=pd.read_csv(args.analysis/'per_quadrant.csv')
    lines=['# Binary VideoSwin: overnight supervisor report','','Prepared from completed runs; no held-out test data were used.','', '## Actual validation results','', '| Training data | Selection | Epoch | Accuracy | Balanced accuracy | Macro-F1 | Lower recall |','|---|---|---:|---:|---:|---:|---:|']
    for arm,data in arms.items():
        for scope,metric in data['metrics'].items():
            lines.append(f'| {arm} | {scope} | {metric["epoch"]} | {metric["accuracy"]:.2%} | {metric["balanced_accuracy"]:.2%} | {metric["macro_f1"]:.4f} | {metric["recall"][1]:.2%} |')
    lines+=['','Confusion matrices: rows=true, columns=predicted, order Upper/Lower.','']
    for arm,data in arms.items():
        for scope,metric in data['metrics'].items():
            lines += [f'**{arm}, {scope}:**','',f'```text\n{metric["confusion_matrix"][0]}\n{metric["confusion_matrix"][1]}\n```','',f'![{arm} {scope} confusion matrix](../analysis/{arm}_{scope}_confusion_matrix.png)','']
    delta=arms['cleaned']['metrics']['best']['accuracy']-arms['original']['metrics']['best']['accuracy']
    lines += [f'The observed best-epoch cleaned-minus-original accuracy difference is {delta*100:+.2f} percentage points. This is descriptive evidence from one seed, not a superiority claim.','','## Per-quadrant binary error analysis','','| Run | Selection | Quadrant | Count | Correct | Errors | Accuracy |','|---|---|---|---:|---:|---:|---:|']
    for row in quadrants.itertuples():lines.append(f'| {row.arm} | {row.selection} | {row.quadrant} | {row.count} | {row.correct} | {row.errors} | {row.accuracy:.2%} |')
    lines+=['','These are Upper/Lower predictions grouped by the original Q1-Q4 label, not four-class quadrant predictions. Detailed error IDs are in `../analysis/per_quadrant.csv`.','','## Exploratory occlusion','',f'Status: {occlusion["status"]}.']
    if occlusion['status']=='skipped':lines.append('Reason: '+occlusion['reason'])
    else:
        lines += [f'{len(occlusion["selected"])} selected clips; {occlusion["mask_evaluations"]} masked evaluations.','Selection was predefined: lowest-ID correct and incorrect segment per model, where available; clip 4; four 112x112 regions; mean RGB fill across all sixteen frames. Correctness refers to the five-clip segment decision; probabilities refer only to clip 4.']
        changes=args.occlusion/'probability_changes.csv'
        if changes.exists():
            table=pd.read_csv(changes)
            lines += ['','| Run | Segment | Region | Change in true-class probability |','|---|---:|---|---:|']
            for r in table.itertuples():lines.append(f'| {r.arm} | {r.segment_id} | {r.region} | {r.delta_p_true:+.4f} |')
        for item in occlusion['selected']:lines+=['',f'![Exploratory occlusion](../occlusion/{item["arm"]}_segment_{item["segment_id"]}_clip4.png)']
    lines+=['','Occlusion is exploratory, selection-biased, and uses artificial out-of-distribution masks. Probability changes do not establish anatomical causality or reliable explanations.','','## Controls and limitations','','- Existing configurations retained: seed 42, 15 fixed epochs, Swin3D-T KINETICS400_V1, frozen backbone in training mode during training, original RGB GT preprocessing, AdamW at 0.001 with weight decay 0.0001, unweighted cross-entropy, balanced sampling, and unchanged 13-segment validation. Initialization hashes match.','- Original training has 63 Upper/18 Lower segments; cleaned training has 46 Upper/7 Lower. Balancing creates 126 versus 92 updates per epoch (1,890 versus 1,380 total). Equal epochs do not mean equal optimization exposure.','- One seed, 13 validation segments, and best-of-15 selection limit inference. One case changes accuracy by 7.69 percentage points. Repeated validation use risks selection optimism; patient-level independence is unknown.','- Training annotation boundary overlaps and legacy square resizing are preserved. The small occlusion sample cannot validate robustness.','- The historical weighted-loss branch has a confirmed singleton mean-reduction weighting defect; neither current balanced arm uses that branch. These runs do not test its causal role in prior collapse.','','## Recommended next experiments (not executed)','','1. Repeat this fixed comparison with additional prespecified seeds; compare paired validation outcomes and uncertainty.','2. Add an equal-update-budget comparison to distinguish cleaning from reduced optimization exposure.','3. Test correctly implemented class weighting as a separate controlled experiment.','4. Review patient grouping and ambiguous annotations; broaden occlusion only with a predefined validation protocol. Keep the held-out test set reserved for a later approved final evaluation.','','## Provenance and artifacts','']
    for arm,data in arms.items():lines+= [f'- {arm} run: `{data["path"]}`; pretrained SHA-256 `{data["provenance"]["pretrained_sha256"]}`.']
    lines += ['- Configs, input hashes, sampling orders, epoch metrics/predictions, and checkpoints are stored in each run; stage logs and launch hashes are in `../logs/`.','- Analysis and clinical-image figures remain local; review sharing permissions before distribution.']
    with (out/'supervisor_report.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    save_json(out/'results.json',{'analysis':analysis,'occlusion':occlusion,'test_used':False})
    print('Supervisor report:',out/'supervisor_report.md',flush=True)


def preflight(args, out):
    import torch
    import matplotlib
    import pandas
    import numpy
    import PIL
    from torchvision.models.video import Swin3D_T_Weights
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback.')
    cache=Path(torch.hub.get_dir())/'checkpoints'/Swin3D_T_Weights.KINETICS400_V1.url.rsplit('/',1)[-1]
    if not cache.is_file():raise FileNotFoundError('Cached pretrained weights missing; no download permitted.')
    if not digest(cache).startswith('7615ae03'):raise ValueError('Pretrained cache hash mismatch.')
    save_json(out/'environment.json',{'interpreter':sys.executable,'gpu':torch.cuda.get_device_name(0),'cached_weights':str(cache),'test_used':False})
    print('Environment and cached weights verified; no downloads or training.',flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['preflight','analyze','occlude','report'])
    parser.add_argument('--original',type=Path,required=True);parser.add_argument('--cleaned',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--analysis',type=Path);parser.add_argument('--occlusion',type=Path)
    parser.add_argument('--remaining-minutes',type=float,default=480)
    args=parser.parse_args()
    if Path(sys.executable).resolve()!=(ROOT/'.venv/Scripts/python.exe').resolve():raise RuntimeError('Use repository .venv interpreter explicitly.')
    for path in [args.output,args.original,args.cleaned]:
        if not path.resolve().is_relative_to((ROOT/'outputs/runs').resolve()):raise ValueError('Paths must remain under outputs/runs.')
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        {'preflight':preflight,'analyze':analyze,'occlude':occlude,'report':report}[args.stage](args,args.output)
    except Exception as error:
        save_json(args.output/'failure.json',{'type':type(error).__name__,'message':str(error)});raise


if __name__=='__main__':main()
