"""Evaluate preserved four-class Swin as binary; compare only completed approved runs."""
import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from overnight_binary_analysis import save_json,read_json,digest,csv_out,figures,inputs
Q=['Q1','Q2','Q3','Q4'];B=['Upper','Lower']


def evaluate(args,out):
    import torch
    import numpy as np
    import pandas as pd
    from torch import nn
    from torchvision.models.video import swin3d_t,Swin3D_T_Weights
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,recall_score
    from data.dataset import DentalQuadrantDataset
    checkpoint_path=ROOT/'outputs/checkpoints/videoswin_gt_head_only_best.pt'
    before=digest(checkpoint_path);meta=ROOT/'data/labels/quadrant_metadata.csv';meta_hash=digest(meta)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if checkpoint['class_names']!=Q or checkpoint['frame_source']!='GT' or checkpoint['num_clips']!=5 or checkpoint['frames_per_clip']!=16:raise ValueError('Checkpoint metadata incompatible with original protocol.')
    state=checkpoint['model_state_dict']
    if tuple(state['head.weight'].shape)!=(4,768) or tuple(state['head.bias'].shape)!=(4,):raise ValueError('Incompatible four-class head.')
    torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback.')
    model=swin3d_t(weights=None);model.head=nn.Linear(model.head.in_features,4)
    model.load_state_dict(state,strict=True);model.to('cuda').eval()
    for p in model.parameters():p.requires_grad=False
    weights=Swin3D_T_Weights.KINETICS400_V1
    cache=Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/',1)[-1]
    backbone_matches=None
    if cache.is_file():
        initial=torch.load(cache,map_location='cpu',weights_only=True)
        backbone_matches=all(k in initial and torch.equal(v,initial[k]) for k,v in state.items() if not k.startswith('head.'))
        del initial
    dataset=DentalQuadrantDataset(meta,data_root=ROOT/'data/raw/Vident-real',split='val',frame_source='GT',num_clips=5,frames_per_clip=16,image_size=224)
    assert len(dataset)==13 and set(dataset.df.split)=={'val'}
    dataset.df.to_csv(out/'validation_manifest.csv',index=False)
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1,1)
    std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1,1)
    rows=[];clip_rows=[];timings=[]
    with torch.no_grad():
        for index in range(len(dataset)):
            torch.cuda.synchronize();start=time.perf_counter();sample=dataset[index]
            assert sample['clips'].shape==(5,16,3,224,224)
            probabilities=[]
            for i,clip in enumerate(sample['clips']):
                x=(clip.to('cuda').permute(1,0,2,3).unsqueeze(0)-mean)/std
                prob=model(x).softmax(1)[0].cpu()
                assert torch.isfinite(prob).all()
                probabilities.append(prob)
                clip_rows.append({'segment_id':sample['segment_id'],'clip_number':i+1,**{f'p_{q}':float(prob[j]) for j,q in enumerate(Q)}})
            p=torch.stack(probabilities).mean(0);pred=int(p.argmax());true=int(sample['label'])
            # Requested decision: map quadrant argmax AFTER original probability aggregation.
            true_binary=0 if true<2 else 1;pred_binary=0 if pred<2 else 1
            rows.append({'segment_id':sample['segment_id'],'original_label':Q[true],'predicted_quadrant':Q[pred],'true_id':true_binary,'predicted_id':pred_binary,'true_binary':B[true_binary],'predicted_binary':B[pred_binary],'correct_binary':true_binary==pred_binary,'correct_quadrant':true==pred,**{f'p_{q}':float(p[j]) for j,q in enumerate(Q)}})
            torch.cuda.synchronize();timings.append(time.perf_counter()-start)
            print(f'validation_segment={sample["segment_id"]} true={Q[true]} predicted={Q[pred]} mapped={B[pred_binary]} seconds={timings[-1]:.3f}',flush=True)
    if digest(checkpoint_path)!=before or digest(meta)!=meta_hash:raise RuntimeError('Checkpoint or metadata changed during evaluation.')
    truth=[r['true_id'] for r in rows];pred=[r['predicted_id'] for r in rows]
    four_f1=float(f1_score([Q.index(r['original_label']) for r in rows],[Q.index(r['predicted_quadrant']) for r in rows],labels=[0,1,2,3],average='macro',zero_division=0))
    if abs(four_f1-checkpoint['val_macro_f1'])>1e-10:raise ValueError('Original four-class validation Macro-F1 was not reproduced.')
    metrics={'segments':13,'accuracy':float(accuracy_score(truth,pred)),'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),'macro_f1':float(f1_score(truth,pred,labels=[0,1],average='macro',zero_division=0)),'confusion_matrix':confusion_matrix(truth,pred,labels=[0,1]).tolist(),'recall':recall_score(truth,pred,labels=[0,1],average=None,zero_division=0).tolist(),'class_order':B,'matrix_rows':'true','matrix_columns':'predicted','four_class_macro_f1':four_f1,'checkpoint_epoch':checkpoint['epoch'],'decision_rule':'mean five four-class probability vectors -> quadrant argmax -> binary mapping','test_used':False}
    csv_out(out/'predictions.csv',rows);csv_out(out/'clip_probabilities.csv',clip_rows)
    quadrant_rows=[]
    for q in Q:
        group=[r for r in rows if r['original_label']==q];correct=sum(r['correct_binary'] for r in group)
        quadrant_rows.append({'quadrant':q,'count':len(group),'binary_correct':correct,'binary_errors':len(group)-correct,'binary_accuracy':correct/len(group),'error_segment_ids':';'.join(str(r['segment_id']) for r in group if not r['correct_binary'])})
    csv_out(out/'per_quadrant.csv',quadrant_rows)
    pd.DataFrame(metrics['confusion_matrix'],index=B,columns=B).rename_axis('true_label').to_csv(out/'confusion_matrix.csv')
    plt=figures();fig,ax=plt.subplots(figsize=(4,4));ax.imshow(metrics['confusion_matrix'],cmap='Blues',vmin=0)
    for i in range(2):
        for j in range(2):ax.text(j,i,str(metrics['confusion_matrix'][i][j]),ha='center',va='center')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=B,yticklabels=B,xlabel='Predicted',ylabel='True',title='Four-class VideoSwin mapped to binary');fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=160);plt.close(fig)
    groups=checkpoint['optimizer_state_dict']['param_groups']
    provenance={'checkpoint':str(checkpoint_path),'checkpoint_sha256':before,'metadata_sha256':meta_hash,'strict_state_dict_load':True,'class_names':Q,'head_shape':[4,768],'backbone_matches_cached_KINETICS400_V1':backbone_matches,'interpreter':sys.executable,'gpu':torch.cuda.get_device_name(0),'versions':{n:importlib.metadata.version(n) for n in ['torch','torchvision','Pillow','numpy','pandas']},'dataset_code_sha256':digest(ROOT/'src/data/dataset.py'),'original_trainer_sha256':digest(ROOT/'src/train_videoswin_gt.py'),'evaluation_preprocessing':'RGB GT; PIL bilinear square 224; /255; ImageNet mean/std; 5x16 clips at 10/30/50/70/90%; mean probabilities','optimizer_lr':[g['lr'] for g in groups],'optimizer_weight_decay':[g['weight_decay'] for g in groups],'segment_seconds':timings,'training_settings_evidence':'checkpoint metadata/optimizer state and current legacy source; full historical environment and RNG provenance absent','test_used':False,'retrained':False}
    save_json(out/'metrics.json',metrics);save_json(out/'provenance.json',provenance);save_json(out/'completed.json',{'evaluation_only':True,'segments':13,'test_used':False})
    print(json.dumps(metrics),flush=True)


def compare(args,out):
    import pandas as pd
    evaluation=args.evaluation.resolve();m=read_json(evaluation/'metrics.json');p=read_json(evaluation/'provenance.json')
    if not read_json(evaluation/'completed.json')['evaluation_only']:raise ValueError('Not evaluation-only evidence.')
    if p['metadata_sha256']!=digest(ROOT/'data/labels/quadrant_metadata.csv'):raise ValueError('Validation metadata changed.')
    rows=[{'model':'Original four-class model evaluated as binary','status':'evaluated','training_segments':81,'selected_epoch':m['checkpoint_epoch'],'accuracy':m['accuracy'],'balanced_accuracy':m['balanced_accuracy'],'macro_f1':m['macro_f1'],'confusion_matrix':json.dumps(m['confusion_matrix']),'source':str(evaluation)}]
    arms=None
    if args.original is not None and args.cleaned is not None:
        arms=inputs(args)
        for name,arm in arms.items():
            if arm['provenance']['input_sha256']['data/labels/quadrant_metadata.csv']!=p['metadata_sha256']:raise ValueError('Validation inputs differ.')
            if arm['provenance']['input_sha256']['src/data/dataset.py']!=p['dataset_code_sha256']:raise ValueError('Dataset preprocessing code differs.')
    elif args.original is not None or args.cleaned is not None:raise ValueError('Provide both binary runs or neither; never substitute historical cleaned checkpoints.')
    for arm,count in [('original',81),('cleaned',53)]:
        if arms:
            data=arms[arm];metric=data['metrics']['best']
            rows.append({'model':f'{arm.capitalize()} {count}-segment binary model','status':'completed','training_segments':count,'selected_epoch':metric['epoch'],'accuracy':metric['accuracy'],'balanced_accuracy':metric['balanced_accuracy'],'macro_f1':metric['macro_f1'],'confusion_matrix':json.dumps(metric['confusion_matrix']),'source':str(data['path'])})
        else:rows.append({'model':f'{arm.capitalize()} {count}-segment binary model','status':'pending approved training','training_segments':count,'selected_epoch':None,'accuracy':None,'balanced_accuracy':None,'macro_f1':None,'confusion_matrix':None,'source':None})
    csv_out(out/'comparison.csv',rows);save_json(out/'comparison.json',{'rows':rows,'test_used':False})
    lines=['# Three-model binary validation comparison','','Same 13 validation segments; confusion-matrix order Upper/Lower, rows=true and columns=predicted.','','| Model | Status | Epoch | Accuracy | Balanced accuracy | Macro-F1 | Confusion matrix |','|---|---|---:|---:|---:|---:|---|']
    for r in rows:
        vals=['Pending']*4 if r['accuracy'] is None else [str(r['selected_epoch']),f'{r["accuracy"]:.2%}',f'{r["balanced_accuracy"]:.2%}',f'{r["macro_f1"]:.4f}']
        lines.append(f'| {r["model"]} | {r["status"]} | '+ ' | '.join(vals)+f' | {r["confusion_matrix"] or "Pending"} |')
    lines+=['','## Four-class checkpoint mapped to binary','','The original evaluation rule was preserved: average five four-class probability vectors, take the quadrant argmax, then map Q1/Q2 to Upper and Q3/Q4 to Lower. This is not argmax after summing quadrant probabilities into binary groups.',f'Strict state-dict loading passed; original four-class Macro-F1 reproduced at {m["four_class_macro_f1"]:.4f}. No retraining occurred.','','| True quadrant | Count | Binary correct | Binary errors | Binary accuracy |','|---|---:|---:|---:|---:|']
    for row in pd.read_csv(evaluation/'per_quadrant.csv').itertuples():lines.append(f'| {row.quadrant} | {row.count} | {row.binary_correct} | {row.binary_errors} | {row.binary_accuracy:.2%} |')
    lines+=['','## Differences in training settings','','| Setting | Existing four-class model | Original binary | Cleaned binary |','|---|---|---|---|','| Objective/head | Q1-Q4, four outputs | Upper/Lower, two outputs | Upper/Lower, two outputs |','| Training data | Original 81 | Original 81 | Cleaned 53 |','| Sampling | Sequential, no oversampling | Balanced, Lower replacement | Balanced, Lower replacement |','| Updates per epoch | 81 | 126 | 92 |','| Loss | Nominal inverse-sqrt-weighted CE; singleton mean reduction cancels weights | Unweighted CE | Unweighted CE |','| Epoch policy | Maximum 15, patience 3; best epoch 9; actual final epoch unknown | Fixed 15 | Fixed 15 |','| Checkpoint selection | Four-class validation Macro-F1 | Binary validation Macro-F1 | Binary validation Macro-F1 |','| Global training seed | Not recorded in legacy source/checkpoint | 42 | 42 |','| Optimizer | AdamW, lr 0.001, weight decay 0.0001 in current source and saved optimizer | Same | Same |','| Backbone | Head-only; stochastic depth remains active during training | Same policy | Same policy |','| Preprocessing / aggregation | RGB GT, square 224, normalization, 5x16 clips; mean probabilities | Same inputs; two-class probabilities | Same inputs; two-class probabilities |','','The four-class frozen backbone matches cached KINETICS400_V1 weights: '+str(p['backbone_matches_cached_KINETICS400_V1'])+'. Historical package versions and initialization seed are not recorded. Current legacy source is evidence of intended settings, not a complete historical run manifest.','', 'Different objectives, sampling, update counts, seeds, stopping and checkpoint selection mean this is a model comparison, not a controlled ablation between all three rows. The two new binary arms remain the controlled comparison at equal epochs. A small repeatedly used validation set and best-epoch selection limit generalization claims. No held-out test frames were accessed.']
    with (out/'comparison_report.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    print('Comparison saved:',out,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['evaluate','compare'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--evaluation',type=Path);parser.add_argument('--original',type=Path);parser.add_argument('--cleaned',type=Path)
    args=parser.parse_args()
    if Path(sys.executable).resolve()!=(ROOT/'.venv/Scripts/python.exe').resolve():raise RuntimeError('Use the explicit repository .venv interpreter.')
    if not args.output.resolve().is_relative_to((ROOT/'outputs/runs').resolve()):raise ValueError('Output must be under outputs/runs.')
    args.output.mkdir(parents=True,exist_ok=False)
    try: {'evaluate':evaluate,'compare':compare}[args.stage](args,args.output)
    except Exception as error:save_json(args.output/'failure.json',{'type':type(error).__name__,'message':str(error)});raise


if __name__=='__main__':main()
