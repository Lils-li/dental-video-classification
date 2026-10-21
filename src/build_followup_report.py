"""CPU-only report collector. Never imports torch or runs inference/training."""
import argparse,csv,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--current',type=Path,required=True);parser.add_argument('--followup',type=Path,required=True)
    args=parser.parse_args();out=args.followup/'report';out.mkdir(exist_ok=False)
    status=read(args.followup/'status.json');rows=[];quadrants=[];issues=[]
    import pandas as pd
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
    metadata=pd.read_csv(ROOT/'data/labels/quadrant_metadata.csv');metadata=metadata[metadata.split=='val']
    paths=[('Original 81 binary',args.current/'original_81_seed42'),('Cleaned 53 balanced binary',args.current/'cleaned_53_seed42'),('Cleaned corrected weighted CE',args.followup/'corrected_weighted'),('Partial last-block fine-tuning',args.followup/'partial_finetune')]
    for name,path in paths:
        if not (path/'completed.json').is_file():
            rows.append({'model':name,'status':'not completed; see stage status','epoch':None,'accuracy':None,'balanced_accuracy':None,'macro_f1':None,'confusion_matrix':None,'source':str(path)});continue
        try:
            completion=read(path/'completed.json')
            if completion['smoke_only'] or completion['test_used']:raise ValueError('Invalid completion marker.')
            config=read(path/'config.json');history=read(path/'metrics.json');best=read(path/'best_checkpoint.json')
            if completion['epochs']!=config['epochs'] or len(history)!=config['epochs']:raise ValueError('Epoch count mismatch.')
            if best['epoch']!=max(history,key=lambda h:h['val']['macro_f1'])['epoch']:raise ValueError('Best epoch mismatch.')
            for scope,epoch in [('best',best['epoch']),('final',config['epochs'])]:
                pred=pd.read_csv(path/f'epoch_{epoch:02d}_val_predictions.csv')
                joined=pred.merge(metadata[['segment_id','label']],on='segment_id',how='outer',validate='one_to_one',indicator=True)
                if len(joined)!=13 or not (joined['_merge']=='both').all() or not (joined.original_label==joined.label).all():raise ValueError('Validation metadata mismatch.')
                truth=joined.true_binary.map({'Upper':0,'Lower':1});prediction=joined.predicted_binary.map({'Upper':0,'Lower':1})
                if not (joined.true_binary==joined.label.map({'Q1':'Upper','Q2':'Upper','Q3':'Lower','Q4':'Lower'})).all():raise ValueError('Label mapping mismatch.')
                metric={'accuracy':float(accuracy_score(truth,prediction)),'balanced_accuracy':float(balanced_accuracy_score(truth,prediction)),'macro_f1':float(f1_score(truth,prediction,labels=[0,1],average='macro',zero_division=0)),'confusion_matrix':confusion_matrix(truth,prediction,labels=[0,1]).tolist()}
                saved=history[epoch-1]['val']
                if any(abs(metric[k]-saved[k])>1e-10 for k in ['accuracy','balanced_accuracy','macro_f1']) or metric['confusion_matrix']!=saved['confusion_matrix']:raise ValueError('Saved metrics inconsistent.')
                rows.append({'model':name+' / '+scope,'status':'completed','epoch':epoch,**metric,'source':str(path)})
                for q in ['Q1','Q2','Q3','Q4']:
                    sub=joined[joined.label==q];errors=int((sub.true_binary!=sub.predicted_binary).sum())
                    quadrants.append({'model':name,'scope':scope,'quadrant':q,'count':len(sub),'errors':errors,'accuracy':1-errors/len(sub)})
        except Exception as e:issues.append(name+': '+str(e));rows.append({'model':name,'status':'rejected: '+str(e),'epoch':None,'accuracy':None,'balanced_accuracy':None,'macro_f1':None,'confusion_matrix':None,'source':str(path)})
    four=args.current/'fourclass_as_binary'
    if not (four/'completed.json').is_file():four=ROOT/'outputs/runs/fourclass_as_binary_2026-10-14_eval01'
    if (four/'completed.json').is_file():
        try:
            m=read(four/'metrics.json')
            if m['test_used'] or m['segments']!=13:raise ValueError('Invalid four-class evaluation.')
            rows.insert(0,{'model':'Original four-class mapped to binary','status':'evaluation only','epoch':m['checkpoint_epoch'],**{k:m[k] for k in ['accuracy','balanced_accuracy','macro_f1','confusion_matrix']},'source':str(four)})
        except Exception as e:issues.append('Four-class evaluation: '+str(e))
    for filename,data in [('comparison.csv',rows),('per_quadrant.csv',quadrants)]:
        if data:
            with (out/filename).open('x',newline='',encoding='utf-8') as stream:
                w=csv.DictWriter(stream,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    summary={'generated_utc':datetime.now(timezone.utc).isoformat(),'current_pipeline':str(args.current),'stage_status':status,'results':rows,'issues':issues,'test_used':False}
    with (out/'results.json').open('x',encoding='utf-8') as stream:json.dump(summary,stream,indent=2)
    text=['# Supervisor report: binary VideoSwin follow-up','','Meeting: 15 October 2026, 16:00 Astana (+05:00). Lab arrival: 12:00. GPU start cutoff: 11:00; follow-up GPU finish cutoff: 11:40.','','## Successfully completed results','','Rows with missing metrics were skipped, failed, or unfinished; they are not presented as completed experiments. Matrices: true rows, predicted columns; Upper/Lower order.','','| Model | Status | Epoch | Accuracy | Balanced accuracy | Macro-F1 | Confusion matrix |','|---|---|---:|---:|---:|---:|---|']
    for r in rows:
        values=['--']*4 if r['accuracy'] is None else [str(r['epoch']),f'{r["accuracy"]:.2%}',f'{r["balanced_accuracy"]:.2%}',f'{r["macro_f1"]:.4f}']
        text.append(f'| {r["model"]} | {r["status"]} | '+ ' | '.join(values)+f' | {r["confusion_matrix"] or "--"} |')
    text+=['','## Stage outcomes','']
    for item in status['stages']:text.append('- '+item['stage']+': '+item['status']+'; '+item.get('reason',''))
    text+=['','## Training settings and interpretation','','- Existing original/cleaned balanced experiments and all original files were left unchanged.','- Corrected weighted experiment: cleaned 53 segments, no oversampling, seeded shuffle once per epoch, inverse-square-root class weights normalized to mean one; cross-entropy reduction=sum retains the singleton target weight; head-only AdamW, lr=0.001, weight decay=0.0001, seed=42, 15 fixed epochs.','- The old weighted mean-reduction branch cancelled class weights for singleton targets. A separate numerical loss/gradient check confirms the corrected reduction retains them.','- Partial pilot, if completed: warm start from the corrected weighted best checkpoint; unfreeze only features[6][-1] and binary head; fresh AdamW with head lr=0.0001 and block lr=0.00001, weight decay=0.0001, three fixed epochs, same corrected loss, data and preprocessing. This is an exploratory warm-start pilot, not an equal-budget ablation.','- All new runs use RGB GT, five clips of sixteen frames at 224x224, original normalization, mean clip probabilities, original 13 validation segments, FP32, deterministic settings, and stochastic-depth training mode.','- Balanced cleaned training has 92 updates/epoch; corrected unbalanced training has 53. Sampling and budget differ, so observed differences cannot be attributed solely to the weighting correction.','- Best-epoch selection reuses validation. One seed, 13 segments, 7 retained Lower training segments, and unknown patient independence limit claims. One validation case changes accuracy by 7.69 points. No test-set evaluation occurred.','- Incomplete runs retain epoch artifacts/checkpoints but are excluded from completed-result claims. Optional experiments are not resized or retried with different parameters after failure.','','## Per-quadrant binary errors','','| Model | Scope | Quadrant | Count | Errors | Accuracy |','|---|---|---|---:|---:|---:|']
    for r in quadrants:text.append(f'| {r["model"]} | {r["scope"]} | {r["quadrant"]} | {r["count"]} | {r["errors"]} | {r["accuracy"]:.2%} |')
    text+=['','## Recommended next experiments (not executed)','','1. Compare corrected weighting against an unweighted, non-oversampled, equal-update control.','2. Repeat prespecified seeds and check patient-group independence.','3. If the last-block pilot is promising, compare equal training budgets and confirm robustness without using the test set for tuning.','','## Artifacts','',f'Current pipeline: `{args.current}`',f'Follow-up: `{args.followup}`','Configs, metrics, predictions, confusion matrices, checkpoints, stage logs and status are retained in these directories. Existing overnight reports include exploratory occlusion if it completed; consult their figures and limitations.']
    if issues:text+=['','## Validation/reporting issues','']+['- '+i for i in issues]
    with (out/'supervisor_report.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(text)+'\n')
    print('Follow-up report:',out/'supervisor_report.md',flush=True)


if __name__=='__main__':main()
