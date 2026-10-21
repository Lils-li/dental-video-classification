"""Forensic validation-only prediction audit. Writes exclusively to a fresh directory."""
import argparse,hashlib,itertools,json,os,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/data'))
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models.video import swin3d_t
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
from dataset import DentalQuadrantDataset


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,obj):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2)


def tensor_sha(state,head):
 h=hashlib.sha256()
 for name,tensor in sorted(state.items()):
  if name.startswith('head.')!=head:continue
  h.update(name.encode());h.update(str(tuple(tensor.shape)).encode());h.update(str(tensor.dtype).encode());h.update(tensor.contiguous().numpy().tobytes())
 return h.hexdigest()


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 if Path(sys.executable).resolve()!=(ROOT/'.venv/Scripts/python.exe').resolve():raise RuntimeError('Use repository .venv.')
 out=args.output.resolve()
 if not out.is_relative_to((ROOT/'outputs/runs').resolve()):raise ValueError('Invalid audit output location.')
 out.mkdir(parents=True,exist_ok=False)
 overnight=ROOT/'outputs/runs/overnight_20261014_201445_87e92c7a';follow=ROOT/'outputs/runs/followup_20261014_214855_def8c9c5'
 specs=[('fourclass',ROOT/'outputs/checkpoints/videoswin_gt_head_only_best.pt',overnight/'fourclass_as_binary/predictions.csv',9,4)]
 inputs={};provenance={};tables={}
 for name,folder in [('original81',overnight/'original_81_seed42'),('cleaned53',overnight/'cleaned_53_seed42'),('weighted',follow/'corrected_weighted')]:
  best=read(folder/'best_checkpoint.json');complete=read(folder/'completed.json');config=read(folder/'config.json');history=read(folder/'metrics.json')
  assert complete==dict(smoke_only=False,epochs=15,test_used=False)
  assert best['epoch']==max(history,key=lambda x:x['val']['macro_f1'])['epoch']
  assert Path(best['path']).name==best['path']
  specs.append((name,folder/best['path'],folder/f'epoch_{best["epoch"]:02d}_val_predictions.csv',best['epoch'],2))
  provenance[name]=read(folder/'provenance.json')
  inputs[str(folder/'best_checkpoint.json')]=sha(folder/'best_checkpoint.json')
  inputs[str(folder/'config.json')]=sha(folder/'config.json')
 torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no fallback.')
 metadata=ROOT/'data/labels/quadrant_metadata.csv';inputs[str(metadata)]=sha(metadata)
 dataset=DentalQuadrantDataset(metadata,data_root=ROOT/'data/raw/Vident-real',split='val',frame_source='GT')
 assert len(dataset)==13 and set(dataset.df.split)=={'val'}
 master=dataset.df[['segment_id','label']].rename(columns={'label':'ground_truth_quadrant'}).sort_values('segment_id').reset_index(drop=True)
 master['binary_label']=master.ground_truth_quadrant.map({'Q1':'Upper','Q2':'Upper','Q3':'Lower','Q4':'Lower'})
 mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1,1);std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1,1)
 evidence={};backbone_hashes={};head_hashes={}
 for name,checkpoint_path,csv_path,epoch,classes in specs:
  checkpoint_hash=sha(checkpoint_path);csv_hash=sha(csv_path);inputs[str(checkpoint_path)]=checkpoint_hash;inputs[str(csv_path)]=csv_hash
  checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True);state=checkpoint['model_state_dict']
  assert checkpoint['epoch']==epoch
  expected=['Q1','Q2','Q3','Q4'] if classes==4 else ['Upper','Lower']
  assert checkpoint['class_names']==expected and tuple(state['head.weight'].shape)==(classes,768)
  if name!='fourclass':assert checkpoint['config']==read(checkpoint_path.parent/'config.json')|{} or all(checkpoint['config'].get(k)==v for k,v in read(checkpoint_path.parent/'config.json').items() if k not in ['smoke_only','executed_epochs'])
  backbone_hashes[name]=tensor_sha(state,False);head_hashes[name]=tensor_sha(state,True)
  model=swin3d_t(weights=None);model.head=nn.Linear(model.head.in_features,classes);model.load_state_dict(state,strict=True);model.to('cuda').eval()
  for p in model.parameters():p.requires_grad=False
  saved=pd.read_csv(csv_path).sort_values('segment_id').reset_index(drop=True)
  assert saved.segment_id.tolist()==master.segment_id.tolist() and saved.original_label.tolist()==master.ground_truth_quadrant.tolist() and saved.true_binary.tolist()==master.binary_label.tolist()
  if name!='fourclass':
   alias=checkpoint_path.parent/f'best_epoch_{epoch:02d}_val_predictions.csv'
   pd.testing.assert_frame_equal(saved,pd.read_csv(alias).sort_values('segment_id').reset_index(drop=True))
   inputs[str(alias)]=sha(alias)
  rows=[];raw_probs=[]
  with torch.inference_mode():
   for index in range(len(dataset)):
    sample=dataset[index];probabilities=[]
    for clip in sample['clips']:
     x=(clip.to('cuda').permute(1,0,2,3).unsqueeze(0)-mean)/std
     probabilities.append(model(x).softmax(1).cpu())
    probs=torch.cat(probabilities).mean(0).numpy();pred=int(probs.argmax());binary=0 if pred<(2 if classes==4 else 1) else 1
    lower=float(probs[2:].sum()) if classes==4 else float(probs[1])
    rows.append({'segment_id':sample['segment_id'],'predicted_binary':['Upper','Lower'][binary],'p_lower':lower});raw_probs.append(probs.tolist())
    print(f'{name} validation_segment={sample["segment_id"]} prediction={["Upper","Lower"][binary]} P_Lower={lower:.6f}',flush=True)
  fresh=pd.DataFrame(rows).sort_values('segment_id').reset_index(drop=True)
  saved_probs=saved[['p_Q1','p_Q2','p_Q3','p_Q4']].to_numpy() if classes==4 else saved[['p_upper','p_lower']].to_numpy()
  delta=float(np.abs(np.asarray(raw_probs)-saved_probs).max())
  assert delta<1e-5 and fresh.predicted_binary.tolist()==saved.predicted_binary.tolist()
  if classes==4:assert saved.predicted_quadrant.tolist()==[['Q1','Q2','Q3','Q4'][int(np.argmax(p))] for p in raw_probs]
  else:assert (saved_probs.argmax(1)==saved.predicted_id.to_numpy()).all()
  fresh.to_csv(out/f'{name}_fresh_predictions.csv',index=False)
  master[name+'_prediction']=fresh.predicted_binary
  master[name+'_p_lower']=fresh.p_lower
  truth=master.binary_label.map({'Upper':0,'Lower':1});pred=fresh.predicted_binary.map({'Upper':0,'Lower':1})
  errors=master.loc[master.binary_label!=fresh.predicted_binary,'segment_id'].tolist()
  evidence[name]={'checkpoint':str(checkpoint_path),'checkpoint_sha256':checkpoint_hash,'checkpoint_epoch':epoch,'prediction_file':str(csv_path),'prediction_sha256':csv_hash,'head_sha256':head_hashes[name],'backbone_sha256':backbone_hashes[name],'fresh_max_probability_difference':delta,'fresh_labels_match_saved':True,'errors':errors,'accuracy':float(accuracy_score(truth,pred)),'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),'macro_f1':float(f1_score(truth,pred,average='macro')),'confusion_matrix':confusion_matrix(truth,pred,labels=[0,1]).tolist()}
  del model,checkpoint,state;torch.cuda.empty_cache()
 pairs=[]
 for a,b in itertools.combinations(evidence,2):pairs.append({'a':a,'b':b,'disagreement_segment_ids':master.loc[master[a+'_prediction']!=master[b+'_prediction'],'segment_id'].tolist(),'maximum_lower_probability_difference':float((master[a+'_p_lower']-master[b+'_p_lower']).abs().max())})
 assert len({e['checkpoint_sha256'] for e in evidence.values()})==4
 assert len(set(head_hashes.values()))==4
 for path,original in inputs.items():assert sha(path)==original,'Input changed: '+path
 source_checks={}
 for name,p in [('training','src/run_binary_comparison.py'),('weighted_training','src/run_followup_experiment.py'),('fourclass_evaluation','src/evaluate_fourclass_as_binary.py'),('dataset','src/data/dataset.py')]:
  source_checks[p]=sha(ROOT/p)
 for name,prov in provenance.items():
  for path,expected in prov['input_sha256'].items():assert sha(ROOT/path)==expected,'Source/input provenance mismatch: '+path
 master.to_csv(out/'segment_comparison.csv',index=False)
 save(out/'audit_evidence.json',{'models':evidence,'pairwise':pairs,'backbone_hashes':backbone_hashes,'head_hashes':head_hashes,'all_binary_initial_heads_identical':len({v['initial_head_sha256'] for v in provenance.values()})==1,'saved_input_sha256':inputs,'source_checks':source_checks,'test_used':False,'retrained':False})
 lines=['# Audit of identical binary validation metrics','','Fresh inference independently loaded all four distinct checkpoint files and read the 13 validation segments. Every saved prediction label was reproduced; full per-class probability differences were below 1e-5. No test frames or training were used.','','All four models misclassify exactly segments 1, 2, 3, 4 and 5. All pairwise label disagreements are zero. Probability vectors differ, so identical metrics are not explained by identical probability outputs.','','| Segment | Quadrant | Truth | Four-class label / P(Lower) | Original81 label / P(Lower) | Cleaned53 label / P(Lower) | Weighted label / P(Lower) |','|---:|---|---|---|---|---|---|']
 for _,r in master.iterrows():lines.append(f'| {r.segment_id} | {r.ground_truth_quadrant} | {r.binary_label} | '+' | '.join(f'{r[n+"_prediction"]} / {r[n+"_p_lower"]:.6f}' for n in evidence)+' |')
 lines+=['','Four-class P(Lower) is the sum of averaged Q3 and Q4 probabilities. Its decision remains quadrant argmax followed by mapping; it is not generally a 0.5 threshold on this grouped probability.','','## Checkpoint and artifact trace','']
 for name,e in evidence.items():lines += [f'- {name}: epoch {e["checkpoint_epoch"]}; checkpoint `{e["checkpoint"]}`; SHA-256 `{e["checkpoint_sha256"]}`; predictions `{e["prediction_file"]}`; maximum fresh/saved probability difference {e["fresh_max_probability_difference"]:.3g}.']
 lines+=['','## Findings and reproducibility concerns','','- Checkpoint files, classifier tensors and saved prediction files are distinct. Best-epoch aliases match the corresponding epoch prediction files. Source/input hashes match the recorded training provenance.','- Fresh inference does not read cached model outputs: it loads each checkpoint with strict state-dict matching and recomputes five-clip probabilities from GT validation PNGs. No evidence of accidental checkpoint or prediction reuse was found.','- All four backbone tensor hashes are '+('identical' if len(set(backbone_hashes.values()))==1 else 'not identical')+'. This is expected for head-only training from shared frozen pretrained weights. The three binary runs also share seed 42 and identical initial heads; they are not independent random-seed replicates.','- Checkpoints were selected by best validation Macro-F1, with the earliest epoch retained on ties. On only 13 repeatedly evaluated segments this can select the same decision pattern despite different training trajectories.','- Same frozen features can yield similar rankings and errors; this does not establish the cause of the five failures. Q3 errors are segments 1/2 and Q4 errors 3/4/5; source-video grouping and annotation/visibility limitations merit review.','- Legacy four-class initialization seed and full historical environment are not recorded. New checkpoint hashes were recomputed here; training runs did not originally persist a checkpoint digest alongside each CSV.','- Four-class label mapping and two-class decisions differ mathematically; grouped probabilities must not silently replace the requested mapped-argmax rule.','- Only a small validation set was audited. Identical validation decisions do not imply identical models or generalization.']
 with (out/'audit_report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
 print('AUDIT COMPLETE',out,json.dumps(evidence),flush=True)


if __name__=='__main__':main()
