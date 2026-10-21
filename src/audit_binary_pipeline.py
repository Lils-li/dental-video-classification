"""Read-only legacy checkpoint evaluation; writes only a fresh audit directory."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_binary_comparison import MAPPING, NAMES, sha, seed_all, summarize, save_predictions, write_json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models.video import swin3d_t
from torchvision.ops import StochasticDepth
from data.dataset import DentalQuadrantDataset


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path,required=True)
    args=parser.parse_args()
    if Path(sys.executable).resolve() != (ROOT/".venv/Scripts/python.exe").resolve():
        raise RuntimeError("Use the repository .venv explicitly.")
    run=args.run_dir.resolve()
    if not run.is_relative_to((ROOT/"outputs/runs").resolve()):
        raise RuntimeError("Audit must use outputs/runs.")
    run.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    seed_all(42)
    device=torch.device("cuda")
    metadata=ROOT/"data/labels/quadrant_metadata.csv"
    dataset=DentalQuadrantDataset(metadata,data_root=ROOT/"data/raw/Vident-real",split="val",frame_source="GT")
    assert len(dataset)==13 and set(dataset.df.split)=={"val"}
    dataset.df.to_csv(run/"validation_manifest.csv",index=False)
    x=torch.tensor([[0.3,-0.2]],requires_grad=True)
    y=torch.tensor([1])
    counts=torch.tensor([46,7],dtype=torch.float32)
    class_weights=1/counts.sqrt();class_weights/=class_weights.mean()
    plain=nn.CrossEntropyLoss()(x,y)
    weighted=nn.CrossEntropyLoss(weight=class_weights)(x,y)
    plain_gradient=torch.autograd.grad(plain,x,retain_graph=True)[0]
    weighted_gradient=torch.autograd.grad(weighted,x)[0]
    assert torch.allclose(plain,weighted) and torch.allclose(plain_gradient,weighted_gradient)
    findings={"singleton_weighted_loss":{"plain":plain.item(),"weighted":weighted.item(),"class_weights":class_weights.tolist(),"gradients_equal":torch.allclose(plain_gradient,weighted_gradient)},"test_used":False,"metadata_sha256":sha(metadata),"interpreter":sys.executable,"checkpoints":{}}
    mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1,1)
    std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1,1)
    for name in ["videoswin_binary_clean_best.pt","videoswin_binary_clean_balanced_best.pt"]:
        path=ROOT/"outputs/checkpoints"/name
        checkpoint=torch.load(path,map_location="cpu",weights_only=True)
        model=swin3d_t(weights=None)
        model.head=nn.Linear(model.head.in_features,2)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        rows,losses,times=[],[],[]
        with torch.no_grad():
            for index in range(len(dataset)):
                torch.cuda.synchronize();start=time.perf_counter()
                sample=dataset[index]
                target=MAPPING[sample["label_name"]]
                target_tensor=torch.tensor([target],device=device)
                probabilities=[];loss_sum=0.
                for clip in sample["clips"]:
                    inp=(clip.to(device).permute(1,0,2,3).unsqueeze(0)-mean)/std
                    logits=model(inp)
                    loss_sum+=nn.functional.cross_entropy(logits,target_tensor).item()/5
                    probabilities.append(logits.softmax(1).cpu())
                prob=torch.cat(probabilities).mean(0);pred=int(prob.argmax())
                rows.append({"segment_id":sample["segment_id"],"original_label":sample["label_name"],"true_id":target,"predicted_id":pred,"true_binary":NAMES[target],"predicted_binary":NAMES[pred],"p_upper":float(prob[0]),"p_lower":float(prob[1]),"correct":target==pred})
                losses.append(loss_sum)
                torch.cuda.synchronize();times.append(time.perf_counter()-start)
                print(f"checkpoint={name} validation_segment={sample['segment_id']} seconds={times[-1]:.3f}",flush=True)
        metrics=summarize(rows,losses)
        save_predictions(run,path.stem,rows,metrics)
        findings["checkpoints"][name]={"sha256":sha(path),"epoch":checkpoint["epoch"],"saved_val_macro_f1":checkpoint["val_macro_f1"],"reevaluated":metrics,"segment_seconds":times}
        assert abs(metrics["macro_f1"]-checkpoint["val_macro_f1"])<1e-10
        if "balanced" in name:
            old=pd.read_csv(ROOT/"outputs/binary_val_predictions.csv").sort_values("segment_id")
            new=pd.DataFrame(rows).sort_values("segment_id")
            assert old.segment_id.tolist()==new.segment_id.tolist()
            findings["saved_csv_matches"]={"labels_equal":old.predicted_binary.tolist()==new.predicted_binary.tolist(),"max_probability_difference":float(np.max(np.abs(old[['p_upper','p_lower']].to_numpy()-new[['p_upper','p_lower']].to_numpy())))}
            model.train()
            for parameter in model.parameters():parameter.requires_grad=False
            with torch.no_grad():
                z1=model(inp);z2=model(inp)
            findings["frozen_backbone_train_mode"]={"active_stochastic_depth_layers":sum(isinstance(m,StochasticDepth) and m.p>0 and m.training for m in model.modules()),"repeated_input_logits_max_difference":float((z1-z2).abs().max())}
        print(json.dumps({"checkpoint":name,"metrics":metrics}),flush=True)
        del model,checkpoint
    write_json(run/"audit_findings.json",findings)
    write_json(run/"completed.json",{"test_used":False,"full_training_started":False})


if __name__=="__main__":main()
