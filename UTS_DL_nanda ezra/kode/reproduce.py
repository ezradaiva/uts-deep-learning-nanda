"""Fast oral-exam reproduction using the saved final classification ensemble."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from kode.pipeline import LABELS, ClassNet, ClassRows, class_arrays, load_data


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--seeds",type=int,nargs="+",default=[7,21,42])
    a=p.parse_args()
    root=a.root.resolve(); device="cuda" if torch.cuda.is_available() else "cpu"
    d=load_data(a.data); arr=class_arrays(d)
    _,_,_,images,signals,valid=arr
    logits=[]
    for seed in a.seeds:
        path=root/"artifacts"/"models"/f"class_final_{seed}.pt"
        model=ClassNet("late").to(device)
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
        model.eval(); result=[]
        with torch.no_grad():
            loader=DataLoader(ClassRows(images,signals,valid,np.arange(len(images))),batch_size=256)
            for im,se,va in loader:
                result.append(model(im.to(device),se.to(device),va.to(device)).cpu().numpy())
        logits.append(np.concatenate(result))
    pred=np.stack(logits).mean(0).argmax(1)
    counts={name:int(count) for name,count in Counter(np.asarray(LABELS)[pred]).items()}
    for name in LABELS: counts.setdefault(name,0)
    print(f"device: {device}")
    print(f"test samples: {len(pred)}")
    print("final ensemble predicted-class counts:")
    for name in LABELS: print(f"  {name}: {counts[name]}")
    print("Compare these counts with artifacts/metrics.json and laporan.pdf.")

if __name__=="__main__": main()
