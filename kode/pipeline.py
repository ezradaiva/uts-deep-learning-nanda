"""Train the three UTS tasks and write the required prediction files.

Run from the project root:
python -m kode.pipeline --data "D:/Kuliah/DL/uts/data_uts_dl.zip"
"""
from __future__ import annotations
import argparse, gzip, io, json, random, sys, textwrap, zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import sklearn
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import classification_report, davies_bouldin_score, mean_squared_error, silhouette_score
from sklearn.manifold import TSNE
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

LABELS = ["normal", "gores", "noda", "retak", "gelembung", "las_dingin", "kontaminasi"]
SENSORS = [f"s{i:02d}" for i in range(1, 22)]
RUL_FEATURES = ["setel_1", "setel_2", "setel_3"] + SENSORS

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def npz_from_zip(z, name):
    inner = zipfile.ZipFile(io.BytesIO(z.read(name)))
    try: return {Path(p).stem: np.load(inner.open(p), allow_pickle=False) for p in inner.namelist()}
    finally: inner.close()

def gz_csv(z, name):
    with z.open(name) as raw, gzip.GzipFile(fileobj=raw) as gz: return pd.read_csv(gz)

def load_data(path):
    with zipfile.ZipFile(path) as z:
        return {
            "train_meta": pd.read_csv(z.open("klasifikasi_train_meta.csv")),
            "test_meta": pd.read_csv(z.open("klasifikasi_test_meta.csv")),
            "cluster_meta": pd.read_csv(z.open("clustering_meta.csv")),
            "class_train": npz_from_zip(z, "klasifikasi_train.npz"),
            "class_test": npz_from_zip(z, "klasifikasi_test.npz"),
            "cluster": npz_from_zip(z, "clustering.npz"),
            "rul_train": gz_csv(z, "rul_train.csv.gz"),
            "rul_test": gz_csv(z, "rul_test.csv.gz"),
            "datasheet": z.read("kamus_data.md").decode("utf-8"),
        }

class ClassRows(Dataset):
    def __init__(self, images, signals, valid, ids, y=None, aug=False):
        self.images, self.signals, self.valid = images, signals, valid
        self.ids, self.y, self.aug = np.asarray(ids), y, aug
    def __len__(self): return len(self.ids)
    def __getitem__(self, j):
        i = self.ids[j]
        im = torch.from_numpy(self.images[i].astype(np.float32, copy=False))[None] / 255.
        sig = torch.from_numpy(self.signals[i].astype(np.float32, copy=False))
        valid = torch.tensor(self.valid[i], dtype=torch.float32)
        if self.aug and random.random() < .5: im = im.flip(-1)
        if self.y is None: return im, sig, valid
        return im, sig, valid, int(self.y[i])

class ImageEncoder(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ch,24,3,padding=1),nn.BatchNorm2d(24),nn.GELU(),nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1),nn.BatchNorm2d(48),nn.GELU(),nn.MaxPool2d(2),
            nn.Conv2d(48,96,3,padding=1),nn.BatchNorm2d(96),nn.GELU(),nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),nn.Linear(96,128),nn.GELU())
    def forward(self,x): return self.net(x)

class SignalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(6,32,7,padding=3),nn.BatchNorm1d(32),nn.GELU(),nn.MaxPool1d(2),
            nn.Conv1d(32,64,5,padding=2),nn.BatchNorm1d(64),nn.GELU(),nn.MaxPool1d(2),
            nn.Conv1d(64,96,3,padding=1),nn.BatchNorm1d(96),nn.GELU(),nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),nn.Linear(96,128),nn.GELU())
    def forward(self,x): return self.net(x)

class ClassNet(nn.Module):
    def __init__(self, mode):
        super().__init__(); self.mode=mode
        if mode in ("image","late"): self.img=ImageEncoder(1)
        if mode in ("sensor","late"): self.sig=SignalEncoder()
        if mode=="early": self.img=ImageEncoder(4)
        if mode=="late": self.head=nn.Sequential(nn.Linear(257,160),nn.GELU(),nn.Dropout(.25),nn.Linear(160,7))
        else: self.head=nn.Sequential(nn.Dropout(.25),nn.Linear(128,7))
    def forward(self,im,sig,valid):
        if self.mode=="image": z=self.img(im)
        elif self.mode=="sensor": z=self.sig(sig)
        elif self.mode=="early":
            s=F.interpolate(sig[:,:3].unsqueeze(2),size=(64,64),mode="bilinear",align_corners=False)*valid[:,None,None,None]
            z=self.img(torch.cat([im,s],dim=1))
        else: return self.head(torch.cat([self.img(im),self.sig(sig),valid[:,None]],dim=1))
        return self.head(z)

def class_arrays(d,fit_indices=None):
    x=d["class_train"]["gambar"]; xt=d["class_test"]["gambar"]
    s=d["class_train"]["sensor"].astype(np.float32); st=d["class_test"]["sensor"].astype(np.float32)
    fit=s if fit_indices is None else s[fit_indices]
    mean=np.nanmean(fit,axis=(0,2),keepdims=True); std=np.nanstd(fit,axis=(0,2),keepdims=True).clip(1e-4)
    v=np.isfinite(s).any((1,2)).astype(np.float32); vt=np.isfinite(st).any((1,2)).astype(np.float32)
    def tx(a):
        mask=np.isfinite(a); a=np.where(mask,(a-mean)/std,0.)
        return np.concatenate([a,mask.astype(np.float32)],axis=1).astype(np.float32)
    return x,tx(s),v,xt,tx(st),vt

def class_eval(y,p,meta,ids):
    rep=classification_report(y,p,labels=range(7),target_names=LABELS,output_dict=True,zero_division=0)
    byline={}
    for line in sorted(meta.iloc[ids]["lini"].unique()):
        m=(meta.iloc[ids]["lini"].to_numpy()==line)
        rep_line=classification_report(np.asarray(y)[m],np.asarray(p)[m],labels=range(7),output_dict=True,zero_division=0)
        byline[str(line)]={"accuracy":float(np.mean(np.asarray(y)[m]==np.asarray(p)[m])),
                           "macro_f1":float(rep_line["macro avg"]["f1-score"])}
    return {"macro_f1":rep["macro avg"]["f1-score"],"accuracy":rep["accuracy"],
            "per_class_f1":{k:rep[k]["f1-score"] for k in LABELS},"line_accuracy":byline}

def fit_class(mode, arr, y, tri, vai, lines, device, seed, epochs, smoothing=0., balance=False, save=None):
    seed_all(seed); x,s,v,*_=arr
    ds=ClassRows(x,s,v,tri,y,True); sampler=None
    if balance:
        counts=pd.Series(lines[tri]).value_counts()
        w=[1./counts[lines[i]] for i in tri]
        sampler=WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),len(w),replacement=True)
    loader=DataLoader(ds,batch_size=128,shuffle=sampler is None,sampler=sampler)
    vl=DataLoader(ClassRows(x,s,v,vai,y),batch_size=256)
    net=ClassNet(mode).to(device); opt=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=1e-4)
    freq=np.bincount(y[tri],minlength=7); cw=(freq.sum()/(7*np.maximum(freq,1)))**.5
    lossfn=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float32,device=device),label_smoothing=smoothing)
    best=-1.; best_state=None; hist=[]
    for ep in range(epochs):
        net.train(); losses=[]
        for im,se,va,t in loader:
            im,se,va,t=im.to(device),se.to(device),va.to(device),t.to(device)
            opt.zero_grad(set_to_none=True); loss=lossfn(net(im,se,va),t); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),3.); opt.step(); losses.append(loss.item())
        net.eval(); preds=[]; truth=[]
        with torch.no_grad():
            for im,se,va,t in vl:
                preds.extend(net(im.to(device),se.to(device),va.to(device)).argmax(1).cpu().numpy()); truth.extend(t.numpy())
        f=classification_report(truth,preds,labels=range(7),output_dict=True,zero_division=0)["macro avg"]["f1-score"]
        hist.append({"epoch":ep+1,"train_loss":float(np.mean(losses)),"val_macro_f1":float(f)})
        print(f"classifier={mode} seed={seed} epoch={ep+1}/{epochs} val_macro_f1={f:.4f}",flush=True)
        if f>best: best=f; best_state={k:q.detach().cpu().clone() for k,q in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval(); outputs=[]
    with torch.no_grad():
        for im,se,va in DataLoader(ClassRows(x,s,v,vai),batch_size=256):
            outputs.append(net(im.to(device),se.to(device),va.to(device)).cpu().numpy())
    if save: torch.save(best_state,save)
    return np.concatenate(outputs),hist

def train_class_final(arr,y,lines,device,seed,epochs,save):
    seed_all(seed); x,s,v,xt,st,vt=arr; counts=pd.Series(lines).value_counts()
    w=[1./counts[a] for a in lines]; sampler=WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),len(y),replacement=True)
    loader=DataLoader(ClassRows(x,s,v,np.arange(len(y)),y,True),batch_size=128,sampler=sampler)
    net=ClassNet("late").to(device); opt=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=1e-4)
    freq=np.bincount(y,minlength=7); cw=(freq.sum()/(7*np.maximum(freq,1)))**.5
    lossfn=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float32,device=device),label_smoothing=.05)
    for _ in range(epochs):
        net.train()
        for im,se,va,t in loader:
            im,se,va,t=im.to(device),se.to(device),va.to(device),t.to(device)
            opt.zero_grad(set_to_none=True); loss=lossfn(net(im,se,va),t); loss.backward(); opt.step()
    torch.save(net.state_dict(),save); net.eval(); out=[]
    with torch.no_grad():
        for im,se,va in DataLoader(ClassRows(xt,st,vt,np.arange(len(xt))),batch_size=256):
            out.append(net(im.to(device),se.to(device),va.to(device)).cpu().numpy())
    return np.concatenate(out)

def task_classification(d,root,device,epochs,seeds):
    meta=d["train_meta"]; y=pd.Categorical(meta["kelas"],categories=LABELS).codes
    if (y<0).any(): raise ValueError("Unrecognized train label")
    ids=np.arange(len(y)); tr,va=train_test_split(ids,test_size=.2,random_state=42,stratify=y); lines=meta["lini"].to_numpy()
    # Validation preprocessing is fit on training-fold sensor values only.
    arr=class_arrays(d,tr)
    results={}; curves={}; suspected=[]
    runs=[("image","image",0.,False),("sensor","sensor",0.,False),("early_fusion","early",0.,False),
          ("late_fusion","late",0.,False),("label_smoothing","late",.1,False),("line_balanced","late",0.,True)]
    for name,mode,smooth,balance in runs:
        logits,h=fit_class(mode,arr,y,tr,va,lines,device,7,epochs,smooth,balance,root/"models"/f"class_{name}.pt")
        results[name]=class_eval(y[va],logits.argmax(1),meta,va); curves[name]=h
        if name=="late_fusion":
            row_loss=F.cross_entropy(torch.from_numpy(logits),torch.from_numpy(y[va].astype(np.int64)),reduction="none").numpy()
            top=np.argsort(row_loss)[-25:][::-1]
            suspected=[{"id_unit":str(meta.iloc[va[i]]["id_unit"]),
                        "inspector_label":str(meta.iloc[va[i]]["kelas"]),
                        "model_label":LABELS[int(logits[i].argmax())],
                        "validation_cross_entropy":float(row_loss[i])} for i in top]
    # Explicit one-batch memorization check: a small CNN should fit a fixed batch.
    seed_all(7); net=ClassNet("late").to(device); opt=torch.optim.AdamW(net.parameters(),lr=1e-3)
    batch=next(iter(DataLoader(ClassRows(arr[0],arr[1],arr[2],tr[:64],y,True),batch_size=64)))
    im,se,valid,t=[q.to(device) for q in batch]
    for _ in range(200):
        net.train();opt.zero_grad(set_to_none=True);loss=nn.CrossEntropyLoss()(net(im,se,valid),t);loss.backward();opt.step()
    net.eval()
    with torch.no_grad(): sanity_acc=float((net(im,se,valid).argmax(1)==t).float().mean().item())
    # Final models are trained on all labeled examples, so their scaler uses all train sensors.
    arr_final=class_arrays(d)
    test=[]
    for seed in seeds:
        print(f"final classification seed={seed}/{seeds}",flush=True)
        test.append(train_class_final(arr_final,y,lines,device,seed,epochs,root/"models"/f"class_final_{seed}.pt"))
    pred=np.stack(test).mean(0).argmax(1)
    pd.DataFrame({"id_unit":d["test_meta"]["id_unit"],"kelas":np.asarray(LABELS)[pred]}).to_csv(root.parent/"prediksi_klasifikasi.csv",index=False)
    # Evaluate the final late model on validation rows with versus without sensor values.
    lm,_=fit_class("late",arr,y,tr,va,lines,device,7,epochs,0.,False)
    m=~arr[2][va].astype(bool)
    missing_eval={"all_missing_rows":int(m.sum()),"all_missing_macro_f1":None,"recorded_rows_macro_f1":None}
    if m.any():
        missing_eval["all_missing_macro_f1"]=float(classification_report(y[va][m],lm.argmax(1)[m],labels=range(7),output_dict=True,zero_division=0)["macro avg"]["f1-score"])
    if (~m).any():
        missing_eval["recorded_rows_macro_f1"]=float(classification_report(y[va][~m],lm.argmax(1)[~m],labels=range(7),output_dict=True,zero_division=0)["macro avg"]["f1-score"])
    final_labels=np.asarray(LABELS)[pred]
    return {"validation":results,"learning_curves":curves,"final_seeds":seeds,
            "one_batch_overfit_accuracy":sanity_acc,"one_batch_overfit_steps":200,
            "missing_sensor_validation":missing_eval,
            "final_test_prediction_counts":pd.Series(final_labels).value_counts().to_dict(),
            "possible_label_review_cases":suspected,
            "missing_sensor_fraction":float(1-arr[2].mean()),
            "normalization":"validation scaler fit on training-fold sensor values only; final scaler fit on all labeled training sensors",
            "split":"stratified by class; report per production line"}

def make_windows(df,means,std,seq=32,stride=8,labeled=True,last_only=False):
    X=[]; Y=[]; G=[]
    for uid,g in df.sort_values(["id_unit","siklus"]).groupby("id_unit",sort=False):
        a=g[RUL_FEATURES].to_numpy(np.float32); mask=np.isfinite(a); a=np.where(mask,a,means)
        a=(a-means)/std; cycle=g["siklus"].to_numpy(np.float32)
        a=np.concatenate([a,(~mask).astype(np.float32),np.log1p(cycle[:,None])/8.],axis=1)
        ends=([len(g)-1] if (not labeled or last_only) else list(range(0,len(g),stride)))
        if labeled and ends[-1]!=len(g)-1: ends.append(len(g)-1)
        target=np.minimum(150.,np.maximum(0.,cycle[-1]-cycle)) if labeled else None
        for end in ends:
            w=a[max(0,end-seq+1):end+1]
            if len(w)<seq: w=np.pad(w,((seq-len(w),0),(0,0)))
            X.append(w); G.append(uid)
            if labeled:Y.append(target[end])
    return (np.asarray(X,np.float32),np.asarray(Y,np.float32),np.asarray(G)) if labeled else (np.asarray(X,np.float32),np.asarray(G))

def make_truncated_endpoints(df,means,std,test_endpoint_cycles,seed=44,seq=32):
    """Simulate one hidden-test endpoint per run-to-failure machine.

    The cutoff distribution comes from observed test history lengths. Features
    stop at the cutoff; only the later failure cycle is used to form validation
    targets. Each machine contributes one prediction, like the test task.
    """
    rng=np.random.default_rng(seed); X=[];Y=[];G=[]
    lengths=np.asarray(test_endpoint_cycles,dtype=np.int32)
    for uid,g in df.sort_values(["id_unit","siklus"]).groupby("id_unit",sort=False):
        cycles=g["siklus"].to_numpy(np.int32); failure_cycle=int(cycles[-1])
        candidates=lengths[lengths<failure_cycle]
        cutoff=int(rng.choice(candidates)) if len(candidates) else max(1,failure_cycle-1)
        end=int(np.searchsorted(cycles,cutoff,side="right")-1)
        end=max(0,end); observed_cycle=int(cycles[end])
        a=g[RUL_FEATURES].to_numpy(np.float32);mask=np.isfinite(a);a=np.where(mask,a,means)
        a=(a-means)/std;a=np.concatenate([a,(~mask).astype(np.float32),np.log1p(cycles[:,None])/8.],axis=1)
        w=a[max(0,end-seq+1):end+1]
        if len(w)<seq:w=np.pad(w,((seq-len(w),0),(0,0)))
        X.append(w);Y.append(min(150.,max(0.,failure_cycle-observed_cycle)));G.append(uid)
    return np.asarray(X,np.float32),np.asarray(Y,np.float32),np.asarray(G)

class SeqNet(nn.Module):
    def __init__(self,kind,dim):
        super().__init__(); self.kind=kind
        if kind=="gru": self.body=nn.GRU(dim,96,num_layers=2,batch_first=True,dropout=.15)
        else:self.body=nn.Sequential(nn.Conv1d(dim,64,5,padding=2),nn.GELU(),nn.Conv1d(64,96,5,padding=4,dilation=2),nn.GELU(),nn.Conv1d(96,96,3,padding=2,dilation=2),nn.GELU())
        self.head=nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.Dropout(.2),nn.Linear(64,1),nn.Sigmoid())
    def forward(self,x):
        if self.kind=="gru": z,_=self.body(x); z=z[:,-1]
        else:z=self.body(x.transpose(1,2))[:,:,-1]
        return 150*self.head(z).squeeze(-1)

def train_seq(kind,X,y,tr,va,device,seed,epochs,save=None):
    seed_all(seed); net=SeqNet(kind,X.shape[-1]).to(device); opt=torch.optim.AdamW(net.parameters(),lr=7e-4,weight_decay=1e-4)
    loader=DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X[tr]),torch.from_numpy(y[tr])),batch_size=256,shuffle=True)
    best=float("inf"); state=None; hist=[]
    for ep in range(epochs):
        net.train(); losses=[]
        for x,t in loader:
            x,t=x.to(device),t.to(device); opt.zero_grad(set_to_none=True); loss=F.smooth_l1_loss(net(x),t,beta=10); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),2); opt.step(); losses.append(loss.item())
        net.eval()
        with torch.no_grad(): p=net(torch.from_numpy(X[va]).to(device)).cpu().numpy()
        rmse=mean_squared_error(y[va],p)**.5; hist.append({"epoch":ep+1,"loss":float(np.mean(losses)),"val_rmse":float(rmse)})
        print(f"RUL {kind} seed={seed} epoch={ep+1}/{epochs} val_rmse={rmse:.3f}",flush=True)
        if rmse<best: best=rmse; state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    net.load_state_dict(state)
    if save:torch.save(state,save)
    with torch.no_grad():p=net(torch.from_numpy(X[va]).to(device)).cpu().numpy()
    return p,hist

def predict_seq_checkpoint(kind,path,X,device):
    net=SeqNet(kind,X.shape[-1]).to(device)
    net.load_state_dict(torch.load(path,map_location=device,weights_only=True));net.eval();out=[]
    with torch.no_grad():
        for (xb,) in DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X)),batch_size=256):
            out.extend(net(xb.to(device)).cpu().numpy())
    return np.asarray(out)

def asym(y,p):
    d=np.asarray(p)-np.asarray(y); return float(np.where(d<0,np.exp(np.clip(-d/13,-50,50))-1,np.exp(np.clip(d/10,-50,50))-1).sum())

def rul_scores(y,p):
    bins={}
    for lo,hi,name in [(0,50,"0-49"),(50,100,"50-99"),(100,151,"100-150")]:
        m=(y>=lo)&(y<hi)
        if m.any():bins[name]={"n":int(m.sum()),"rmse":float(mean_squared_error(y[m],p[m])**.5),"asymmetric_score":asym(y[m],p[m])}
    return {"rmse":float(mean_squared_error(y,p)**.5),"asymmetric_score":asym(y,p),"by_rul_range":bins}

def task_rul(d,root,device,epochs,seeds):
    df=d["rul_train"].copy()
    machines=df["id_unit"].drop_duplicates().to_numpy()
    spl=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42)
    train_machine_idx,held_machine_idx=next(spl.split(machines,groups=machines))
    train_machines=machines[train_machine_idx]; held_machines=machines[held_machine_idx]
    # Split held-out machines into calibration and evaluation groups.
    split_cal=GroupShuffleSplit(n_splits=1,test_size=.5,random_state=43)
    cal_idx,val_idx=next(split_cal.split(held_machines,groups=held_machines))
    cal_machines=held_machines[cal_idx];val_machines=held_machines[val_idx]
    # Fit preprocessing on training machines only; calibration/evaluation machines stay unseen.
    train_rows=df[df["id_unit"].isin(train_machines)]
    means=train_rows[RUL_FEATURES].mean().to_numpy(np.float32)
    std=train_rows[RUL_FEATURES].std().fillna(1).to_numpy(np.float32); std[std<1e-5]=1
    Xtr,ytr,_=make_windows(df[df["id_unit"].isin(train_machines)],means,std,labeled=True)
    test_lengths=d["rul_test"].groupby("id_unit")["siklus"].max().to_numpy()
    Xcal,ycal,_=make_truncated_endpoints(df[df["id_unit"].isin(cal_machines)],means,std,test_lengths,seed=43)
    Xva,yv,_=make_truncated_endpoints(df[df["id_unit"].isin(val_machines)],means,std,test_lengths,seed=44)
    X=np.concatenate([Xtr,Xcal,Xva]);y=np.concatenate([ytr,ycal,yv])
    tr=np.zeros(len(X),dtype=bool);tr[:len(Xtr)]=True
    cal=np.zeros(len(X),dtype=bool);cal[len(Xtr):len(Xtr)+len(Xcal)]=True
    va=np.zeros(len(X),dtype=bool);va[len(Xtr)+len(Xcal):]=True
    pg,hg=train_seq("gru",X,y,tr,va,device,7,epochs,root/"models"/"rul_gru.pt")
    pt,ht=train_seq("tcn",X,y,tr,va,device,7,epochs,root/"models"/"rul_tcn.pt")
    # Non-neural baseline uses current readings and trailing-window summaries.
    Z=np.concatenate([X[:,-1],X.mean(1),X.std(1)],axis=1)
    base=HistGradientBoostingRegressor(max_iter=160,l2_regularization=1.,random_state=7).fit(Z[tr],y[tr])
    pb=np.clip(base.predict(Z[va]),0,150)
    ens=[];cal_ens=[]
    for seed in seeds:
        p,_=train_seq("gru",X,y,tr,va,device,seed,epochs,root/"models"/f"rul_val_{seed}.pt");ens.append(p)
        cal_ens.append(predict_seq_checkpoint("gru",root/"models"/f"rul_val_{seed}.pt",X[cal],device))
    pv=np.mean(ens,axis=0)
    cal_pred=np.mean(cal_ens,axis=0)
    abs_residual=np.abs(ycal-cal_pred)
    q_index=min(1.,np.ceil(.9*(len(abs_residual)+1))/len(abs_residual))
    q=float(np.quantile(abs_residual,q_index,method="higher"))
    lo_eval=np.clip(pv-q,0,150);hi_eval=np.clip(pv+q,0,150)
    coverage=float(np.mean((yv>=lo_eval)&(yv<=hi_eval)))
    interval_width=float(np.mean(hi_eval-lo_eval))
    # Refit final models on all labeled machines, then transform test using train-only statistics.
    full_means=df[RUL_FEATURES].mean().to_numpy(np.float32)
    full_std=df[RUL_FEATURES].std().fillna(1).to_numpy(np.float32);full_std[full_std<1e-5]=1
    Xfull,yfull,_=make_windows(df,full_means,full_std,labeled=True)
    test,ids=make_windows(d["rul_test"],full_means,full_std,labeled=False)
    final=[]
    for seed in seeds:
        seed_all(seed);net=SeqNet("gru",Xfull.shape[-1]).to(device);opt=torch.optim.AdamW(net.parameters(),lr=7e-4,weight_decay=1e-4)
        loader=DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(Xfull),torch.from_numpy(yfull)),batch_size=256,shuffle=True)
        for _ in range(epochs):
            net.train()
            for xb,yb in loader:
                xb,yb=xb.to(device),yb.to(device);opt.zero_grad(set_to_none=True);loss=F.smooth_l1_loss(net(xb),yb,beta=10);loss.backward();opt.step()
        torch.save(net.state_dict(),root/"models"/f"rul_final_{seed}.pt");net.eval();pr=[]
        with torch.no_grad():
            for (xb,) in DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(test)),batch_size=256):pr.extend(net(xb.to(device)).cpu().numpy())
        final.append(pr)
    p=np.clip(np.mean(final,axis=0),0,150)
    pd.DataFrame({"id_unit":ids,"rul":p}).to_csv(root.parent/"prediksi_rul.csv",index=False)
    intervals=pd.DataFrame({"id_unit":ids,"rul_mean":p,
        "rul_lower_90":np.clip(p-q,0,150),
        "rul_upper_90":np.clip(p+q,0,150),
        "seed_std":np.std(np.stack(final),axis=0)})
    intervals.to_csv(root/"rul_prediction_intervals.csv",index=False)
    return {"validation":{"gru":rul_scores(yv,pg),"tcn":rul_scores(yv,pt),"hist_gradient_boosting":rul_scores(yv,pb),"gru_ensemble":rul_scores(yv,pv)},
            "uncertainty":{"calibration_residual_quantile":q,"validation_90pct_interval_coverage":coverage,
                           "mean_interval_width":interval_width,
            "method":"split conformal absolute-residual interval; calibration and validation machines are disjoint; each endpoint is truncated using a sampled hidden-test history length"},
            "learning_curves":{"gru":hg,"tcn":ht},"final_seeds":seeds,
            "label_definition":"min(150, max(0, failure_cycle - observed_cycle)); validation and calibration endpoints are truncated to lengths sampled from the hidden-test history-length distribution; split grouped by machine"}

class AE(nn.Module):
    def __init__(self):
        super().__init__();self.ie=nn.Sequential(nn.Linear(4096,512),nn.GELU(),nn.Linear(512,128),nn.GELU(),nn.Linear(128,32))
        self.id=nn.Sequential(nn.Linear(32,128),nn.GELU(),nn.Linear(128,512),nn.GELU(),nn.Linear(512,4096),nn.Sigmoid())
        self.se=nn.Sequential(nn.Linear(128,96),nn.GELU(),nn.Linear(96,32));self.sd=nn.Sequential(nn.Linear(32,96),nn.GELU(),nn.Linear(96,128))
    def forward(self,i,s):
        a=self.ie(i);b=self.se(s);return self.id(a),self.sd(b),a,b

class DEC(nn.Module):
    def __init__(self,dim,k):
        super().__init__();self.project=nn.Sequential(nn.Linear(dim,64),nn.GELU(),nn.Linear(64,32));self.centers=nn.Parameter(torch.randn(k,32)*.03)
    def forward(self,x):
        z=self.project(x);q=(1+torch.cdist(z,self.centers).square()).pow(-1);return q/q.sum(1,keepdim=True)

def task_cluster(d,root,device,epochs):
    seed_all(7)
    im=d["cluster"]["gambar"].astype(np.float32).reshape(-1,4096)/255.;sp=StandardScaler().fit_transform(d["cluster"]["spektrum"].astype(np.float32))
    ti=torch.from_numpy(im);ts=torch.from_numpy(sp);net=AE().to(device);loader=DataLoader(torch.utils.data.TensorDataset(ti,ts),batch_size=256,shuffle=True)
    opt=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=1e-5);curve=[]
    for _ in range(epochs):
        net.train();losses=[]
        for i,s in loader:
            i,s=i.to(device),s.to(device)
            # Rotation/reflection invariance for micrographs; small shifts/noise for spectra.
            img=i.reshape(-1,64,64)
            flips=torch.rand(len(img),device=device)
            img=torch.stack([torch.flip(q,(1,)) if flips[n]<.33 else (torch.flip(q,(0,)) if flips[n]<.66 else q) for n,q in enumerate(img)])
            i_aug=img.reshape(-1,4096)
            shifts=torch.randint(-2,3,(1,),device=device).item()
            s_aug=torch.roll(s,shifts=shifts,dims=1)+torch.randn_like(s)*.015
            opt.zero_grad(set_to_none=True);ri,rs,_,_=net(i_aug,s_aug)
            loss=F.mse_loss(ri,i)+F.mse_loss(rs,s);loss.backward();opt.step();losses.append(loss.item())
        curve.append(float(np.mean(losses)))
        print(f"clustering autoencoder epoch={len(curve)}/{epochs} loss={curve[-1]:.5f}",flush=True)
    net.eval();ei=[];es=[]
    with torch.no_grad():
        for i,s in DataLoader(torch.utils.data.TensorDataset(ti,ts),batch_size=512):
            _,_,a,b=net(i.to(device),s.to(device));ei.append(a.cpu().numpy());es.append(b.cpu().numpy())
    e1=StandardScaler().fit_transform(np.concatenate(ei));e2=StandardScaler().fit_transform(np.concatenate(es));z=np.concatenate([e1,e2],axis=1)
    rng=np.random.default_rng(7);sample=rng.choice(len(z),min(5000,len(z)),replace=False);crit=[]
    for k in range(2,13):
        km=KMeans(n_clusters=k,n_init=10,random_state=7).fit(z[sample])
        crit.append({"k":k,"silhouette":float(silhouette_score(z[sample],km.labels_)),"davies_bouldin":float(davies_bouldin_score(z[sample],km.labels_))})
    k=max(crit,key=lambda r:r["silhouette"])["k"];km=KMeans(n_clusters=k,n_init=20,random_state=7).fit(z);lk=km.labels_
    km_image=KMeans(n_clusters=k,n_init=10,random_state=7).fit(e1[sample])
    km_spectrum=KMeans(n_clusters=k,n_init=10,random_state=7).fit(e2[sample])
    dec=DEC(z.shape[1],k).to(device);tz=torch.from_numpy(z).to(device)
    with torch.no_grad():
        proj=dec.project(tz).cpu().numpy();c=KMeans(n_clusters=k,n_init=10,random_state=7).fit(proj).cluster_centers_;dec.centers.copy_(torch.from_numpy(c).to(device))
    opt=torch.optim.Adam(dec.parameters(),lr=5e-4)
    for _ in range(max(10,epochs)):
        with torch.no_grad():
            q=dec(tz);f=q.sum(0);p=(q*q/f);p=p/p.sum(1,keepdim=True)
        opt.zero_grad(set_to_none=True);q=dec(tz);loss=F.kl_div(q.clamp_min(1e-8).log(),p,reduction="batchmean");loss.backward();opt.step()
    with torch.no_grad():ld=dec(tz).argmax(1).cpu().numpy()
    torch.save(net.state_dict(),root/"models"/"cluster_ae.pt");torch.save(dec.state_dict(),root/"models"/"cluster_dec.pt")
    np.save(root/"cluster_kmeans.npy",lk);np.save(root/"cluster_dec.npy",ld)
    pd.DataFrame({"id_sampel":d["cluster_meta"]["id_sampel"],"cluster":ld.astype(int)}).to_csv(root.parent/"cluster.csv",index=False)
    ts=TSNE(n_components=2,init="pca",learning_rate="auto",random_state=7,perplexity=30).fit_transform(z[sample[:min(2500,len(sample))]])
    fig,ax=plt.subplots(figsize=(8,6));sc=ax.scatter(ts[:,0],ts[:,1],c=ld[sample[:len(ts)]],s=5,cmap="tab20")
    ax.set_title("t-SNE — fused image/spectrum embeddings, DEC labels");fig.colorbar(sc,ax=ax,label="cluster")
    fig.tight_layout();fig.savefig(root/"cluster_tsne.png",dpi=160);plt.close(fig)
    return {"k_criteria":crit,"selected_k":int(k),"kmeans_silhouette":float(silhouette_score(z[sample],lk[sample])),
            "dec_silhouette":float(silhouette_score(z[sample],ld[sample])),
            "image_only_silhouette":float(silhouette_score(e1[sample],km_image.labels_)),
            "spectrum_only_silhouette":float(silhouette_score(e2[sample],km_spectrum.labels_)),
            "autoencoder_loss":curve,"embedding_plot":"artifacts/cluster_tsne.png","ari":None,
            "ari_note":"Hidden material reference labels are not included. Microscope ID is metadata, not a material label."}

def data_overview(d):
    tm,te=d["train_meta"],d["test_meta"];rt,re=d["rul_train"],d["rul_test"]
    sensor=d["class_train"]["sensor"]
    rs=rt[SENSORS];q1=rs.quantile(.25);q3=rs.quantile(.75);iqr=(q3-q1).replace(0,np.nan)
    outlier_mask=(rs<q1-3*iqr)|(rs>q3+3*iqr)
    outlier=(outlier_mask.sum()/rs.notna().sum()).fillna(0)
    return {
        "classification":{"train_n":len(tm),"test_n":len(te),"image_train_shape":list(d["class_train"]["gambar"].shape),
            "signal_shape":list(sensor.shape),"classes":tm["kelas"].value_counts().to_dict(),
            "train_line_fraction":tm["lini"].value_counts(normalize=True).round(4).to_dict(),
            "test_line_fraction":te["lini"].value_counts(normalize=True).round(4).to_dict(),
            "class_by_line":pd.crosstab(tm["lini"],tm["kelas"]).to_dict(),
            "train_shift_fraction":tm["shift"].value_counts(normalize=True).round(4).to_dict(),
            "test_shift_fraction":te["shift"].value_counts(normalize=True).round(4).to_dict(),
            "signal_nan_fraction":float(np.isnan(sensor).mean()),
            "rows_with_missing_signal_fraction":float(np.isnan(sensor).any((1,2)).mean()),
            "all_signal_missing_rows":int(np.isnan(sensor).all((1,2)).sum())},
        "rul":{"train_machines":int(rt["id_unit"].nunique()),"test_machines":int(re["id_unit"].nunique()),
            "train_rows":len(rt),"test_rows":len(re),"train_missing_value_fraction":float(rt.isna().to_numpy().mean()),
            "cycle_quantiles":rt.groupby("id_unit")["siklus"].max().quantile([0,.25,.5,.75,1]).to_dict(),
            "sensor_std":rt[SENSORS].std().round(3).to_dict(),
            "sensor_robust_outlier_fraction_per_channel":outlier.round(5).to_dict(),
            "setting_sensor_correlations":rt[["setel_1","setel_2","setel_3"]+SENSORS].corr().loc[["setel_1","setel_2","setel_3"],SENSORS].round(3).to_dict()},
        "clustering":{"n":len(d["cluster_meta"]),"image_shape":list(d["cluster"]["gambar"].shape),
            "spectrum_shape":list(d["cluster"]["spektrum"].shape),"microscope_counts":d["cluster_meta"]["mikroskop"].value_counts().to_dict()}}

def make_eda(d,root):
    meta=d["train_meta"]; images=d["class_train"]["gambar"]; signals=d["class_train"]["sensor"].astype(np.float32)
    fig,axs=plt.subplots(2,7,figsize=(14,5))
    for j,label in enumerate(LABELS):
        i=int(np.flatnonzero(meta["kelas"].to_numpy()==label)[0])
        axs[0,j].imshow(images[i],cmap="gray");axs[0,j].set_title(label,fontsize=8);axs[0,j].axis("off")
        sig=signals[i]
        for ch in range(3): axs[1,j].plot(sig[ch],lw=.6,label=f"ch{ch+1}" if j==0 else None)
        axs[1,j].set_xticks([]);axs[1,j].tick_params(labelsize=6)
    axs[1,0].legend(fontsize=6);fig.suptitle("Contoh train: satu citra dan sinyal per kelas",fontsize=12)
    fig.tight_layout();fig.savefig(root/"eda_class_examples.png",dpi=160);plt.close(fig)
    cm=meta["kelas"].value_counts().reindex(LABELS)
    fig,ax=plt.subplots(figsize=(9,4));cm.plot.bar(ax=ax);ax.set_title("Distribusi kelas train");ax.set_ylabel("jumlah")
    fig.tight_layout();fig.savefig(root/"eda_class_distribution.png",dpi=160);plt.close(fig)
    cimg=d["cluster"]["gambar"];spec=d["cluster"]["spektrum"].astype(np.float32)
    fig,axs=plt.subplots(2,6,figsize=(12,4))
    picks=np.linspace(0,len(cimg)-1,6,dtype=int)
    for j,i in enumerate(picks):
        axs[0,j].imshow(cimg[i],cmap="gray");axs[0,j].set_title(str(d["cluster_meta"].iloc[i]["mikroskop"]),fontsize=8);axs[0,j].axis("off")
        axs[1,j].plot(spec[i],lw=.8);axs[1,j].set_xticks([]);axs[1,j].tick_params(labelsize=6)
    fig.suptitle("Material samples: micrograph and composition spectrum")
    fig.tight_layout();fig.savefig(root/"eda_cluster_examples.png",dpi=160);plt.close(fig)

def write_pdf(metrics,path,root):
    pages=[("Laporan UTS Deep Learning","Kelompok: isi nama anggota sebelum pengumpulan.\nAngka performa berasal dari validasi; label test tersembunyi tidak digunakan.\n\nPipeline: CNN gambar + 1D-CNN sensor; GRU/TCN RUL; autoencoder multimodal + DEC."),
           ("Eksplorasi data",json.dumps(metrics["data_overview"],ensure_ascii=False,indent=2)[:4500]),
           ("Task 1 — Klasifikasi","Split stratified 80/20. Normalisasi sensor dihitung dari train; nilai hilang diimputasi nol setelah standardisasi dan mask kanal ditambahkan. Dibandingkan image-only, sensor-only, early fusion, late fusion, label smoothing, dan line-balanced sampling.\n\n"+json.dumps(metrics["classification"],ensure_ascii=False,indent=2)[:4000]),
           ("Task 2 — RUL","Label = min(150, max(0, siklus akhir mesin − siklus saat ini)). Validasi dipisah berdasarkan mesin. Sensor dinormalisasi dari train dan missingness diwakili mask kanal. Interval memakai ensemble tiga seed; bila interval melintasi ambang servis, pemeriksaan operator lebih aman daripada menunggu prediksi titik.\n\n"+json.dumps(metrics["rul"],ensure_ascii=False,indent=2)[:3500]),
           ("Task 3 — Clustering","Autoencoder mempelajari kedua modalitas. Augmentasi gambar berupa flip horizontal/vertikal; spektrum diberi shift kecil dan noise untuk ketahanan pada distorsi. Jumlah cluster dibandingkan dengan silhouette dan Davies–Bouldin; t-SNE tersimpan di artifacts/cluster_tsne.png. ARI tidak bisa dihitung karena referensi material tersembunyi tidak disertakan.\n\n"+json.dumps(metrics["clustering"],ensure_ascii=False,indent=2)[:2500])]
    pages.append(("Visual inspeksi modalitas","Contoh gambar dan sinyal untuk tiap kelas serta contoh gambar mikrostruktur dan spektrum komposisi."))
    with PdfPages(path) as pdf:
        for title,text in pages:
            fig=plt.figure(figsize=(8.27,11.69));fig.text(.08,.95,title,fontsize=17,weight="bold",va="top")
            body="\n".join(textwrap.fill(line,width=94) for line in text.splitlines())
            fig.text(.08,.90,body,fontsize=8,family="monospace",va="top")
            if title=="Eksplorasi data":
                img=plt.imread(root/"eda_class_distribution.png")
                ax=fig.add_axes([.10,.05,.80,.28]);ax.imshow(img);ax.axis("off")
            if title.startswith("Task 3"):
                img=plt.imread(root/"cluster_tsne.png")
                ax=fig.add_axes([.10,.05,.80,.28]);ax.imshow(img);ax.axis("off")
            if title=="Visual inspeksi modalitas":
                img=plt.imread(root/"eda_class_examples.png")
                ax=fig.add_axes([.04,.12,.92,.66]);ax.imshow(img);ax.axis("off")
                img2=plt.imread(root/"eda_cluster_examples.png")
                ax2=fig.add_axes([.14,.02,.72,.22]);ax2.imshow(img2);ax2.axis("off")
            pdf.savefig(fig);plt.close(fig)

def main():
    from kode.reporting import write_pdf as render_pdf
    from kode.docx_reporting import write_docx as render_docx
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--out",type=Path,default=Path("."))
    p.add_argument("--epochs-class",type=int,default=10);p.add_argument("--epochs-rul",type=int,default=12);p.add_argument("--epochs-cluster",type=int,default=20)
    p.add_argument("--seeds",type=int,nargs="+",default=[7,21,42]);a=p.parse_args()
    if not a.data.is_file():raise FileNotFoundError(a.data)
    root=a.out.resolve();root.mkdir(parents=True,exist_ok=True);art=root/"artifacts";art.mkdir(exist_ok=True);(art/"models").mkdir(exist_ok=True)
    device="cuda" if torch.cuda.is_available() else "cpu";print("device:",device)
    d=load_data(a.data)
    make_eda(d,art)
    metrics={"runtime":{"device":device,"python":sys.version.split()[0],"torch":torch.__version__,
             "numpy":np.__version__,"pandas":pd.__version__,"scikit_learn":sklearn.__version__,
             "matplotlib":matplotlib.__version__,"seeds":a.seeds},
             "data_overview":data_overview(d),
             "classification":task_classification(d,art,device,a.epochs_class,a.seeds),
             "rul":task_rul(d,art,device,a.epochs_rul,a.seeds),
             "clustering":task_cluster(d,art,device,a.epochs_cluster)}
    (art/"metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding="utf-8")
    render_pdf(metrics,root/"laporan.pdf",art)
    try:
        render_docx(metrics,root/"laporan.docx",art)
        print("Generated editable laporan.docx")
    except ImportError as exc:
        print(f"Editable Word report was skipped because python-docx is unavailable: {exc}")
    print("Generated laporan.pdf, prediksi_klasifikasi.csv, prediksi_rul.csv, cluster.csv")
if __name__=="__main__":main()
