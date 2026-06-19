import os,sys,json,time
sys.path.insert(0,'.')
import numpy as np
from data import register_runtime_sample
from gui_app.nmf_panel import build_nmf_input, fit_nmf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
OUT="docs/explainer/figs"; RUN="runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60"
path=r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy"; vmax=5.0; Ny,Nx=128,128; N=Ny*Nx
asg=np.load(os.path.join(RUN,"eval","inference.npz"))["assigns"].astype(int); K=int(len(np.unique(asg)))
key=register_runtime_sample(path,scan_shape=(Ny,Nx),vmax=vmax)
def bmean3(a,f): n,H,W=a.shape; s=H//f; return a.reshape(n,s,f,s,f).mean((2,4))
def aug(Xs): s=Xs.shape[1]; return np.concatenate([np.roll(Xs,sh,1) for sh in (0,s//4,s//2,3*s//4)],0).reshape(-1,s*s).astype(np.float32)
X,_,cs,_=build_nmf_input(key,dict(input="polar",log=False,sparse=False,theta_shift=False),vmax_override=vmax)
P=cs[0]; Xd=bmean3(np.clip(X.reshape(N,P,P),0,None).astype(np.float32),4); del X
W,H,e=fit_nmf(Xd.reshape(N,-1),aug(Xd),n_components=min(2*K,30),max_iter=300)
lab=KMeans(K,n_init=8,random_state=0).fit_predict(StandardScaler().fit_transform(W))
ari=float(adjusted_rand_score(asg,lab)); ami=float(adjusted_mutual_info_score(asg,lab))
j=json.load(open(os.path.join(OUT,"polar_nmf_vs_dino.json")))
j["IMC_SI5"]=dict(K=K,N=N,ncomp=min(2*K,30),ARI_new=round(ari,3),AMI_new=round(ami,3))
json.dump(j,open(os.path.join(OUT,"polar_nmf_vs_dino.json"),"w"),indent=2)
np.save(os.path.join(OUT,"polarnmf_labels_IMC_SI5.npy"),lab.reshape(Ny,Nx))
print(f"IMC_SI5: K={K} ARI={ari:.3f} AMI={ami:.3f}",flush=True)
