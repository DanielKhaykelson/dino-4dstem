"""FULL-resolution NMF+kmeans on Na007a/Na006a: central 256x256 (no downsampling),
n_components=30, proper iterations. Times cube-read / NMF.fit / KMeans separately."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import open_lazy_cube
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
F=r"docs/explainer/figs"; CROP=256; NCOMP=30
S={"Na007a":(r"D:/DINOSR/data/Na007a.prz",(126,100),12),"Na006a":(r"D:/DINOSR/data/Na006a.prz",(100,100),11)}
for nm,(cp,(Ny,Nx),K) in S.items():
    print(f"\n=== {nm} FULL NMF (central {CROP}x{CROP}, n_comp={NCOMP}, K={K}) ===",flush=True)
    cube=open_lazy_cube(cp,scan_shape=(Ny,Nx)); _,_,H,W=cube.shape
    a=(H-CROP)//2; b=a+CROP
    t0=time.time(); X=np.zeros((Ny*Nx,CROP*CROP),np.float32)
    for rx in range(Ny):
        blk=np.asarray(cube[rx],np.float32)[:,a:b,a:b]
        X[rx*Nx:(rx+1)*Nx]=np.log1p(np.clip(blk,0,None)).reshape(Nx,-1)
    t_read=time.time()-t0; print(f"  cube read+crop: {t_read:.0f}s  matrix {X.nbytes/1e9:.1f} GB",flush=True)
    t0=time.time(); Wd=NMF(n_components=NCOMP,init="nndsvda",max_iter=400,random_state=0,tol=1e-4).fit_transform(X)
    t_nmf=time.time()-t0; print(f"  NMF.fit: {t_nmf:.0f}s",flush=True)
    t0=time.time(); lab=KMeans(K,n_init=10,random_state=0).fit_predict(StandardScaler().fit_transform(Wd))
    t_km=time.time()-t0; print(f"  KMeans: {t_km:.0f}s   TOTAL NMF+kmeans = {t_read+t_nmf+t_km:.0f}s",flush=True)
    np.save(f"{F}/{nm}_nmf_labels.npy",lab.reshape(Ny,Nx)); del X
print("DONE",flush=True)
