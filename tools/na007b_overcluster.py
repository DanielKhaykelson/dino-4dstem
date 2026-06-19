"""Na007b over/under-clustering analysis: DINO (auto K via emergent prototypes)
vs NMF+auto-K (silhouette). Tests: (1) are DINO classes mutually diffraction-
distinct? (2) within one NMF cluster, are the DINO classes distinct (NMF under-
clustered)? (3) crystallinity per class. Full-res NMF (central 256, n_comp=30)."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import open_lazy_cube
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
F=r"docs/explainer/figs"; CUBE=r"D:/DINOSR/data/Na007b_nbed.cube.npy"; RUN=r"runs/_gui/Na007b_k60_m097_vmax2"
CROP=256; NCOMP=30
asg=np.load(os.path.join(RUN,"eval","inference.npz"),allow_pickle=True)["assigns"].astype(int)
Ny,Nx=tuple(json.load(open(os.path.join(RUN,"_train_kwargs.json")))["_sample_config"]["scan_shape"])
Kd=int(asg.max())+1; flake=(asg!=0)
cube=open_lazy_cube(CUBE,scan_shape=(Ny,Nx)); _,_,H,W=cube.shape
a0=(H-CROP)//2; a1=a0+CROP; cyx=(H-1)/2.0
yy,xx=np.indices((H,W)); rr=np.sqrt((yy-cyx)**2+(xx-cyx)**2); beam=max(8,round(0.11*H)); post=rr>=beam
nb=H//2; lo=max(int(0.10*nb),beam+1); hi=int(0.90*nb)
# pass 1: NMF matrix + DINO per-class sums + crystallinity factors
print("pass1...",flush=True); t0=time.time()
X=np.zeros((Ny*Nx,CROP*CROP),np.float32); dsum=np.zeros((Kd,H,W)); dcnt=np.zeros(Kd)
ph=np.full(Ny*Nx,np.nan); av=np.full(Ny*Nx,np.nan); sc=np.zeros(Ny*Nx)
for rx in range(Ny):
    blk=np.asarray(cube[rx],np.float32)
    for ry in range(Nx):
        i=rx*Nx+ry; pat=blk[ry]
        X[i]=np.log1p(np.clip(pat[a0:a1,a0:a1],0,None)).ravel()
        sc[i]=pat[post].sum()
        m,v,_=_radial_mean_var(pat,(cyx,cyx),beam_px=beam); seg=m[lo:hi]
        if seg.size and seg.sum()>0:
            halo=np.exp(_snip_baseline(np.log(np.clip(seg,1e-6,None))))
            ph[i]=np.clip(seg-halo,0,None).sum()/(seg.sum()+1e-9); av[i]=np.mean(v[lo:hi]/(seg*seg+1e-9))
        c=asg[i]; dsum[c]+=pat; dcnt[c]+=1
    if rx%30==0: print(f"  row {rx} ({time.time()-t0:.0f}s)",flush=True)
print(f"  NMF.fit (n_comp={NCOMP})...",flush=True); t0=time.time()
Wd=NMF(n_components=NCOMP,init="nndsvda",max_iter=400,random_state=0,tol=1e-4).fit_transform(np.clip(X,0,None))
print(f"  NMF.fit {time.time()-t0:.0f}s; auto-K silhouette...",flush=True)
Ws=StandardScaler().fit_transform(Wd); best=(-1,None,None)
for k in range(3,19):
    lab=KMeans(k,n_init=8,random_state=0).fit_predict(Ws)
    s=silhouette_score(Ws,lab,sample_size=4000,random_state=0)
    if s>best[0]: best=(s,k,lab)
sil,Kn,nmf_lab=best
print(f"  NMF auto-K = {Kn} (silhouette {sil:.3f}); DINO active K = {Kd}",flush=True)
# pass 2: NMF per-cluster sums
print("pass2 (NMF cluster means)...",flush=True); t0=time.time()
nsum=np.zeros((Kn,H,W)); ncnt=np.zeros(Kn); nl=nmf_lab.reshape(Ny,Nx)
for rx in range(Ny):
    blk=np.asarray(cube[rx],np.float32)
    for ry in range(Nx):
        c=nl[rx,ry]; nsum[c]+=blk[ry]; ncnt[c]+=1
    if rx%40==0: print(f"  row {rx} ({time.time()-t0:.0f}s)",flush=True)
dmean=np.array([dsum[c]/max(dcnt[c],1) for c in range(Kd)])
nmean=np.array([nsum[c]/max(ncnt[c],1) for c in range(Kn)])
np.save(f"{F}/na007b_dino_classmeans.npy",dmean); np.save(f"{F}/na007b_nmf_classmeans.npy",nmean)
np.save(f"{F}/na007b_nmf_autolabels.npy",nl); np.save(f"{F}/na007b_cryst.npz.npy",np.vstack([ph,av,sc]))
# distinctness (beam-masked ring corr)
ring=(rr>0.18*nb)&(rr<0.95*nb)
def disp(x): x=np.log1p(np.clip(x,0,None)); v=x[ring]; return (x-v.min())/(v.ptp()+1e-9)
Dd=[disp(m) for m in dmean]; Dn=[disp(m) for m in nmean]
def r(a,b): return float(np.corrcoef(a[ring],b[ring])[0,1])
# test1: DINO pairwise r
import itertools
rmat=np.eye(Kd)
for a,b in itertools.combinations(range(Kd),2): rmat[a,b]=rmat[b,a]=r(Dd[a],Dd[b])
print("\n[Test1] DINO classes mutual distinctness (beam-masked r): off-diag min/median/max = "
      f"{rmat[~np.eye(Kd,dtype=bool)].min():.3f}/{np.median(rmat[~np.eye(Kd,dtype=bool)]):.3f}/{rmat[~np.eye(Kd,dtype=bool)].max():.3f}",flush=True)
redundant=[(a,b,round(rmat[a,b],3)) for a,b in itertools.combinations(range(Kd),2) if rmat[a,b]>0.97]
print(f"  DINO pairs with r>0.97 (possible over-split): {redundant}",flush=True)
# test2: within each NMF cluster, the DINO classes present + their mutual distinctness
print("\n[Test2] DINO classes inside each NMF cluster (under-clustering check):",flush=True)
ct=np.zeros((Kn,Kd),int)
for i in range(Ny*Nx):
    if flake[i]: ct[nmf_lab[i],asg[i]]+=1
for c in range(Kn):
    present=[d for d in range(Kd) if ct[c,d]>0.04*ct[c].sum() and ct[c].sum()>0]  # DINO classes >4% of cluster
    if len(present)>=2:
        rr2=[rmat[a,b] for a,b in itertools.combinations(present,2)]
        print(f"  NMF cluster {c} (n={ct[c].sum()}) contains DINO {present} -> min pairwise r={min(rr2):.3f} "
              f"({'DISTINCT->NMF under-clustered' if min(rr2)<0.9 else 'similar'})",flush=True)
    elif len(present)==1:
        print(f"  NMF cluster {c} (n={ct[c].sum()}) = DINO {present} (1:1)",flush=True)
# crystallinity per class
def med(arr,labels,K): 
    return [float(np.nanmedian(arr[labels==c])) if (labels==c).any() else np.nan for c in range(K)]
print("\n[Test3] crystallinity (peak/halo) median per DINO class:",[round(x,3) for x in med(ph,asg,Kd)],flush=True)
print("        per NMF cluster:",[round(x,3) for x in med(ph,nmf_lab,Kn)],flush=True)
# figures: per-class average grids
def grid(means,title,fn):
    K=len(means); nc=5; nr=int(np.ceil(K/nc)); fig=Figure(figsize=(2.3*nc,2.3*nr),facecolor="white")
    for c in range(K):
        ax=fig.add_subplot(nr,nc,c+1); im=np.log1p(np.clip(means[c][a0:a1,a0:a1],0,None))
        o=im[im>0]; ax.imshow(im,cmap="inferno",vmax=(np.percentile(o,99.5) if o.size else 1),interpolation="nearest",aspect="equal")
        ax.set_title(f"{c}",fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title,fontsize=12); fig.tight_layout(rect=[0,0,1,0.95]); FigureCanvasAgg(fig)
    fig.savefig(f"{F}/{fn}",dpi=140,facecolor="white",bbox_inches="tight")
grid(dmean,f"Na007b DINO per-class avg diffraction (K={Kd})","na007b_dino_classmeans.png")
grid(nmean,f"Na007b NMF per-cluster avg diffraction (auto-K={Kn})","na007b_nmf_classmeans.png")
print("\nwrote class-mean grids + arrays. DONE",flush=True)
