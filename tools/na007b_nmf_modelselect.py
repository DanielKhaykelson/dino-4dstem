"""NMF rank/K selection + components + reconstruction for Na007b (paper M&M).
Selection criteria (4): silhouette(KMeans on loadings); reconstruction-error/
explained-variance elbow; cophenetic correlation & dispersion of the consensus
matrix over random restarts (Brunet 2004 = NMF-native stability/ambiguity).
Final NMF emits component basis patterns + spatial loadings + reconstruction."""
import os, sys, json, time, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import open_lazy_cube
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
F=r"docs/explainer/figs"; CUBE=r"D:/DINOSR/data/Na007b_nbed.cube.npy"; RUN=r"runs/_gui/Na007b_k60_m097_vmax2"
Ny,Nx=tuple(json.load(open(os.path.join(RUN,"_train_kwargs.json")))["_sample_config"]["scan_shape"])
cube=open_lazy_cube(CUBE,scan_shape=(Ny,Nx)); _,_,H,W=cube.shape; C=256; a0=(H-C)//2; a1=a0+C
def bmean(a,f): s=a.shape[0]//f; return a.reshape(s,f,s,f).mean((1,3))
print("cube pass: build 64^2 + 128^2 feature matrices...",flush=True); t0=time.time()
N=Ny*Nx; X64=np.zeros((N,64*64),np.float32); X128=np.zeros((N,128*128),np.float32)
for rx in range(Ny):
    blk=np.asarray(cube[rx],np.float32)
    for ry in range(Nx):
        cr=np.clip(blk[ry][a0:a1,a0:a1],0,None); i=rx*Nx+ry
        X64[i]=np.log1p(bmean(cr,4)).ravel(); X128[i]=np.log1p(bmean(cr,2)).ravel()
print(f"  matrices built {time.time()-t0:.0f}s",flush=True)
RANKS=[4,6,8,10,12,14,16,18,20]; NRUN=12; SUB=1800
rng=np.random.RandomState(0); sub=rng.choice(N,SUB,replace=False)
rec_err=[]; evar=[]; coph=[]; disp=[]; sil=[]
Xn=np.clip(X64,0,None); normX=np.linalg.norm(Xn)
for r in RANKS:
    Cm=np.zeros((SUB,SUB)); base=None
    for run in range(NRUN):
        m=NMF(n_components=r,init="random",max_iter=200,random_state=run,tol=1e-3)
        Wr=m.fit_transform(Xn)
        if run==0: base=(m,Wr)
        lab=Wr[sub].argmax(1); Cm+=(lab[:,None]==lab[None,:])
    Cm/=NRUN
    d=1.0-Cm; np.fill_diagonal(d,0.0); cd=squareform(d,checks=False)
    Z=linkage(cd,method="average"); coph.append(float(cophenet(Z,cd)[0]))
    disp.append(float(np.mean(4*(Cm-0.5)**2)))
    m0,W0=base; rec_err.append(float(m0.reconstruction_err_))
    evar.append(float(1-(np.linalg.norm(Xn-W0@m0.components_)**2)/(normX**2)))
    Ws=StandardScaler().fit_transform(W0)
    lab=KMeans(r,n_init=8,random_state=0).fit_predict(Ws)
    sil.append(float(silhouette_score(Ws,lab,sample_size=4000,random_state=0)))
    print(f"  rank {r}: coph={coph[-1]:.3f} disp={disp[-1]:.3f} evar={evar[-1]:.3f} sil={sil[-1]:.3f}",flush=True)
coph=np.array(coph); knee=RANKS[int(np.argmax(coph[:-1]-coph[1:]))]  # rank before largest cophenetic drop
silK=RANKS[int(np.argmax(sil))]
print(f"\n[K choice] cophenetic-knee={knee}  silhouette-max={silK}  (NMF rank choice is method/goal-dependent)",flush=True)
# selection plot
fig=Figure(figsize=(11,8),facecolor="white")
def panel(idx,y,ttl,yl,mark=None):
    ax=fig.add_subplot(2,2,idx); ax.plot(RANKS,y,"o-",color="#1C7293",lw=2)
    if mark: ax.axvline(mark,color="#C0392B",ls="--",lw=1.5,label=f"choice={mark}"); ax.legend(fontsize=9)
    ax.set_xlabel("NMF rank / K"); ax.set_ylabel(yl); ax.set_title(ttl,fontsize=11); ax.grid(alpha=.3)
panel(1,sil,"(a) Silhouette (KMeans on loadings) — weak/ambiguous","silhouette",silK)
panel(2,evar,"(b) Explained variance (reconstruction elbow)","explained var")
panel(3,coph,"(c) Cophenetic correlation (consensus stability)","cophenetic r",knee)
panel(4,disp,"(d) Dispersion coefficient (assignment ambiguity)","dispersion")
fig.suptitle(f"Na007b NMF model selection — silhouette weak (max {max(sil):.2f}); "
             f"cophenetic/dispersion = NMF-native stability",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96]); FigureCanvasAgg(fig); fig.savefig(f"{F}/na007b_nmf_modelselection.png",dpi=150,facecolor="white")
# FINAL NMF at knee rank on 128^2
print(f"\nfinal NMF rank={knee} on 128^2 ...",flush=True); t0=time.time()
X=np.clip(X128,0,None); mf=NMF(n_components=knee,init="nndsvda",max_iter=500,random_state=0,tol=1e-4)
Wf=mf.fit_transform(X); Hf=mf.components_; Xr=Wf@Hf
EV=float(1-(np.linalg.norm(X-Xr)**2)/(np.linalg.norm(X)**2))
print(f"  fit {time.time()-t0:.0f}s; explained variance={EV:.3f}",flush=True)
np.save(f"{F}/na007b_nmf_W.npy",Wf); np.save(f"{F}/na007b_nmf_H.npy",Hf)
# components: basis patterns (H) + spatial loadings (W)
nc=5; nr=int(np.ceil(knee/nc))
figc=Figure(figsize=(2.2*nc,2.2*nr),facecolor="white")
for k in range(knee):
    ax=figc.add_subplot(nr,nc,k+1); im=Hf[k].reshape(128,128)
    ax.imshow(im,cmap="inferno",interpolation="nearest"); ax.set_title(f"comp {k}",fontsize=9); ax.axis("off")
figc.suptitle(f"Na007b NMF component basis patterns (rank {knee}, EV={EV:.2f})",fontsize=12)
figc.tight_layout(rect=[0,0,1,0.95]); FigureCanvasAgg(figc); figc.savefig(f"{F}/na007b_nmf_components.png",dpi=140,facecolor="white")
figl=Figure(figsize=(2.2*nc,2.2*nr),facecolor="white")
for k in range(knee):
    ax=figl.add_subplot(nr,nc,k+1); ax.imshow(Wf[:,k].reshape(Ny,Nx),cmap="viridis",interpolation="nearest")
    ax.set_title(f"comp {k}",fontsize=9); ax.axis("off")
figl.suptitle(f"Na007b NMF spatial loadings (rank {knee})",fontsize=12)
figl.tight_layout(rect=[0,0,1,0.95]); FigureCanvasAgg(figl); figl.savefig(f"{F}/na007b_nmf_loadings.png",dpi=140,facecolor="white")
# reconstruction examples
ex=[int(0.30*N),int(0.5*N),int(0.7*N),int(0.85*N)]
figr=Figure(figsize=(9,3*len(ex)),facecolor="white")
for j,i in enumerate(ex):
    o=X[i].reshape(128,128); rc=Xr[i].reshape(128,128); res=o-rc; vm=np.percentile(o,99.5)
    for c,(img,ttl,cm,vlim) in enumerate([(o,"original",("inferno",(0,vm))),(rc,"reconstruction",("inferno",(0,vm))),(res,"residual",("RdBu_r",(-vm/2,vm/2)))]):
        ax=figr.add_subplot(len(ex),3,3*j+c+1); ax.imshow(img,cmap=cm[0],vmin=cm[1][0],vmax=cm[1][1],interpolation="nearest")
        ax.set_title(f"pos {i} {ttl}" if j==0 or c==0 else ttl,fontsize=9); ax.axis("off")
figr.suptitle(f"Na007b NMF reconstruction (rank {knee}, global EV={EV:.3f}, log1p patterns)",fontsize=12)
figr.tight_layout(rect=[0,0,1,0.96]); FigureCanvasAgg(figr); figr.savefig(f"{F}/na007b_nmf_reconstruction.png",dpi=140,facecolor="white")
print("wrote modelselection / components / loadings / reconstruction. DONE",flush=True)
