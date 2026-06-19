"""NMF+KMeans reference on Na007b: recover the Line region, compare to SAM & DINO
(IoU + beam-masked Pearson r of region-average diffraction), build a 3-row panel.
Reference point: classical NMF MAY work on this easy sample; on hard IMC it does not."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from data import open_lazy_cube
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle
F=r"docs/explainer/figs"; PXNM=16.0; BAR=500.0; DS=40; NCOMP=20; K=12
CUBE=r"D:/DINOSR/data/Na007b_nbed.cube.npy"; RUN=r"runs/_gui/Na007b_k60_m097_vmax2"
orient=np.load(r"D:/DINOSR/NanoLetters/Figure4/ComapreToKmap/Na007b_orient_img.npy").astype(float)
asg=np.load(os.path.join(RUN,"eval","inference.npz"),allow_pickle=True)["assigns"].astype(int)
Ny,Nx=tuple(json.load(open(os.path.join(RUN,"_train_kwargs.json")))["_sample_config"]["scan_shape"])
flake=asg!=0; sam_line=np.isfinite(orient)&(orient!=0); dino_line=np.isin(asg,[1,8])
cube=open_lazy_cube(CUBE,scan_shape=(Ny,Nx)); _,_,H,W=cube.shape
cy=(H-1)/2; yy,xx=np.indices((H,W)); rr=np.sqrt((yy-cy)**2+(xx-cy)**2)
# pass 1: downsampled patterns
print("pass1 features...",flush=True); t0=time.time()
P=np.zeros((Ny*Nx,DS*DS),np.float32)
for rx in range(Ny):
    blk=np.asarray(cube[rx],np.float32)
    for ry in range(Nx):
        P[rx*Nx+ry]=cv2.resize(np.log1p(np.clip(blk[ry],0,None)),(DS,DS),interpolation=cv2.INTER_AREA).ravel()
    if rx%30==0: print(f"  row {rx} ({time.time()-t0:.0f}s)",flush=True)
W_=NMF(n_components=NCOMP,init="nndsvda",max_iter=400,random_state=0).fit_transform(np.clip(P,0,None))
lab=KMeans(K,n_init=10,random_state=0).fit_predict(StandardScaler().fit_transform(W_))
# NMF Line = clusters mostly inside SAM Line (within flake)
nmf_line=np.zeros(Ny*Nx,bool)
for c in range(K):
    cf=(lab==c)&flake
    if cf.sum()>0 and (cf&sam_line).sum()/cf.sum()>0.5: nmf_line|=(lab==c)
nmf_line&=flake; nmf_rest=flake&~nmf_line
def iou(a,b): u=(a|b).sum(); return (a&b).sum()/u if u else 0
print(f"NMF_line={int(nmf_line.sum())}  IoU(NMF,SAM)={iou(nmf_line,sam_line):.3f}  IoU(NMF,DINO)={iou(nmf_line,dino_line):.3f}",flush=True)
# pass 2: avg diffraction for NMF line/rest
print("pass2 avg...",flush=True); t0=time.time()
sL=np.zeros((H,W)); sR=np.zeros((H,W)); nL=nR=0
mL=nmf_line.reshape(Ny,Nx); mR=nmf_rest.reshape(Ny,Nx)
for rx in range(Ny):
    blk=np.asarray(cube[rx],np.float32)
    if mL[rx].any(): sL+=blk[mL[rx]].sum(0); nL+=int(mL[rx].sum())
    if mR[rx].any(): sR+=blk[mR[rx]].sum(0); nR+=int(mR[rx].sum())
    if rx%30==0: print(f"  row {rx} ({time.time()-t0:.0f}s)",flush=True)
nmfL=(sL/max(nL,1)).astype(np.float32); nmfR=(sR/max(nR,1)).astype(np.float32)
np.save(f"{F}/na007b_nmf_labels.npy",lab.reshape(Ny,Nx))
np.save(f"{F}/na007b_avg_NMF_line.npy",nmfL); np.save(f"{F}/na007b_avg_NMF_rest.npy",nmfR)
# beam-masked r vs SAM/DINO line averages
ring=(rr>0.18*min(H,W)/2)&(rr<0.98*min(H,W)/2)
def disp(a): a=np.log1p(np.clip(a,0,None)); v=a[ring]; return (a-v.min())/(v.ptp()+1e-9)
SAM=disp(np.load(f"{F}/na007b_avg_SAM_line.npy")); DINO=disp(np.load(f"{F}/na007b_avg_DINO_line(1,8).npy")); NM=disp(nmfL)
def pear(a,b): return float(np.corrcoef(a[ring],b[ring])[0,1])
print(f"beam-masked r: NMF_line vs SAM_line={pear(NM,SAM):.3f}  vs DINO_line={pear(NM,DINO):.3f}",flush=True)
# 3-row panel (SAM, DINO, NMF) on BF
bf=np.load(f"{F}/na007b_BF.npy").astype(float); g=np.log1p(np.clip(bf,0,None)); g=(g-g.min())/(g.ptp()+1e-9)
BLUE=np.array([0.13,0.45,0.95]); GRN=np.array([0.20,0.78,0.55])
def ov(lm,rm,a=0.5):
    rgb=np.dstack([g,g,g]).copy(); rgb[rm.reshape(Ny,Nx)]=(1-a)*rgb[rm.reshape(Ny,Nx)]+a*GRN
    rgb[lm.reshape(Ny,Nx)]=(1-a)*rgb[lm.reshape(Ny,Nx)]+a*BLUE; return rgb
def dd(im):
    a=np.log1p(np.clip(im,0,None)); a=a[H//2-130:H//2+130,H//2-130:H//2+130]; o=a[a>0]
    return a,(np.percentile(o,99.5) if o.size else a.max())
rows=[("SAM",sam_line,flake&~sam_line,np.load(f"{F}/na007b_avg_SAM_line.npy"),np.load(f"{F}/na007b_avg_SAM_rest.npy")),
      ("DINO",dino_line,flake&~dino_line,np.load(f"{F}/na007b_avg_DINO_line(1,8).npy"),np.load(f"{F}/na007b_avg_DINO_rest.npy")),
      ("NMF+kmeans",nmf_line,nmf_rest,nmfL,nmfR)]
fig=Figure(figsize=(11,11.5),facecolor="white"); ct=["Segmented areas on BF","Line — avg diffraction","Rest — avg diffraction"]
for ri,(rl,lm,rm,la,ra) in enumerate(rows):
    ax=fig.add_subplot(3,3,ri*3+1); ax.imshow(ov(lm,rm),interpolation="nearest",aspect="equal")
    w=BAR/PXNM; ax.add_patch(Rectangle((5,Ny-8),w,2.4,color="white",ec="black",lw=0.4))
    if ri==0: ax.text(5+w/2,Ny-9,f"{int(BAR)} nm",color="white",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.set_ylabel(rl,fontsize=15,fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
    if ri==0: ax.set_title(ct[0],fontsize=13)
    for ci,im in enumerate([la,ra]):
        ax=fig.add_subplot(3,3,ri*3+2+ci); img,vm=dd(im); ax.imshow(img,cmap="inferno",vmax=vm,interpolation="nearest",aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        if ri==0: ax.set_title(ct[1+ci],fontsize=13)
fig.suptitle("Na007b — SAM / DINO / NMF+kmeans: all recover the same Line domain (easy sample)",fontsize=14,fontweight="bold")
cap=(f"NMF+kmeans Line vs SAM: IoU={iou(nmf_line,sam_line):.2f}; vs DINO: IoU={iou(nmf_line,dino_line):.2f}. "
     f"Beam-masked Pearson r of Line avg diffraction: NMF vs SAM={pear(NM,SAM):.3f}, NMF vs DINO={pear(NM,DINO):.3f}. "
     "On this well-ordered sample classical NMF+kmeans also recovers the domain; for the hard low-dose IMC films it does not (Fig. 3, ARI 0.11-0.21).")
fig.text(0.5,0.005,cap,ha="center",va="bottom",fontsize=9.4,wrap=True)
fig.tight_layout(rect=[0,0.05,1,0.96]); FigureCanvasAgg(fig)
fig.savefig(f"{F}/na007b_3way_panel.png",dpi=150,facecolor="white",bbox_inches="tight")
print("wrote na007b_3way_panel.png + nmf labels/means",flush=True)
