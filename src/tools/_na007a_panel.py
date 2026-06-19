import os, json, time, numpy as np
import sys; sys.path.insert(0, os.getcwd())
from data import open_lazy_cube
from skimage.filters import threshold_otsu
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle
F=r"docs/explainer/figs"; PXNM=16.0; BAR=500.0; Ny,Nx=126,100
CUBE=r"D:/DINOSR/data/Na007a.prz"
orient=np.load(r"D:/DINOSR/NanoLetters/Figure4/ComapreToKmap/Na007a_orient_img.npy").astype(float)
asg=np.load(f"{F}/Na007a_transfer_assigns.npy").ravel()
bf=np.load(f"{F}/Na007a_BF.npy").astype(float).ravel()
sam=(np.isfinite(orient)&(orient!=0))
flake=np.log1p(np.clip(bf,0,None)); flake=flake<threshold_otsu(flake)   # sample dark in BF
dino_line=(asg==1)
sam_rest=flake&~sam; dino_rest=flake&~dino_line
def iou(a,b): u=(a|b).sum(); return (a&b).sum()/u if u else 0
print(f"SAM={int(sam.sum())} DINO_class1={int(dino_line.sum())} flake={int(flake.sum())}  IoU(class1,SAM)={iou(dino_line,sam):.3f}",flush=True)
cube=open_lazy_cube(CUBE,scan_shape=(Ny,Nx)); _,_,H,W=cube.shape
cy=(H-1)/2; yy,xx=np.indices((H,W)); rr=np.sqrt((yy-cy)**2+(xx-cy)**2)
ha_m=(rr>=0.40*min(H,W)/2)&(rr<=0.98*min(H,W)/2)
masks={"SAM_line":sam,"SAM_rest":sam_rest,"DINO_line":dino_line,"DINO_rest":dino_rest}
sums={k:np.zeros((H,W)) for k in masks}; cnt={k:0 for k in masks}; ha=np.zeros(Ny*Nx); mr={k:m.reshape(Ny,Nx) for k,m in masks.items()}
t0=time.time()
for rx in range(Ny):
    blk=np.asarray(cube[rx],np.float32)
    ha[rx*Nx:(rx+1)*Nx]=blk[:,ha_m].sum(1)
    for k in masks:
        idx=mr[k][rx]
        if idx.any(): sums[k]+=blk[idx].sum(0); cnt[k]+=int(idx.sum())
    if rx%30==0: print(f"  row {rx} ({time.time()-t0:.0f}s)",flush=True)
means={k:(sums[k]/max(cnt[k],1)) for k in masks}; ha=ha.reshape(Ny,Nx)
for k in masks: np.save(f"{F}/na007a_avg_{k}.npy",means[k])
np.save(f"{F}/na007a_HAADF.npy",ha)
# panel: rows SAM/DINO, cols areas-on-HAADF / Line avg / rest avg
g=np.log1p(np.clip(ha,0,None)); g=(g-g.min())/(g.ptp()+1e-9)   # HAADF: sample bright
BLUE=np.array([0.13,0.45,0.95]); GRN=np.array([0.20,0.78,0.55])
def ovl(lm,rm,a=0.5):
    rgb=np.dstack([g,g,g]).copy(); rgb[rm.reshape(Ny,Nx)]=(1-a)*rgb[rm.reshape(Ny,Nx)]+a*GRN
    rgb[lm.reshape(Ny,Nx)]=(1-a)*rgb[lm.reshape(Ny,Nx)]+a*BLUE; return rgb
def dd(im):
    a=np.log1p(np.clip(im,0,None)); a=a[H//2-130:H//2+130,H//2-130:H//2+130]; o=a[a>0]
    return a,(np.percentile(o,99.5) if o.size else max(a.max(),1e-6))
rows=[("SAM",sam,sam_rest,"SAM_line","SAM_rest"),("DINO transfer\n(class 1)",dino_line,dino_rest,"DINO_line","DINO_rest")]
ct=["Areas on HAADF","Line — avg diffraction","Rest — avg diffraction"]
fig=Figure(figsize=(11,8),facecolor="white")
for ri,(rl,lm,rm,lk,rk) in enumerate(rows):
    ax=fig.add_subplot(2,3,ri*3+1); ax.imshow(ovl(lm,rm),interpolation="nearest",aspect="equal")
    w=BAR/PXNM; ax.add_patch(Rectangle((4,Ny-7),w,2,color="white",ec="black",lw=0.4))
    if ri==0: ax.text(4+w/2,Ny-8,f"{int(BAR)} nm",color="white",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.set_ylabel(rl,fontsize=14,fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
    if ri==0: ax.set_title(ct[0],fontsize=12)
    for ci,k in enumerate([lk,rk]):
        ax=fig.add_subplot(2,3,ri*3+2+ci); img,vm=dd(means[k]); ax.imshow(img,cmap="inferno",vmax=vm,interpolation="nearest",aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        if ri==0: ax.set_title(ct[1+ci],fontsize=12)
fig.suptitle("Na007a (transfer from Na007b model) — SAM vs DINO class 1, on HAADF",fontsize=13,fontweight="bold")
fig.text(0.5,0.01,f"DINO Line = transferred model class 1 ({int(dino_line.sum())} px). IoU(class1, SAM)={iou(dino_line,sam):.3f}. "
         f"Areas on virtual HAADF (sample bright), scale 500 nm (16 nm/px). DINO transfer inference 18.8s vs full NMF+kmeans 677s (~36x faster).",
         ha="center",va="bottom",fontsize=9.2,wrap=True)
fig.tight_layout(rect=[0,0.05,1,0.95]); FigureCanvasAgg(fig)
fig.savefig(f"{F}/na007a_panel.png",dpi=150,facecolor="white",bbox_inches="tight"); print("wrote na007a_panel.png",flush=True)
