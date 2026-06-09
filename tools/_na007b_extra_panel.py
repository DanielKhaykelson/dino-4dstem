import os, json, time, numpy as np
import sys; sys.path.insert(0, os.getcwd())
from data import open_lazy_cube
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle
F=r"docs/explainer/figs"; PXNM=16.0; BAR=500.0
CUBE=r"D:/DINOSR/data/Na007b_nbed.cube.npy"; RUN=r"runs/_gui/Na007b_k60_m097_vmax2"
orient=np.load(r"D:/DINOSR/NanoLetters/Figure4/ComapreToKmap/Na007b_orient_img.npy").astype(float)
asg=np.load(os.path.join(RUN,"eval","inference.npz"),allow_pickle=True)["assigns"].astype(int)
Ny,Nx=tuple(json.load(open(os.path.join(RUN,"_train_kwargs.json")))["_sample_config"]["scan_shape"])
lab=np.load(f"{F}/na007b_nmf_labels.npy").ravel()
flake=asg!=0; sam=np.isfinite(orient)&(orient!=0); dino=np.isin(asg,[1,8])
# reconstruct nmf_line (clusters mostly in SAM Line, within flake)
nmf=np.zeros_like(flake)
for c in np.unique(lab):
    cf=(lab==c)&flake
    if cf.sum()>0 and (cf&sam).sum()/cf.sum()>0.5: nmf|=(lab==c)
nmf&=flake
ex_d=dino&~sam; ex_n=nmf&~sam
print(f"extra DINO={int(ex_d.sum())}  extra NMF={int(ex_n.sum())}",flush=True)
cube=open_lazy_cube(CUBE,scan_shape=(Ny,Nx)); _,_,H,W=cube.shape
sD=np.zeros((H,W)); sN=np.zeros((H,W)); nD=nN=0
mD=ex_d.reshape(Ny,Nx); mN=ex_n.reshape(Ny,Nx); t0=time.time()
for rx in range(Ny):
    blk=np.asarray(cube[rx],np.float32)
    if mD[rx].any(): sD+=blk[mD[rx]].sum(0); nD+=int(mD[rx].sum())
    if mN[rx].any(): sN+=blk[mN[rx]].sum(0); nN+=int(mN[rx].sum())
    if rx%30==0: print(f"  row {rx} ({time.time()-t0:.0f}s)",flush=True)
avgD=(sD/max(nD,1)).astype(np.float32); avgN=(sN/max(nN,1)).astype(np.float32)
np.save(f"{F}/na007b_avg_DINO_extra.npy",avgD); np.save(f"{F}/na007b_avg_NMF_extra.npy",avgN)
# ---- build 3x4 panel ----
bf=np.load(f"{F}/na007b_BF.npy").astype(float); g=np.log1p(np.clip(bf,0,None)); g=(g-g.min())/(g.ptp()+1e-9)
BLUE=np.array([0.13,0.45,0.95]); RED=np.array([0.92,0.12,0.18])
def seg(line,ex,a=0.55):
    rgb=np.dstack([g,g,g]).copy()
    rgb[(line&~ex).reshape(Ny,Nx)]=(1-a)*rgb[(line&~ex).reshape(Ny,Nx)]+a*BLUE
    rgb[ex.reshape(Ny,Nx)]=(1-a)*rgb[ex.reshape(Ny,Nx)]+a*RED
    return rgb
def dd(im):
    a=np.log1p(np.clip(im,0,None)); a=a[H//2-130:H//2+130,H//2-130:H//2+130]; o=a[a>0]
    return a,(np.percentile(o,99.5) if o.size else max(a.max(),1e-6))
def ld(k): return np.load(f"{F}/na007b_avg_{k}.npy").astype(float)
rows=[("SAM",sam,np.zeros_like(sam),ld("SAM_line"),ld("SAM_rest"),None),
      ("DINO",dino,ex_d,ld("DINO_line(1,8)"),ld("DINO_rest"),avgD),
      ("NMF+kmeans",nmf,ex_n,ld("NMF_line"),ld("NMF_rest"),avgN)]
ct=["Segmentation on BF\n(blue=in SAM, red=extra)","Line — avg diffraction","Rest — avg diffraction","Extra (not in SAM) — avg"]
fig=Figure(figsize=(13.5,10.5),facecolor="white")
for ri,(rl,line,ex,la,ra,ea) in enumerate(rows):
    ax=fig.add_subplot(3,4,ri*4+1); ax.imshow(seg(line,ex),interpolation="nearest",aspect="equal")
    w=BAR/PXNM; ax.add_patch(Rectangle((5,Ny-8),w,2.4,color="white",ec="black",lw=0.4))
    if ri==0: ax.text(5+w/2,Ny-9,f"{int(BAR)} nm",color="white",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.set_ylabel(rl,fontsize=14,fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
    if ri==0: ax.set_title(ct[0],fontsize=12)
    for ci,im in enumerate([la,ra]):
        ax=fig.add_subplot(3,4,ri*4+2+ci); img,vm=dd(im); ax.imshow(img,cmap="inferno",vmax=vm,interpolation="nearest",aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        if ri==0: ax.set_title(ct[1+ci],fontsize=12)
    ax=fig.add_subplot(3,4,ri*4+4)
    if ea is None or ex.sum()==0:
        ax.text(0.5,0.5,"n/a\n(SAM is the\nreference)",ha="center",va="center",fontsize=11,color="#888"); ax.axis("off")
    else:
        img,vm=dd(ea); ax.imshow(img,cmap="inferno",vmax=vm,interpolation="nearest",aspect="equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"n={int(ex.sum())}",fontsize=10)
    if ri==0: ax.set_title(ct[3],fontsize=12)
fig.suptitle("Na007b — SAM / DINO / NMF+kmeans: Line domain, region-average diffraction, and the 'extra' frames each method adds beyond SAM",fontsize=13,fontweight="bold")
fig.text(0.5,0.005,f"Segmentation on virtual BF (sample dark, vacuum bright), scale 500 nm (16 nm/px); blue = Line agreeing with SAM, red = extra pixels not in SAM. "
         f"Extra: DINO {int(ex_d.sum())} px, NMF {int(ex_n.sum())} px. Column 4 = average diffraction of those extra pixels (is NMF adding bad frames?).",
         ha="center",va="bottom",fontsize=9.3,wrap=True)
fig.tight_layout(rect=[0,0.04,1,0.96]); FigureCanvasAgg(fig)
fig.savefig(f"{F}/na007b_3way_panel.png",dpi=150,facecolor="white",bbox_inches="tight"); print("wrote na007b_3way_panel.png",flush=True)
