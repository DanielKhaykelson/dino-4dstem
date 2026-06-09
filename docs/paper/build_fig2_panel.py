"""Figure 2 panel: Na007b SAM(top)/DINO(bottom) — col1 = Line(blue)/rest(green)
areas on virtual BF (+500nm scale bar, 16nm/px); col2/3 = region-average diffraction.
Requires precomputed: na007b_BF.npy, na007b_avg_{SAM_line,SAM_rest,DINO_line(1,8),DINO_rest}.npy
(see tools/_bf_from_cube.py and tools/_diff_compare_na007b.py)."""
import os, json, numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle
F=r"docs/explainer/figs"; PXNM=16.0; BAR=500.0
orient=np.load(r"D:/DINOSR/NanoLetters/Figure4/ComapreToKmap/Na007b_orient_img.npy").astype(float)
RUN=r"runs/_gui/Na007b_k60_m097_vmax2"
asg=np.load(os.path.join(RUN,"eval","inference.npz"),allow_pickle=True)["assigns"].astype(int)
Ny,Nx=tuple(json.load(open(os.path.join(RUN,"_train_kwargs.json")))["_sample_config"]["scan_shape"])
bf=np.load(f"{F}/na007b_BF.npy").astype(float)
flake=asg!=0; SL=(np.isfinite(orient)&(orient!=0)).reshape(Ny,Nx); SR=(flake&~(np.isfinite(orient)&(orient!=0))).reshape(Ny,Nx)
DL=np.isin(asg,[1,8]).reshape(Ny,Nx); DR=(flake&~np.isin(asg,[1,8])).reshape(Ny,Nx)
g=np.log1p(np.clip(bf,0,None)); g=(g-g.min())/(g.ptp()+1e-9)
BLUE=np.array([0.13,0.45,0.95]); GRN=np.array([0.20,0.78,0.55])
def ov(lm,rm,a=0.5):
    r=np.dstack([g,g,g]).copy(); r[rm]=(1-a)*r[rm]+a*GRN; r[lm]=(1-a)*r[lm]+a*BLUE; return r
M={k:np.load(f"{F}/na007b_avg_{k}.npy").astype(float) for k in ["SAM_line","SAM_rest","DINO_line(1,8)","DINO_rest"]}
def dd(im):
    a=np.log1p(np.clip(im,0,None)); H=a.shape[0]; a=a[H//2-130:H//2+130,H//2-130:H//2+130]; o=a[a>0]
    return a,(np.percentile(o,99.5) if o.size else a.max())
fig=Figure(figsize=(11,8.2),facecolor="white"); ct=["Segmented areas on BF","Line — average diffraction","Rest — average diffraction"]
for ri,(rl,lm,rm,lk,rk) in enumerate([("SAM",SL,SR,"SAM_line","SAM_rest"),("DINO",DL,DR,"DINO_line(1,8)","DINO_rest")]):
    ax=fig.add_subplot(2,3,ri*3+1); ax.imshow(ov(lm,rm),interpolation="nearest",aspect="equal")
    w=BAR/PXNM; ax.add_patch(Rectangle((5,Ny-8),w,2.4,color="white",ec="black",lw=0.4))
    ax.text(5+w/2,Ny-9,f"{int(BAR)} nm",color="white",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.set_ylabel(rl,fontsize=16,fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
    if ri==0: ax.set_title(ct[0],fontsize=13)
    for ci,k in enumerate([lk,rk]):
        ax=fig.add_subplot(2,3,ri*3+2+ci); img,vm=dd(M[k]); ax.imshow(img,cmap="inferno",vmax=vm,interpolation="nearest",aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        if ri==0: ax.set_title(ct[1+ci],fontsize=13)
fig.suptitle("Na007b — SAM (top) vs DINO (bottom): same domains, same diffraction",fontsize=14,fontweight="bold")
fig.text(0.5,0.005,"Col 1: Line(blue)/rest(green) on virtual BF, scale 500 nm (16 nm/px). IoU 0.74, Dice 0.85; beam-masked r=0.999 (line/rest), 0.5 (line-vs-rest).",ha="center",va="bottom",fontsize=9.2,wrap=True)
fig.tight_layout(rect=[0,0.06,1,0.95]); FigureCanvasAgg(fig)
fig.savefig(f"{F}/na007b_sam_dino_panel.png",dpi=160,facecolor="white",bbox_inches="tight"); print("wrote na007b_sam_dino_panel.png")
