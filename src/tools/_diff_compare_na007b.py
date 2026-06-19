import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import open_lazy_cube
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
ORIENT=r"D:/DINOSR/NanoLetters/Figure4/ComapreToKmap/Na007b_orient_img.npy"
CUBE=r"D:/DINOSR/data/Na007b_nbed.cube.npy"
RUN=r"runs/_gui/Na007b_k60_m097_vmax2"; OUT=r"docs/explainer/figs"
Ny,Nx=tuple(json.load(open(os.path.join(RUN,"_train_kwargs.json")))["_sample_config"]["scan_shape"])
orient=np.load(ORIENT).astype(float)
asg=np.load(os.path.join(RUN,"eval","inference.npz"),allow_pickle=True)["assigns"].astype(int)
flake=asg!=0
sam_line=np.isfinite(orient)&(orient!=0)
sam_rest=flake&~sam_line
dino_line=np.isin(asg,[1,8])
dino_rest=flake&~dino_line
masks={"SAM_line":sam_line,"SAM_rest":sam_rest,"DINO_line(1,8)":dino_line,"DINO_rest":dino_rest}
for k,m in masks.items(): print(f"{k}: {int(m.sum())} px",flush=True)
cube=open_lazy_cube(CUBE,scan_shape=(Ny,Nx)); _,_,H,W=cube.shape
sums={k:np.zeros((H,W),np.float64) for k in masks}; cnt={k:0 for k in masks}
mr={k:m.reshape(Ny,Nx) for k,m in masks.items()}
t0=time.time()
for rx in range(Ny):
    blk=np.asarray(cube[rx],np.float32)
    for k in masks:
        idx=mr[k][rx]
        if idx.any(): sums[k]+=blk[idx].sum(0); cnt[k]+=int(idx.sum())
    if rx%20==0: print(f"  row {rx}/{Ny} ({time.time()-t0:.0f}s)",flush=True)
means={k:(sums[k]/max(cnt[k],1)).astype(np.float32) for k in masks}
for k in masks: np.save(f"{OUT}/na007b_avg_{k}.npy",means[k])

# --- comparison metrics on log-scaled, min-max-normalized patterns ---
try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAVE=True
except Exception:
    HAVE=False
def disp(a): 
    a=np.log1p(np.clip(a,0,None)); a-=a.min(); return a/(a.max()+1e-9)
D={k:disp(means[k]) for k in masks}
keys=list(masks)
def pear(a,b): return float(np.corrcoef(a.ravel(),b.ravel())[0,1])
def ss(a,b): return float(ssim_fn(a,b,data_range=1.0)) if HAVE else float('nan')
print("\n=== SSIM / Pearson (log, normalized) ===",flush=True)
for i in range(4):
    for j in range(i+1,4):
        print(f"  {keys[i]:16s} vs {keys[j]:16s}  SSIM={ss(D[keys[i]],D[keys[j]]):.3f}  r={pear(D[keys[i]],D[keys[j]]):.3f}",flush=True)
print("\nKEY: line-vs-line and rest-vs-rest should be HIGH; line-vs-rest LOW",flush=True)

fig=Figure(figsize=(10,9),facecolor="white")
order=["SAM_line","DINO_line(1,8)","SAM_rest","DINO_rest"]
for i,k in enumerate(order):
    ax=fig.add_subplot(2,2,i+1)
    vm=np.percentile(D[k][D[k]>0],99.5)
    ax.imshow(D[k],cmap="inferno",vmax=vm,interpolation="nearest",aspect="equal")
    ax.set_title(f"{k}  (n={cnt[k]})",fontsize=12); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Na007b average diffraction by region\nSAM_line vs DINO_line: SSIM={ss(D['SAM_line'],D['DINO_line(1,8)']):.2f} r={pear(D['SAM_line'],D['DINO_line(1,8)']):.2f}   |   "
             f"SAM_rest vs DINO_rest: SSIM={ss(D['SAM_rest'],D['DINO_rest']):.2f} r={pear(D['SAM_rest'],D['DINO_rest']):.2f}",fontsize=12,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95]); FigureCanvasAgg(fig)
fig.savefig(f"{OUT}/na007b_avgdiff_compare.png",dpi=150,facecolor="white",bbox_inches="tight")
print("wrote na007b_avgdiff_compare.png  skimage=",HAVE,flush=True)
