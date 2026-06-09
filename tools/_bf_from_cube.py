import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import open_lazy_cube
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
MASKDIR=r"D:/DINOSR/NanoLetters/Figure4/ComapreToKmap"; OUT=r"docs/explainer/figs"
S={"Na007b":(f"{MASKDIR}/Na007b_orient_img.npy", r"D:/DINOSR/data/Na007b_nbed.cube.npy",(126,100)),
   "Na007a":(f"{MASKDIR}/Na007a_orient_img.npy", r"D:/DINOSR/data/Na007a.prz",(126,100))}
for nm,(mp,cp,(Ny,Nx)) in S.items():
    orient=np.load(mp).astype(float).reshape(Ny,Nx)
    cube=open_lazy_cube(cp,scan_shape=(Ny,Nx)); _,_,H,W=cube.shape
    cy,cx=H/2.0,W/2.0; R=min(H,W)/2.0
    yy,xx=np.indices((H,W)); rr=np.sqrt((yy-cy)**2+(xx-cx)**2)
    bf_mask=rr<=0.12*R                              # virtual BF = central disk
    ha_mask=(rr>=0.40*R)&(rr<=0.98*R)               # GUI HAADF = outer annulus
    bf=np.zeros((Ny,Nx),np.float32); ha=np.zeros((Ny,Nx),np.float32); t0=time.time()
    for rx in range(Ny):
        blk=np.asarray(cube[rx],dtype=np.float32)   # (Nx,H,W)
        bf[rx]=blk[:,bf_mask].sum(1); ha[rx]=blk[:,ha_mask].sum(1)
        if rx%20==0: print(f"  {nm} row {rx}/{Ny} ({time.time()-t0:.0f}s)",flush=True)
    np.save(f"{OUT}/{nm}_BF.npy",bf); np.save(f"{OUT}/{nm}_HAADF.npy",ha)
    mask=np.isfinite(orient)&(orient!=0)
    for tag,img in (("BF",bf),("HAADF",ha)):
        fig=Figure(figsize=(5.2,6.4),facecolor="white"); ax=fig.add_subplot(111)
        ax.imshow(img,cmap="gray",aspect="equal",interpolation="nearest")
        ov=np.ma.masked_where(~mask,orient)
        im=ax.imshow(ov,cmap="hsv",alpha=0.65,aspect="equal",interpolation="nearest")
        ax.set_title(f"{nm}: SAM orientation mask on virtual {tag}\n(cube-integrated, {int(mask.sum())} indexed px)",fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im,ax=ax,fraction=0.046,pad=0.03,label="orientation (deg)")
        fig.tight_layout(); FigureCanvasAgg(fig)
        fig.savefig(f"{OUT}/{nm}_sam_on_{tag}.png",dpi=150,facecolor="white",bbox_inches="tight")
    print(f"{nm}: done -> {nm}_sam_on_BF.png / _HAADF.png",flush=True)
print("ALL DONE",flush=True)
