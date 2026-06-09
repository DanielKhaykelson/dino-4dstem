"""Transfer (Na007b model -> Na007a, Na006a) vs NMF+kmeans, compared to the SAM
reference, with compute timing. DINO Line = transferred model classes 1+8 (no
re-fit); NMF re-fit per sample."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2, torch
from data import LoadPRZ, open_lazy_cube
from dino_sr_contrastive_model import load_contrastive_checkpoint
from contrastive_eval import infer_scan
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle
F=r"docs/explainer/figs"; MD=r"D:/DINOSR/NanoLetters/Figure4/ComapreToKmap"
NB=r"runs/_gui/Na007b_k60_m097_vmax2/best.pth"; VMAX=2.0; MR,MC,CC,COM=11,22,140,True
LINE=[1,8]; DS=40; PXNM=16.0
samples={"Na007a":(r"D:/DINOSR/data/Na007a.prz",(126,100),f"{MD}/Na007a_orient_img.npy"),
         "Na006a":(r"D:/DINOSR/data/Na006a.prz",(100,100),f"{MD}/Na006a_orient_img.npy")}
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def iou(a,b): u=(a|b).sum(); return (a&b).sum()/u if u else 0.0
RES={}
for nm,(cp,(Ny,Nx),op) in samples.items():
    print(f"\n=== {nm} ({Ny}x{Nx}) ===",flush=True)
    orient=np.load(op).astype(float); sam=(np.isfinite(orient)&(orient!=0))
    # ---- DINO transfer (inference only) ----
    ds=LoadPRZ(cp,resize=192,vmax=VMAX)
    model,_,_,_=load_contrastive_checkpoint(NB,device=dev); model.eval()
    t0=time.time()
    inf=infer_scan(model,ds,dev,dense_remap=True,polar_size=192,polar_mask_cols=MC,
                   center_crop_size=CC,com_centering=COM,center_mask_radius=MR,eval_temp=0.06,batch_size=128)
    t_dino=time.time()-t0
    asg=np.asarray(inf["assigns"]).astype(int); Kd=len(np.unique(asg))
    dino_line=np.isin(asg,LINE)
    # best-overlap DINO classes (sanity vs fixed 1,8)
    best=[c for c in np.unique(asg) if ((asg==c)&sam).sum()/max((asg==c).sum(),1)>0.5]
    dino_line_ov=np.isin(asg,best)
    print(f"DINO transfer: {t_dino:.1f}s  Kactive={Kd}  frames/s={Ny*Nx/t_dino:.0f}",flush=True)
    print(f"  IoU(class1+8, SAM)={iou(dino_line,sam):.3f}   IoU(best-overlap {best}, SAM)={iou(dino_line_ov,sam):.3f}",flush=True)
    # ---- NMF+kmeans (re-fit) + BF ----
    cube=open_lazy_cube(cp,scan_shape=(Ny,Nx)); _,_,H,W=cube.shape
    cy=(H-1)/2; yy,xx=np.indices((H,W)); rr=np.sqrt((yy-cy)**2+(xx-cy)**2); bfm=rr<=0.12*min(H,W)/2
    t0=time.time(); P=np.zeros((Ny*Nx,DS*DS),np.float32); bf=np.zeros(Ny*Nx)
    for rx in range(Ny):
        blk=np.asarray(cube[rx],np.float32)
        for ry in range(Nx):
            P[rx*Nx+ry]=cv2.resize(np.log1p(np.clip(blk[ry],0,None)),(DS,DS),interpolation=cv2.INTER_AREA).ravel()
            bf[rx*Nx+ry]=blk[ry][bfm].sum()
    Wd=NMF(n_components=min(2*Kd,30),init="nndsvda",max_iter=400,random_state=0).fit_transform(np.clip(P,0,None))
    lab=KMeans(Kd,n_init=10,random_state=0).fit_predict(StandardScaler().fit_transform(Wd))
    t_nmf=time.time()-t0
    nmf_line=np.zeros(Ny*Nx,bool)
    for c in np.unique(lab):
        cm=lab==c
        if cm.sum()>0 and (cm&sam).sum()/cm.sum()>0.5: nmf_line|=cm
    print(f"NMF+kmeans: {t_nmf:.1f}s   IoU(NMF, SAM)={iou(nmf_line,sam):.3f}",flush=True)
    RES[nm]=dict(Ny=Ny,Nx=Nx,bf=bf.reshape(Ny,Nx),sam=sam.reshape(Ny,Nx),
                 dino=dino_line.reshape(Ny,Nx),nmf=nmf_line.reshape(Ny,Nx),
                 t_dino=t_dino,t_nmf=t_nmf,iou_d=iou(dino_line,sam),iou_n=iou(nmf_line,sam),Kd=Kd)
    np.save(f"{F}/{nm}_BF.npy",bf.reshape(Ny,Nx)); np.save(f"{F}/{nm}_transfer_assigns.npy",asg.reshape(Ny,Nx))
    np.save(f"{F}/{nm}_nmf_labels.npy",lab.reshape(Ny,Nx))
# ---- panel: 2 rows (samples) x 3 cols (SAM / DINO-transfer / NMF) on BF ----
BLUE=np.array([0.13,0.45,0.95])
fig=Figure(figsize=(11,7.6),facecolor="white")
cols=[("SAM (reference)","sam",None),("DINO transfer (class 1+8)","dino","t_dino"),("NMF+kmeans","nmf","t_nmf")]
ious={"dino":"iou_d","nmf":"iou_n"}
for ri,nm in enumerate(["Na007a","Na006a"]):
    R=RES[nm]; g=np.log1p(np.clip(R["bf"],0,None)); g=(g-g.min())/(g.ptp()+1e-9)
    for ci,(ct,key,tk) in enumerate(cols):
        ax=fig.add_subplot(2,3,ri*3+ci+1)
        rgb=np.dstack([g,g,g]).copy(); m=R[key]; a=0.55; rgb[m]=(1-a)*rgb[m]+a*BLUE
        ax.imshow(rgb,interpolation="nearest",aspect="equal")
        w=500/PXNM; ax.add_patch(Rectangle((4,R["Ny"]-7),w,2,color="white",ec="black",lw=0.4))
        ax.set_xticks([]); ax.set_yticks([])
        if ci==0: ax.set_ylabel(nm,fontsize=14,fontweight="bold")
        ttl=ct
        if key in ious: ttl+=f"\nIoU vs SAM={R[ious[key]]:.2f}"
        if tk: ttl+=f"  ·  {R[tk]:.0f}s"
        ax.set_title(ttl,fontsize=10)
fig.suptitle("Transfer (Na007b model) vs NMF+kmeans on new samples — Line domain on BF, vs SAM reference + compute time",fontsize=12,fontweight="bold")
cap=(f"DINO = inference only with the Na007b-trained model (no re-training); Line = its classes 1+8. NMF+kmeans re-fit per sample. Scale 500 nm (16 nm/px).  "
     f"Times: Na007a DINO {RES['Na007a']['t_dino']:.0f}s / NMF {RES['Na007a']['t_nmf']:.0f}s; Na006a DINO {RES['Na006a']['t_dino']:.0f}s / NMF {RES['Na006a']['t_nmf']:.0f}s.")
fig.text(0.5,0.01,cap,ha="center",va="bottom",fontsize=9.2,wrap=True)
fig.tight_layout(rect=[0,0.05,1,0.95]); FigureCanvasAgg(fig)
fig.savefig(f"{F}/transfer_vs_nmf_panel.png",dpi=150,facecolor="white",bbox_inches="tight")
print("\nwrote transfer_vs_nmf_panel.png",flush=True)
