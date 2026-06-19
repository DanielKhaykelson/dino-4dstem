"""Figure 4 (KEY): IMC crystallization via DINO, compared to ACOM (alpha/gamma
CIFs). For SI3/SI4/SI5: per-DINO-class average diffraction + crystallinity
(peak/halo) -> amorphous-halo vs crystalline classes; per-pixel crystallinity map
= crystallization-degree map; ACOM phase map (alpha vs gamma, SI3/SI4) with
AMI/ARI vs DINO. Question answered: does DINO cluster by crystal orientation
(like ACOM) or by crystallinity/halo structure? Same sub-cluster approach as NaPHI."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import open_lazy_cube
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline
from sklearn.metrics import adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
OUT = "docs/paper/draft_v2/figs"
IMC = {
 "SI3": dict(run="runs/_gui/IMC_SI3_m097k60", path=r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy", scan=(128, 128)),
 "SI4": dict(run="runs/_gui/IMC_SI4_m097_k60", path=r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy", scan=(128, 128)),
 "SI5": dict(run="runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60", path=r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy", scan=(128, 128)),
}
def load_acom(run, N):
    d = os.path.join(run, "acom", "maps")
    if not os.path.exists(os.path.join(d, "mpfull_phase_id.npy")): return None
    ph = np.load(os.path.join(d, "mpfull_phase_id.npy")).ravel()
    corr = np.load(os.path.join(d, "mpfull_winning_corr.npy")).ravel() if os.path.exists(os.path.join(d, "mpfull_winning_corr.npy")) else None
    return ph[:N], (corr[:N] if corr is not None else None)

summary = {}
fig = Figure(figsize=(16, 9.5), facecolor="white")
ncol = 5
for ri, (name, c) in enumerate(IMC.items()):
    t0 = time.time(); Ny, Nx = c["scan"]; N = Ny * Nx
    asg = np.load(os.path.join(c["run"], "eval", "inference.npz"))["assigns"].astype(int)
    K = int(asg.max()) + 1
    cube = open_lazy_cube(c["path"], scan_shape=(Ny, Nx)); _, _, H, Wd = cube.shape
    cyx = (H - 1) / 2.0; yy, xx = np.indices((H, Wd)); rr = np.sqrt((yy - cyx)**2 + (xx - cyx)**2)
    beam = max(8, round(0.11 * H)); post = rr >= beam; nb = H // 2; lo = max(int(0.10*nb), beam+1); hi = int(0.90*nb)
    dsum = np.zeros((K, H, Wd)); dcnt = np.zeros(K); ph = np.full(N, np.nan); sc = np.zeros(N)
    for rx in range(Ny):
        blk = np.asarray(cube[rx], np.float32)
        for ry in range(Nx):
            i = rx*Nx+ry; pat = blk[ry]; sc[i] = pat[post].sum()
            m, v, _ = _radial_mean_var(pat, (cyx, cyx), beam_px=beam); seg = m[lo:hi]
            if seg.size and seg.sum() > 0:
                halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None)))); ph[i] = np.clip(seg-halo, 0, None).sum()/(seg.sum()+1e-9)
            dsum[asg[i]] += pat; dcnt[asg[i]] += 1
    davg = np.array([dsum[k]/max(dcnt[k], 1) for k in range(K)])
    phc = {k: float(np.nanmedian(ph[asg == k])) for k in range(K) if (asg == k).sum() > 20}
    amork = min(phc, key=phc.get); crysk = max(phc, key=phc.get)
    acom = load_acom(c["run"], N)
    ami_ph = ari_ph = None
    if acom is not None:
        phase, corr = acom
        idx = np.isfinite(phase) & (phase >= 0)
        if corr is not None: idx &= (corr > np.nanpercentile(corr, 50))
        ami_ph = round(float(AMI(asg[idx], phase[idx])), 3); ari_ph = round(float(ARI(asg[idx], phase[idx])), 3)
    summary[name] = dict(K=K, amorphous_class=int(amork), crystalline_class=int(crysk),
                         ph_amorph=round(phc[amork], 3), ph_cryst=round(phc[crysk], 3),
                         ACOM_AMI=ami_ph, ACOM_ARI=ari_ph,
                         cryst_frac=round(float(np.nanmean(ph[asg != 0] > 0.5)), 3) if (asg != 0).any() else None)
    print(f"[{name}] K={K} amorph_cls={amork}(p/h={phc[amork]:.2f}) cryst_cls={crysk}(p/h={phc[crysk]:.2f}) ACOM_AMI={ami_ph} ({time.time()-t0:.0f}s)", flush=True)
    # row panels
    phmap = ph.reshape(Ny, Nx)
    def cr(m): return m[H//2-110:H//2+110, H//2-110:H//2+110]
    a = fig.add_subplot(3, ncol, ri*ncol+1); a.imshow(asg.reshape(Ny, Nx), cmap="tab20", interpolation="nearest"); a.set_ylabel(f"IMC {name}", fontsize=12, fontweight="bold")
    if ri == 0: a.set_title("DINO class map", fontsize=10)
    a.set_xticks([]); a.set_yticks([])
    a = fig.add_subplot(3, ncol, ri*ncol+2); im = a.imshow(phmap, cmap="viridis", vmin=0, vmax=np.nanpercentile(ph, 98));
    if ri == 0: a.set_title("crystallinity (peak/halo)\n= crystallization degree", fontsize=10)
    a.set_xticks([]); a.set_yticks([])
    a = fig.add_subplot(3, ncol, ri*ncol+3)
    if acom is not None:
        a.imshow(phase.reshape(Ny, Nx), cmap="Set1", interpolation="nearest")
        a.set_title(("ACOM phase (alpha/gamma)\n" if ri == 0 else "") + f"AMI vs DINO={ami_ph}", fontsize=10)
    else:
        a.text(0.5, 0.5, "no ACOM\n(SI5)", ha="center", va="center", fontsize=11); a.set_title("ACOM phase" if ri == 0 else "", fontsize=10)
    a.set_xticks([]); a.set_yticks([])
    a = fig.add_subplot(3, ncol, ri*ncol+4); a.imshow(cr(davg[amork]), cmap="inferno", vmin=0, vmax=5)
    a.set_title(f"amorphous class {amork}\n(p/h={phc[amork]:.2f})" if ri == 0 else f"amorph c{amork}", fontsize=9); a.set_xticks([]); a.set_yticks([])
    a = fig.add_subplot(3, ncol, ri*ncol+5); a.imshow(cr(davg[crysk]), cmap="inferno", vmin=0, vmax=5)
    a.set_title(f"crystalline class {crysk}\n(p/h={phc[crysk]:.2f})" if ri == 0 else f"cryst c{crysk}", fontsize=9); a.set_xticks([]); a.set_yticks([])
    np.save(os.path.join(OUT, f"fig4_{name}_phmap.npy"), phmap)
fig.suptitle("Figure 4 — IMC crystallization: DINO clusters by crystallinity/halo (not orientation); ACOM phase agreement is near-zero", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96]); FigureCanvasAgg(fig); fig.savefig(os.path.join(OUT, "fig4_imc_crystallization.png"), dpi=150, facecolor="white")
json.dump(summary, open(os.path.join(OUT, "fig4_imc_summary.json"), "w"), indent=2)
print("\nSUMMARY:", json.dumps(summary, indent=1), flush=True)
print("wrote fig4_imc_crystallization.png", flush=True)
