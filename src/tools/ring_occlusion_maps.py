"""K-class maps under ring occlusion, shown as CHANGE maps. For each IMC film we
zero a ring/band before the identical eval pipeline (crop, polar mask, vmax,
centre mask; clipped to FOV) and re-infer the full 128x128 scan. To make the
effect visible (most of the scan is amorphous and never changes), each column
DIMS the pixels whose class is unchanged from the unmasked map and shows only the
CHANGED pixels in their new colour. Single thin rings + a wide mid-q band + ALL
rings (beam->FOV) + an off-ring control, so single-ring effects are calibrated
against the collapse when everything is removed.

Title: dAll = 1-ARI over whole scan; dGr = 1-ARI over grain pixels; %chg of scan.

  python tools/ring_occlusion_maps.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch.utils.data import Dataset
from data import LoadPRZ
from dino_sr_contrastive_model import load_contrastive_checkpoint
from contrastive_eval import infer_scan
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
import matplotlib as mpl
try:
    from sklearn.metrics import adjusted_rand_score as ARI
except Exception:
    ARI = None

RESIZE = 192; RAW = 512; NY = NX = 128
# (label, r_lo_raw, r_hi_raw); None,None = unmasked
def band(center, hw=5): return (center - hw, center + hw)
def cols(fov, beam):
    return [("unmasked", None, None), ("7.4Å", *band(73)), ("4.75Å", *band(114)),
            ("3.9Å", *band(138)), ("mid-q 4.9–3.4Å", 95, 150),
            ("high-q→edge\n(≤3.6Å)", 145, 256),
            ("all rings", beam, fov), ("off-ring ctrl", *band(63))]
IMC = {
 "SI3": dict(run="runs/_gui/IMC_SI3_m097k60", path=r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy",
             crop=140, pmask=30, cmr=15, fov=187),
 "SI4": dict(run="runs/_gui/IMC_SI4_m097_k60", path=r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy",
             crop=120, pmask=40, cmr=20, fov=160),
 "SI5": dict(run="runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60", path=r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy",
             crop=120, pmask=40, cmr=20, fov=160),
}
FIGS = "docs/paper/draft_v2/figs"; OUT = "docs/explainer/figs"; REVIEW = os.path.join(FIGS, "latest_review")


class BandMaskDS(Dataset):
    def __init__(self, base, rlo192, rhi192):
        self.base = base
        yy, xx = np.indices((RESIZE, RESIZE)); c = (RESIZE - 1) / 2.0
        rr = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
        self.keep = torch.from_numpy(((rr < rlo192) | (rr > rhi192)).astype(np.float32))
    def __len__(self): return len(self.base)
    def __getitem__(self, i): return self.base[i] * self.keep


def run_sample(name, c, device):
    model, _, _, _ = load_contrastive_checkpoint(os.path.join(c["run"], "best.pth"), device=device)
    base = LoadPRZ(c["path"], resize=RESIZE, vmax=5.0)
    beam = max(8, round(0.11 * RAW))
    gid = np.load(os.path.join(FIGS, f"grain_acom_v2_{name}.npz"))["gid"]
    gmask = gid >= 0
    kw = dict(polar_size=192, polar_mask_cols=c["pmask"], center_crop_size=c["crop"],
              center_mask_radius=c["cmr"], dense_remap=False, batch_size=256)
    out = []; a0 = None
    for lab, rlo, rhi in cols(c["fov"], beam):
        if rlo is None:
            ds = base
        else:
            ds = BandMaskDS(base, rlo * RESIZE / RAW, rhi * RESIZE / RAW)
        a = infer_scan(model, ds, device, **kw)["assigns"]
        if a0 is None:
            a0 = a; dAll = dGr = 0.0; chg = 0.0
        else:
            dAll = (1 - ARI(a0, a)) if ARI else np.nan
            dGr = (1 - ARI(a0[gmask], a[gmask])) if ARI else np.nan
            chg = float(np.mean(a != a0))
        out.append((lab, a.reshape(NY, NX), dAll, dGr, chg))
        print(f"[{name}] {lab:>14}: dAll={dAll:.2f} dGr={dGr:.2f} chg={chg*100:.0f}%", flush=True)
    return out


def mean_diffraction(name):
    """Mean diffraction over the sample's grain pixels, from grain_acom_v2 sums."""
    z = np.load(os.path.join(FIGS, f"grain_acom_v2_{name}.npz"))
    gsum, gcnt, vac = z["gsum"], z["gcnt"], z["vac"]; H = int(z["H"])
    keep = ~vac
    return gsum[keep].sum(0) / max(gcnt[keep].sum(), 1.0), H


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {n: run_sample(n, IMC[n], device) for n in IMC}
    ncol = len(res["SI3"])
    NR = len(IMC) + 1                                       # +1 = the "what is blocked" key row
    fig = Figure(figsize=(2.15 * ncol, 2.3 * NR + 0.8), facecolor="white")
    for ri, n in enumerate(IMC):
        maps = res[n]; a0 = maps[0][1]
        ids = sorted(np.unique(a0).tolist()); lut = {p: k for k, p in enumerate(ids)}
        bc = mpl.colormaps.get_cmap("tab20").resampled(max(len(ids), 1))
        cmap = ListedColormap([bc(k) for k in range(len(ids))]); cmap.set_bad("#eeeeee")
        for ci, (lab, mp, dAll, dGr, chg) in enumerate(maps):
            ax = fig.add_subplot(NR, ncol, ri * ncol + ci + 1); ax.set_xticks([]); ax.set_yticks([])
            idx = np.vectorize(lambda v: lut.get(v, -1))(mp).astype(float)
            if ci == 0:
                idx[idx < 0] = np.nan
                ax.imshow(idx, cmap=cmap, interpolation="nearest", vmin=0, vmax=len(ids) - 1)
                ax.set_title("unmasked", fontsize=8)
            else:
                changed = mp != a0                       # CHANGE map: dim unchanged
                disp = np.where(changed, idx, np.nan)
                ax.imshow(np.zeros_like(idx), cmap=ListedColormap(["#f2f2f2"]), interpolation="nearest")
                ax.imshow(disp, cmap=cmap, interpolation="nearest", vmin=0, vmax=len(ids) - 1)
                ax.set_title(f"{lab}\nΔgrain={dGr:.2f}  {chg*100:.0f}% chg", fontsize=7.5)
            if ci == 0:
                ax.set_ylabel(n, fontsize=12, fontweight="bold", rotation=0, labelpad=22, va="center")

    # ---- bottom row: WHICH reciprocal-space region each column zeroes ----
    mp_pat, H = mean_diffraction("SI4"); cyx = (H - 1) / 2.0
    beam = max(8, round(0.11 * RAW)); DISP = 178
    s0 = int(cyx) - DISP; cr = slice(s0, int(cyx) + DISP); pc = cyx - s0
    patch = mp_pat[cr, cr].astype(np.float32)
    yy, xx = np.indices(patch.shape); rr = np.sqrt((yy - pc) ** 2 + (xx - pc) ** 2)
    bm = rr > beam; ref = patch[bm]
    lo, hi = np.percentile(ref, 2), np.percentile(ref, 99.5)
    disp = np.log1p(np.clip(patch, lo, hi) * bm - lo)
    bands = cols(IMC["SI4"]["fov"], beam)
    for ci, (lab, rlo, rhi) in enumerate(bands):
        ax = fig.add_subplot(NR, ncol, len(IMC) * ncol + ci + 1); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(disp, cmap="gray", interpolation="nearest")
        if rlo is not None:
            ov = np.zeros((*patch.shape, 4), np.float32)
            ov[(rr >= rlo) & (rr <= rhi)] = [1.0, 0.10, 0.10, 0.55]      # red = zeroed band
            ax.imshow(ov, interpolation="nearest")
            ax.set_title(f"blocked: {lab}", fontsize=7, color="#B22222")
        else:
            ax.set_title("SI4 mean diffraction", fontsize=7)
        if ci == 0:
            ax.set_ylabel("what is\nblocked\n(reciprocal)", fontsize=9, fontweight="bold",
                          rotation=0, labelpad=24, va="center")
    fig.suptitle("DINO K-class CHANGE maps under ring occlusion (rows 1-3 = real-space scans; bottom row = the reciprocal-space region each column zeroes).\n"
                 "Col 1 = unmasked class map; other cols show ONLY pixels whose class changed when that ring/band is zeroed (colour = new class, grey = unchanged). "
                 "Bottom row: red annulus = the masked band on the SI4 mean diffraction.\nSingle thin rings barely move the (mostly amorphous) map and the off-ring "
                 "control does nothing, but the mid-q band and 'all rings' collapse it — the class is encoded redundantly across the mid-q ring system. Δgrain = 1−ARI over grain pixels.",
                 fontsize=9.2)
    fig.tight_layout(rect=[0.02, 0, 1, 0.92]); FigureCanvasAgg(fig)
    fn = "ring_occlusion_maps_hiq.png"
    p = os.path.join(OUT, fn); fig.savefig(p, dpi=160, facecolor="white")
    import shutil; os.makedirs(REVIEW, exist_ok=True); shutil.copy(p, os.path.join(REVIEW, fn))
    print(f"wrote {fn}", flush=True)


if __name__ == "__main__":
    main()
