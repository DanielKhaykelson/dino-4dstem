"""Causal ring-occlusion probe: which diffraction ring does the trained DINO model
actually USE to classify grains? For each ring we zero a thin annulus (+-HW px in
the model's 192 input space) BEFORE the identical eval transform (CenterCrop=crop,
Resize, PolarTransform, PolarMaskLeft, same vmax, same center_mask_radius), re-run
inference over the grain pixels, and measure how much the assignment breaks down
vs the UNMASKED re-inference:
  disruption = 1 - ARI(masked, unmasked)      (1 = totally rewires the classes)
  conf_drop  = mean top-softmax(unmasked) - mean top-softmax(masked)
  grain_flip = fraction of grains whose majority class changes
Rings are clipped to the model field of view (crop). An off-ring band is the
control. If two rings both matter, both show high disruption.

  python tools/ring_occlusion_probe.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch.utils.data import Dataset, Subset
from data import LoadPRZ
from dino_sr_contrastive_model import load_contrastive_checkpoint
from contrastive_eval import infer_scan
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
try:
    from sklearn.metrics import adjusted_rand_score as ARI
except Exception:
    def ARI(a, b):                       # fallback exact ARI
        a = np.asarray(a); b = np.asarray(b); n = len(a)
        ca = {v: i for i, v in enumerate(np.unique(a))}; cb = {v: i for i, v in enumerate(np.unique(b))}
        C = np.zeros((len(ca), len(cb)))
        for x, y in zip(a, b): C[ca[x], cb[y]] += 1
        si = C.sum(1); sj = C.sum(0)
        cij = (C * (C - 1) / 2).sum(); ai = (si * (si - 1) / 2).sum(); bj = (sj * (sj - 1) / 2).sum()
        e = ai * bj / (n * (n - 1) / 2); m = (ai + bj) / 2
        return float((cij - e) / (m - e + 1e-12))

INV = 0.00185; RESIZE = 192; RAW = 512; HW = 2          # mask half-width in 192-space
RINGS = [("7.4Å", 73), ("6.0Å", 90), ("4.75Å", 114), ("3.9Å", 138), ("off-ring(ctrl)", 63)]
IMC = {
 "SI3": dict(run="runs/_gui/IMC_SI3_m097k60", path=r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy",
             crop=140, pmask=30, cmr=15, fov=187),
 "SI4": dict(run="runs/_gui/IMC_SI4_m097_k60", path=r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy",
             crop=120, pmask=40, cmr=20, fov=160),
 "SI5": dict(run="runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60", path=r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy",
             crop=120, pmask=40, cmr=20, fov=160),
}
FIGS = "docs/paper/draft_v2/figs"; OUT = "docs/explainer/figs"; REVIEW = os.path.join(FIGS, "latest_review")


class RingMaskDS(Dataset):
    """Wrap LoadPRZ; zero an annulus at 192-radius r192 (+-HW) on each frame."""
    def __init__(self, base, r192):
        self.base = base
        yy, xx = np.indices((RESIZE, RESIZE)); c = (RESIZE - 1) / 2.0
        rr = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
        self.keep = torch.from_numpy(((rr < r192 - HW) | (rr > r192 + HW)).astype(np.float32))
    def __len__(self): return len(self.base)
    def __getitem__(self, i): return self.base[i] * self.keep


def run_sample(name, c, device):
    model, _, _, _ = load_contrastive_checkpoint(os.path.join(c["run"], "best.pth"), device=device)
    base = LoadPRZ(c["path"], resize=RESIZE, vmax=5.0)
    gid = np.load(os.path.join(FIGS, f"grain_acom_v2_{name}.npz"))["gid"]
    idx = np.where(gid >= 0)[0]
    kw = dict(polar_size=192, polar_mask_cols=c["pmask"], center_crop_size=c["crop"],
              center_mask_radius=c["cmr"], dense_remap=False, batch_size=256)

    def infer(ds):
        r = infer_scan(model, Subset(ds, idx.tolist()), device, **kw)
        return r["assigns"], r["soft_probs"].max(1)

    a0, p0 = infer(base)                                   # unmasked baseline
    g = gid[idx]
    def grain_major(a):
        out = {}
        for gg in np.unique(g):
            lab = a[g == gg]; out[gg] = np.bincount(lab).argmax()
        return out
    m0 = grain_major(a0)
    rows = []
    for rlab, Rraw in RINGS:
        if Rraw > c["fov"] and "ctrl" not in rlab:
            rows.append((rlab, Rraw, np.nan, np.nan, np.nan, True)); continue
        r192 = Rraw * RESIZE / RAW
        a, p = infer(RingMaskDS(base, r192))
        disruption = 1.0 - ARI(a0, a)
        conf_drop = float(p0.mean() - p.mean())
        mm = grain_major(a)
        flip = float(np.mean([m0[gg] != mm[gg] for gg in m0]))
        rows.append((rlab, Rraw, disruption, conf_drop, flip, False))
        print(f"[{name}] mask {rlab:>14} (r={Rraw}px d): disruption={disruption:.2f} conf_drop={conf_drop:+.3f} grain_flip={flip:.2f}", flush=True)
    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {n: run_sample(n, IMC[n], device) for n in IMC}
    fig = Figure(figsize=(15, 4.6), facecolor="white")
    for ri, n in enumerate(IMC):
        rows = res[n]
        labs = [r[0] for r in rows]; dis = [r[2] for r in rows]; fl = [r[4] for r in rows]
        x = np.arange(len(labs)); w = 0.4
        ax = fig.add_subplot(1, 3, ri + 1)
        ax.bar(x - w / 2, dis, w, label="disruption (1−ARI)", color="#C0392B")
        ax.bar(x + w / 2, fl, w, label="grain class-flip frac", color="#2E86C1")
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7, rotation=30, ha="right")
        ax.set_ylim(0, 1.05); ax.set_title(f"{n}: ring occlusion (FOV={IMC[n]['fov']}px)", fontsize=10)
        if ri == 0: ax.set_ylabel("effect of removing the ring"); ax.legend(fontsize=7)
        for xi, d in enumerate(dis):
            if np.isfinite(d): ax.text(xi - w/2, d + 0.02, f"{d:.2f}", ha="center", fontsize=6)
            else: ax.text(xi, 0.5, "outside\nFOV", ha="center", fontsize=6, color="#888")
    fig.suptitle("Causal ring-occlusion probe: zero each ring (±2px, model space) → re-infer grain pixels → how much the DINO classes change.\n"
                 "High bar = the model relies on that ring to classify grains. Off-ring band = control. (Same crop / polar-mask / vmax as training; clipped to FOV.)",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9]); FigureCanvasAgg(fig)
    p = os.path.join(OUT, "ring_occlusion_probe.png"); fig.savefig(p, dpi=160, facecolor="white")
    import shutil; os.makedirs(REVIEW, exist_ok=True); shutil.copy(p, os.path.join(REVIEW, "ring_occlusion_probe.png"))
    json.dump({n: [[r[0], r[1], None if np.isnan(r[2]) else round(r[2],3),
                    None if np.isnan(r[3]) else round(r[3],3),
                    None if np.isnan(r[4]) else round(r[4],3)] for r in res[n]] for n in IMC},
              open(os.path.join(FIGS, "ring_occlusion_probe.json"), "w"), indent=2)
    print("wrote ring_occlusion_probe.png + .json", flush=True)


if __name__ == "__main__":
    main()
