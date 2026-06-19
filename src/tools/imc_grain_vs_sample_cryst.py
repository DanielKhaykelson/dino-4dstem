"""Grain-average vs whole-sample crystallinity / peak-finding for IMC.

The point: DINO is rotation-invariant, so each grain (class) groups probes of the
SAME crystallinity but DIFFERENT orientation. Averaging within a grain turns sparse
single-frame Bragg (buried in shot noise -> peak-finding fails per pixel) into sharp
RINGS with real SNR. So run peak-finding + crystallinity (p/h) on:
  (1) every grain's average pattern   (DINO-concentrated)
  (2) the whole-sample average pattern (control: dilutes sparse crystalline grains
      into the dominant glass)
and ACOM (aggregate, from existing per-pixel maps) per grain. Then compare.

Reuses imc_glassorder_{name}.npz (has dmean = per-class mean pattern, assigns, H)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.signal import find_peaks
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cmx

OUT = "docs/paper/draft_v2/figs"
INV_ANG = 0.00185                       # A^-1 per raw px (IMC q-calib)
ALPHA_D = [7.4, 6.0, 4.75, 3.9]         # strong alpha rings in range
MIN_PIX = 50                            # ignore tiny grains
ACOM = {                                 # per-pixel ACOM maps (SI5 has none)
 "SI3": "runs/_gui/IMC_SI3_m097k60/acom/maps",
 "SI4": "runs/_gui/IMC_SI4_m097_k60/acom/maps",
}

def radial_cryst(pat, cyx, beam, lo, hi):
    """return (r_axis, baseline-subtracted radial peak, seg, halo, p/h)."""
    m, _, _ = _radial_mean_var(pat, (cyx, cyx), beam_px=beam)
    seg = m[lo:hi]
    if not (seg.size and seg.sum() > 0):
        return None
    halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None))))
    peak = np.clip(seg - halo, 0.0, None)
    ph = float(peak.sum() / (seg.sum() + 1e-9))
    return np.arange(lo, hi), peak, seg, halo, ph

def find_alpha_peaks(r_axis, peak, halo, alpha_r, tol=0.06):
    """find_peaks on the baseline-subtracted radial; return (#alpha-matched, list).
    prominence scaled to the halo so weak rings on a big halo still count."""
    if peak.max() <= 0:
        return 0, []
    prom = max(0.05 * peak.max(), 0.02 * float(np.median(halo)))
    idx, props = find_peaks(peak, prominence=prom, distance=4)
    found = []
    for j, p in enumerate(idx):
        r = r_axis[p]; d = 1.0 / (r * INV_ANG + 1e-9)
        rel = peak[p] / (halo[p] + 1e-9)          # ring strength over local halo
        hit = any(abs(r - ar) / ar <= tol for ar in alpha_r)
        found.append(dict(r=int(r), d=round(float(d), 2), rel=round(float(rel), 3), alpha=bool(hit)))
    nA = sum(f["alpha"] for f in found)
    return nA, found

def acom_per_grain(mapdir, asg, K):
    pid = np.load(os.path.join(mapdir, "mpfull_phase_id.npy")).ravel()
    corr = np.load(os.path.join(mapdir, "mpfull_winning_corr.npy")).ravel()
    rmat = np.load(os.path.join(mapdir, "mpfull_winning_rmat.npy")).reshape(-1, 3, 3)
    N = min(len(corr), len(asg)); pid, corr, asg2 = pid[:N], corr[:N], asg[:N]; rmat = rmat[:N]
    thr = np.nanpercentile(corr, 60)                 # "indexed" = corr in top 40%
    za = rmat[:, :, 2]                               # zone-axis (beam dir in crystal frame)
    za = za / (np.linalg.norm(za, axis=1, keepdims=True) + 1e-9)
    out = {}
    for k in range(K):
        sel = (asg2 == k)
        if sel.sum() < MIN_PIX: continue
        c = corr[sel]; frac = float((c > thr).mean())
        # zone-axis order parameter (mean resultant length) on indexed pixels
        ind = sel & (corr > thr)
        order = float(np.linalg.norm(za[ind].mean(0))) if ind.sum() >= 10 else None
        out[k] = dict(mean_corr=round(float(np.nanmean(c)), 3), frac_indexed=round(frac, 3),
                      zone_order=None if order is None else round(order, 3), n=int(sel.sum()))
    # whole sample
    out["ALL"] = dict(mean_corr=round(float(np.nanmean(corr)), 3),
                      frac_indexed=round(float((corr > thr).mean()), 3),
                      zone_order=round(float(np.linalg.norm(za[corr > thr].mean(0))), 3),
                      n=int(N))
    return out

summary = {}
fig = Figure(figsize=(15, 11), facecolor="white")
NAMES = ["SI3", "SI4", "SI5"]
for ri, name in enumerate(NAMES):
    z = np.load(os.path.join(OUT, f"imc_glassorder_{name}.npz"))
    dmean = z["dmean"]; asg = z["assigns"].astype(int); H = int(z["H"]); Ny, Nx = z["scan"]
    K = dmean.shape[0]; cyx = (H - 1) / 2.0
    nb = H // 2; beam = max(8, round(0.11 * H)); lo = beam + 1; hi = int(0.85 * nb)
    alpha_r = [1.0 / (d * INV_ANG) for d in ALPHA_D]
    counts = np.array([(asg == k).sum() for k in range(K)])
    # whole-sample average = count-weighted mean of grain averages == mean of all frames
    wsum = (dmean * counts[:, None, None]).sum(0) / max(counts.sum(), 1)

    grains = {}
    for k in range(K):
        if counts[k] < MIN_PIX: continue
        rc = radial_cryst(dmean[k], cyx, beam, lo, hi)
        if rc is None: continue
        r_axis, peak, seg, halo, ph = rc
        nA, found = find_alpha_peaks(r_axis, peak, halo, alpha_r)
        grains[k] = dict(n=int(counts[k]), ph=round(ph, 3), n_alpha=nA,
                         peaks=[f for f in found if f["alpha"]],
                         _r=r_axis, _peak=peak, _halo=halo)
    # whole sample
    r_axis, wpeak, wseg, whalo, wph = radial_cryst(wsum, cyx, beam, lo, hi)
    wnA, wfound = find_alpha_peaks(r_axis, wpeak, whalo, alpha_r)

    acom = acom_per_grain(ACOM[name], asg, K) if name in ACOM else None
    # ---- record ----
    g_ph = {k: g["ph"] for k, g in grains.items()}
    cryst_grains = sorted([k for k in grains if grains[k]["n_alpha"] >= 1],
                          key=lambda k: -grains[k]["ph"])
    summary[name] = dict(
        K=K, n_grains=len(grains),
        whole_sample=dict(ph=round(wph, 3), n_alpha=wnA,
                          alpha_d=[f["d"] for f in wfound if f["alpha"]]),
        grains={int(k): dict(n=grains[k]["n"], ph=grains[k]["ph"], n_alpha=grains[k]["n_alpha"],
                             alpha_d=[p["d"] for p in grains[k]["peaks"]],
                             acom=(acom.get(k) if acom else None)) for k in grains},
        crystalline_grains=[int(k) for k in cryst_grains],
        max_grain_ph=round(max(g_ph.values()), 3),
        whole_vs_best="whole p/h=%.3f (%d alpha-peaks) | best grain p/h=%.3f (%d alpha-peaks)" % (
            wph, wnA, max(g_ph.values()), max(grains[k]["n_alpha"] for k in grains)),
        acom_whole=(acom["ALL"] if acom else None),
    )

    # ---- plots: (1) overlaid radial profiles  (2) p/h bar  (3) ACOM/peaks ----
    ax = fig.add_subplot(3, 3, ri*3 + 1)
    order = sorted(grains, key=lambda k: grains[k]["ph"])
    cmap = cmx.get_cmap("viridis")
    phmax = max(g_ph.values()) + 1e-9
    for k in order:
        g = grains[k]; pk = g["_peak"]; nrm = pk / (pk.max() + 1e-9)
        ax.plot(g["_r"], nrm + 0.0, color=cmap(g["ph"] / phmax), lw=1.1, alpha=0.85)
    wn = wpeak / (wpeak.max() + 1e-9)
    ax.plot(r_axis, wn, color="k", lw=2.6, ls="--", label=f"whole sample (p/h={wph:.2f}, {wnA}α)")
    for ar, d in zip(alpha_r, ALPHA_D):
        if lo < ar < hi: ax.axvline(ar, color="r", ls=":", lw=0.8, alpha=0.6)
    ax.set_title(f"IMC {name}: grain-avg radial peaks (color=p/h)\nred dotted = α rings {ALPHA_D}Å", fontsize=9)
    ax.set_xlabel("radius (px)"); ax.set_ylabel("baseline-sub. (norm)"); ax.legend(fontsize=7)

    ax = fig.add_subplot(3, 3, ri*3 + 2)
    ks = sorted(grains, key=lambda k: -grains[k]["ph"])
    vals = [grains[k]["ph"] for k in ks]
    cols = ["#C0392B" if grains[k]["n_alpha"] >= 1 else "#888" for k in ks]
    ax.bar([f"c{k}" for k in ks], vals, color=cols)
    ax.axhline(wph, color="k", ls="--", lw=2, label=f"whole-sample p/h={wph:.3f}")
    ax.set_ylabel("grain-avg p/h"); ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.set_title(f"{name}: per-grain crystallinity (red=α-peaks found)", fontsize=9); ax.legend(fontsize=7)

    ax = fig.add_subplot(3, 3, ri*3 + 3)
    if acom:
        ks2 = [k for k in ks if acom.get(k)]
        ph_v = [grains[k]["ph"] for k in ks2]
        fi_v = [acom[k]["frac_indexed"] for k in ks2]
        sc = ax.scatter(ph_v, fi_v, c=[grains[k]["n_alpha"] for k in ks2], cmap="Reds", s=60, edgecolor="k", vmin=0, vmax=4)
        for k, x, y in zip(ks2, ph_v, fi_v): ax.annotate(f"c{k}", (x, y), fontsize=7)
        ax.axhline(acom["ALL"]["frac_indexed"], color="k", ls="--", lw=1.5, label=f"whole ACOM frac={acom['ALL']['frac_indexed']}")
        ax.set_xlabel("grain-avg p/h"); ax.set_ylabel("ACOM frac indexed (corr>p60)")
        ax.set_title(f"{name}: ACOM-per-grain vs crystallinity\n(color=#α-peaks)", fontsize=9); ax.legend(fontsize=7)
        fig.colorbar(sc, ax=ax, fraction=0.046)
    else:
        ax.scatter([grains[k]["ph"] for k in ks], [grains[k]["n_alpha"] for k in ks],
                   c="#C0392B", s=60, edgecolor="k")
        for k in ks: ax.annotate(f"c{k}", (grains[k]["ph"], grains[k]["n_alpha"]), fontsize=7)
        ax.set_xlabel("grain-avg p/h"); ax.set_ylabel("# α-peaks on grain avg")
        ax.set_title(f"{name}: no ACOM (SI5) — α-peaks vs p/h\nwhole-sample p/h={wph:.3f} ({wnA} α-peaks)", fontsize=9)

    print(f"[{name}] K={K} grains={len(grains)} | WHOLE p/h={wph:.3f} ({wnA} α) "
          f"| crystalline grains(α-peak)={cryst_grains} | best grain p/h={max(g_ph.values()):.3f}", flush=True)
    if acom: print(f"    ACOM whole: {acom['ALL']}", flush=True)

fig.suptitle("IMC: peak-finding + crystallinity on DINO GRAIN AVERAGES vs WHOLE-SAMPLE average\n"
             "rotation-invariant grains concentrate sparse α (rings emerge); whole-sample average dilutes it", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(fig)
fig.savefig(os.path.join(OUT, "imc_grain_vs_sample_cryst.png"), dpi=150, facecolor="white")
json.dump(summary, open(os.path.join(OUT, "imc_grain_vs_sample_cryst.json"), "w"), indent=2)
print("\n==== SUMMARY ====")
for n in NAMES:
    s = summary[n]; print(f"{n}: {s['whole_vs_best']} | crystalline grains={s['crystalline_grains']}")
print("wrote imc_grain_vs_sample_cryst.png + .json", flush=True)
