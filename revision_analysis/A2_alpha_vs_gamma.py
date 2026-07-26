"""
A2  -  alpha vs gamma indomethacin discrimination on the observed radial profiles.

Referee (Fig S7): the precursor/needle indexing overlays alpha sticks only; single-phase-alpha
needs "and NOT gamma". Our own alpha+gamma synthetic phantom shows the CLASSIFIER cannot separate
the polymorphs, so phase ID must come from indexing the class-average patterns, not from DINO.

This script computes kinematical powder patterns for BOTH alpha and gamma indomethacin directly
from the CIFs (no pymatgen: CIF parsed here; exact d-spacings from the lattice, Z-weighted
structure-factor intensities from the symmetry-expanded atoms), and overlays them on the observed
azimuthally-averaged radial profiles of the least-ordered precursor class and a mature needle class
(log-y, d-spacing axis, calibration 0.00185 A^-1/px). It then tabulates observed peaks vs the
nearest alpha and gamma reflection to test whether gamma can be excluded.

Output: figs/Review/A2_alpha_vs_gamma.png + printed d-spacing table.
"""
import os, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CIF = {"alpha": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/cifs/alpha.cif",
       "gamma": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/cifs/gamma.cif"}
INV = 0.00185
DMIN, DMAX = 3.3, 9.5
Z = {"H": 1, "C": 6, "N": 7, "O": 8, "CL": 17, "Cl": 17, "S": 16}


def parse_cif(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    def val(tag):
        m = re.search(tag + r"\s+([0-9.\-()]+)", txt)
        return float(re.sub(r"\(.*?\)", "", m.group(1))) if m else None
    a = val(r"_cell_length_a"); b = val(r"_cell_length_b"); c = val(r"_cell_length_c")
    al = val(r"_cell_angle_alpha"); be = val(r"_cell_angle_beta"); ga = val(r"_cell_angle_gamma")
    # symmetry operations
    lines = txt.splitlines()
    sym = []
    for i, ln in enumerate(lines):
        if "symop_operation_xyz" in ln or "symmetry_equiv_pos_as_xyz" in ln:
            j = i + 1
            while j < len(lines):
                s = lines[j].strip()
                if not s or s.startswith("_") or s.startswith("loop_"):
                    break
                m = re.search(r"['\"]?\s*([xyz0-9+\-/, .]+?)\s*['\"]?\s*$", s)
                if m and ("," in m.group(1)):
                    ops = m.group(1).split(",")
                    if len(ops) == 3:
                        sym.append([o.strip() for o in ops])
                j += 1
            break
    if not sym:
        sym = [["x", "y", "z"]]
    # atom_site loop: find column order, then read fract x/y/z + element
    atoms = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            cols = []; j = i + 1
            while j < len(lines) and lines[j].strip().startswith("_"):
                cols.append(lines[j].strip()); j += 1
            if any("_atom_site_fract_x" in c for c in cols):
                cx = cols.index("_atom_site_fract_x"); cy = cols.index("_atom_site_fract_y"); cz = cols.index("_atom_site_fract_z")
                csym = next((k for k, c in enumerate(cols) if "type_symbol" in c), None)
                clab = next((k for k, c in enumerate(cols) if "_atom_site_label" in c), 0)
                while j < len(lines):
                    s = lines[j].strip()
                    if not s or s.startswith("_") or s.startswith("loop_"):
                        break
                    parts = s.split()
                    if len(parts) >= len(cols):
                        try:
                            x = float(re.sub(r"\(.*?\)", "", parts[cx]))
                            y = float(re.sub(r"\(.*?\)", "", parts[cy]))
                            z = float(re.sub(r"\(.*?\)", "", parts[cz]))
                        except ValueError:
                            j += 1; continue
                        raw = parts[csym] if csym is not None else parts[clab]
                        el = re.sub(r"[^A-Za-z]", "", raw)[:2]
                        el = el[0].upper() + el[1:].lower() if len(el) == 2 else el.upper()
                        zz = Z.get(el, Z.get(el[:1].upper(), 6))
                        atoms.append((x, y, z, zz))
                    j += 1
                break
        i += 1
    return dict(a=a, b=b, c=c, al=al, be=be, ga=ga, sym=sym, atoms=atoms)


def pattern(cif):
    import numpy as np
    a, b, c = cif["a"], cif["b"], cif["c"]
    al, be, ga = np.radians([cif["al"], cif["be"], cif["ga"]])
    # reciprocal metric via direct metric tensor
    ca, cb, cg = np.cos([al, be, ga]); sg = np.sin(ga)
    G = np.array([[a*a, a*b*cg, a*c*cb],
                  [a*b*cg, b*b, b*c*ca],
                  [a*c*cb, b*c*ca, c*c]])
    Gs = np.linalg.inv(G)   # reciprocal metric tensor; d(hkl) = 1/sqrt(h.Gs.h)
    # expand atoms by symmetry
    exp = []
    for (x, y, z, zz) in cif["atoms"]:
        for ops in cif["sym"]:
            env = {"x": x, "y": y, "z": z}
            p = [eval(o.replace("/", "./"), {"__builtins__": {}}, env) for o in ops]
            exp.append((p[0] % 1.0, p[1] % 1.0, p[2] % 1.0, zz))
    exp = np.array(exp)
    hmax = 8
    refs = {}
    for h in range(-hmax, hmax + 1):
        for k in range(-hmax, hmax + 1):
            for l in range(-hmax, hmax + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                hkl = np.array([h, k, l], float)
                d = 1.0 / np.sqrt(hkl @ Gs @ hkl)
                if not (DMIN <= d <= DMAX):
                    continue
                phase = 2 * np.pi * (exp[:, 0]*h + exp[:, 1]*k + exp[:, 2]*l)
                F = np.sum(exp[:, 3] * np.exp(1j * phase))
                I = float(abs(F) ** 2)
                key = round(d, 2)
                if key in refs:
                    refs[key][0] += I
                else:
                    refs[key] = [I, (h, k, l)]
    items = [(d, v[0], v[1]) for d, v in refs.items() if v[0] > 1e-6]
    mx = max(i for _, i, _ in items)
    return sorted([(d, i / mx, hkl) for d, i, hkl in items], key=lambda t: -t[0])


def observed_profiles():
    import numpy as np
    from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline
    FIGS = "docs/paper/draft_v2/figs"
    def field_profile(name, pick):
        z = np.load(f"{FIGS}/grain_acom_v2_{name}.npz")
        cls, vac, gsum, gcnt, gscat = z["cls"], z["vac"], z["gsum"], z["gcnt"], z["gscat"]
        H = int(z["H"]); c = (H-1)/2.0; beam = max(8, round(0.11*H)); lo = beam+1
        hi = min(int(0.35/INV), 187 if name == "SI3" else 160)
        med = np.median(gscat[~vac]); sp = {}
        for cc in sorted(set(cls[~vac].tolist())):
            idx = [g for g in range(gsum.shape[0]) if cls[g] == cc and not vac[g] and gscat[g] >= med]
            if len(idx) < 2:
                continue
            avg = sum(gsum[g] for g in idx) / max(sum(gcnt[g] for g in idx), 1)
            m, v, _ = _radial_mean_var(avg, (c, c), beam_px=beam); seg = m[lo:hi]; vs = v[lo:hi]
            sp[cc] = (np.percentile(np.sqrt(np.clip(vs, 0, None))/np.clip(seg, 1e-9, None), 90), m, lo, hi)
        order = sorted(sp, key=lambda c: sp[c][0]); cc = order[0] if pick == "low" else order[-1]
        _, m, lo, hi = sp[cc]; r = np.arange(lo, hi); d = 1.0/(r*INV)
        return d, m[lo:hi]
    dP, pP = field_profile("SI3", "low")    # precursor
    dN, pN = field_profile("SI4", "high")   # mature needle
    return (dP, pP), (dN, pN)


def main():
    os.chdir("D:/DINOSR/Claude/PaperRun_claude/dino_sr_contrastive"); sys.path.insert(0, "src")
    import numpy as np
    from scipy.signal import find_peaks
    import matplotlib; matplotlib.use("Agg")
    from matplotlib.figure import Figure

    pats = {}
    for ph in ("alpha", "gamma"):
        cif = parse_cif(CIF[ph]); pats[ph] = pattern(cif)
        print(f"{ph}: a={cif['a']} b={cif['b']} c={cif['c']} angles=({cif['al']},{cif['be']},{cif['ga']}) "
              f"symops={len(cif['sym'])} atoms(asym)={len(cif['atoms'])}")
        print(f"   strong reflections (d, I) d>{DMIN}:",
              [(d, round(i, 2)) for d, i, _ in pats[ph] if i > 0.15][:12])

    (dP, pP), (dN, pN) = observed_profiles()

    def obs_peaks(d, prof):
        y = np.log(np.clip(prof, 1e-9, None)); base = np.convolve(y, np.ones(15)/15, "same")
        idx, _ = find_peaks(y - base, prominence=0.05, distance=4)
        peaks = sorted([(float(d[i]), float(prof[i])) for i in idx], key=lambda t: -t[1])
        return peaks[:6]
    print("\nObserved peaks (d, rel I):")
    for tag, (d, p) in [("precursor", (dP, pP)), ("needle", (dN, pN))]:
        pk = obs_peaks(d, p)
        print(f"  {tag}: " + ", ".join(f"{dd:.2f}A" for dd, _ in pk))
        for dd, _ in pk:
            na = min(pats["alpha"], key=lambda t: abs(t[0]-dd)); ng = min(pats["gamma"], key=lambda t: abs(t[0]-dd))
            print(f"     obs {dd:.2f}A -> alpha {na[0]:.2f}A {na[2]} (dd {abs(na[0]-dd):.2f})   "
                  f"gamma {ng[0]:.2f}A {ng[2]} (dd {abs(ng[0]-dd):.2f})")

    fig = Figure(figsize=(14, 5.4), facecolor="white")
    gs = fig.add_gridspec(1, 2, left=0.06, right=0.99, top=0.9, bottom=0.13, wspace=0.16)
    for ax_i, (tag, d, prof) in enumerate([("precursor (least-ordered class, SI3)", dP, pP),
                                           ("mature needle (SI4)", dN, pN)]):
        ax = fig.add_subplot(gs[0, ax_i])
        pn = prof / np.nanmax(prof)
        ax.semilogy(d, pn, "k-", lw=1.8, label="observed", zorder=5)
        for ph, col, off in [("alpha", "#2c7fb8", 1.0), ("gamma", "#d62728", 1.0)]:
            for dd, ii, hkl in pats[ph]:
                if ii < 0.05 or dd < d.min() or dd > d.max():
                    continue
                ax.vlines(dd, 1e-3, ii, color=col, lw=1.6*ii+0.4, alpha=0.85)
            ax.plot([], [], color=col, lw=2, label=f"{ph} (kinematical)")
        ax.set_xlabel("d-spacing (A)", fontsize=11); ax.set_ylabel("normalized intensity (log)", fontsize=11)
        ax.set_xlim(d.max(), d.min()); ax.set_ylim(max(1e-3, np.nanmin(pn)*0.5), 1.3)
        ax.set_title(tag, fontsize=11, fontweight="bold"); ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.25, which="both")
    fig.suptitle("A2  Indexing the observed classes: alpha vs gamma indomethacin (kinematical sticks over the observed radial profile)",
                 fontsize=12, fontweight="bold")
    p = "docs/paper/draft_v2/figs/Review/A2_alpha_vs_gamma.png"; fig.savefig(p, dpi=150, facecolor="white")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
