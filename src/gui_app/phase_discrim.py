"""phase_discrim.py -- tell similar phases apart by what DIFFERS between
their references (fiber mode: one known viewing direction).

Spot-count matching (dataref_core) scores every reference on all of its
spots, so phases that share most reflections -- a structure and its
superstructure, or two phases with coinciding reflections -- are told apart
by a few spots drowned in the shared ones, and an ABSENT reflection costs
almost nothing.  Here references are compared in PAIRS, on what differs:

  * every reference's 2-D lattice (basis from its spots) is fitted to each
    pattern: in-plane rotation from the spot match, then strain + offset
    by least squares (strain_core) -- all positions below come from the
    PATTERN'S OWN lattice, so they move with its strain;

  * SHARED LATTICE (one lattice is a sub-lattice of the other, e.g. a
    superstructure, or they are the same): the references are compared
    with each other directly -- each is FOLDED onto the common (coarser)
    cell: the intensity at every fractional position (i/4, j/4) of the
    cell, averaged over all cells (no peak detection, so even weak
    satellites count; each site minus its own local background mean).
    Positions where the two references differ by > z_diff sigma - and,
    by Friedel's law, so does the opposite position -f with the same
    sign - are the discriminating positions (a half-order superstructure:
    (1/2, 0)).  Each reference also
    gives the expected strength of the discriminating positions IN its
    phase, relative to the pattern's own main reflections measured the
    same way (so spot width, dose and thickness scale out; any shadow at
    those positions in the other phase is part of its expectation).  A pattern is scored by the Gaussian
    log-likelihood ratio of its measured strengths under the two
    expectations -- an ABSENT satellite is evidence too;

  * DISTINCT LATTICES (two true phases): per pattern, which reference
    matches the detected spot positions better (spot-match score, as a
    two-proportion test scaled by the number of spots);

  * a lattice counts as indexing a pattern only if it explains >= 60% of
    its detected spots; a pattern no lattice indexes is "off-axis" (another
    phase or orientation than the references); a pair whose references do
    not differ anywhere is reported (positions alone cannot separate it).

Outputs per pattern: posterior per reference, best / second best, the
weakest pairwise win of the best (margin, log-likelihood units) and a
status.  Pure numpy; works on raw counts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gui_app.dataref_core import match_vectors, spot_vectors
from gui_app.strain_core import (lattice_basis, refine_subpixel, strain_one,
                                 to_xy)

OK, OFF_AXIS, METRIC_ONLY, TOO_FEW = 0, 1, 2, 3
STATUS = {OK: "ok", OFF_AXIS: "off-axis", METRIC_ONLY: "cannot separate",
          TOO_FEW: "too few spots"}

_N_FOLD = 4                       # fractional cell positions i/4, j/4
OFFSETS = np.array([(i / _N_FOLD, j / _N_FOLD) for i in range(_N_FOLD)
                    for j in range(_N_FOLD) if (i, j) != (0, 0)])
# index of each position's Friedel partner (-f mod 1)
_FRIEDEL = np.array([int(np.flatnonzero(np.all(np.isclose(
    OFFSETS, (-f) % 1.0), axis=1))[0]) for f in OFFSETS])


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def lattice_sites(B, t, r_min, r_max, nmax=40):
    """All lattice points B·hk + t (x, y-up) with r_min <= |B·hk| <= r_max,
    and their hk."""
    h = np.arange(-nmax, nmax + 1)
    hk = np.stack(np.meshgrid(h, h, indexing="ij"), -1).reshape(-1, 2)
    xy = hk @ B.T + t
    r = np.hypot(*(xy - t).T)
    keep = (r >= r_min) & (r <= r_max)
    return xy[keep], hk[keep]


class SiteMeter:
    """Disk-mean minus local-background intensity at sub-pixel sites.

    Background = MEAN of an annulus around the site, leaving out pixels
    close to any main reflection (``avoid``, from the fitted lattice):
    with them, the tails of broad main spots raise the background; with a
    median instead of the mean, low-count Poisson noise biases every
    noisy pattern relative to a high-count reference average."""

    def __init__(self, w: float = 2.0):
        self.w = float(w)
        R = int(np.ceil(w + 5))
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        r = np.hypot(yy, xx)
        self.disk = np.stack([yy[r <= w], xx[r <= w]], 1)
        ann = (r >= w + 1.5) & (r <= w + 5)
        self.ann = np.stack([yy[ann], xx[ann]], 1)
        self.R = R
        self.r_avoid = w + 3.0

    def spot_sigma(self, pat, centre, xy):
        """Width (sigma, px) of the pattern's spots: second moment of the
        mean background-subtracted profile around the sites xy."""
        f = np.asarray(pat, np.float32)
        H, W = f.shape
        h = 6
        rows = np.rint(centre[0] - xy[:, 1]).astype(int)
        cols = np.rint(centre[1] + xy[:, 0]).astype(int)
        ok = (rows >= h) & (rows < H - h) & (cols >= h) & (cols < W - h)
        if ok.sum() < 3:
            return 1.5
        yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
        patch = f[rows[ok][:, None, None] + yy, cols[ok][:, None, None] + xx]
        m = patch.mean(0)
        border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]]).mean()
        p = np.clip(m - border, 0, None)
        r2 = yy ** 2 + xx ** 2
        sel = r2 <= 25
        if p[sel].sum() <= 0:
            return 1.5
        return float(np.clip(np.sqrt((p[sel] * r2[sel]).sum()
                                     / (2 * p[sel].sum())), 0.8, 6.0))

    def __call__(self, pat, centre, xy, avoid=None, r_avoid=None):
        """xy (x, y-up) relative to centre -> (values, valid mask).
        avoid: (M, 2) main-reflection positions (same frame) whose
        surroundings are excluded from the background (a site's own
        position is never excluded)."""
        f = np.asarray(pat, np.float32)
        H, W = f.shape
        if len(xy) == 0:
            return np.zeros(0), np.zeros(0, bool)
        rows = np.rint(centre[0] - xy[:, 1]).astype(int)
        cols = np.rint(centre[1] + xy[:, 0]).astype(int)
        R = self.R
        valid = (rows >= R) & (rows < H - R) & (cols >= R) & (cols < W - R)
        out = np.zeros(len(xy))
        if not valid.any():
            return out, valid
        r, c = rows[valid], cols[valid]
        d = f[r[:, None] + self.disk[None, :, 0],
              c[:, None] + self.disk[None, :, 1]].mean(1)
        ar = r[:, None] + self.ann[None, :, 0]
        ac = c[:, None] + self.ann[None, :, 1]
        av = f[ar, ac]
        keep = np.ones(av.shape, bool)
        if avoid is not None and len(avoid):
            from scipy.spatial import cKDTree
            tree = cKDTree(np.asarray(avoid, float))
            px = np.stack([ac - centre[1], centre[0] - ar], -1).reshape(-1, 2)
            dd, ii = tree.query(px, k=2)
            site = np.repeat(xy[valid], self.ann.shape[0], axis=0)
            own = np.hypot(*(np.asarray(avoid)[ii[:, 0]] - site).T) < 1.0
            dist = np.where(own, dd[:, 1], dd[:, 0])
            keep = (dist > (r_avoid or self.r_avoid)).reshape(av.shape)
        n = keep.sum(1)
        bg = np.where(n >= 6, (av * keep).sum(1) / np.maximum(n, 1),
                      np.median(av, axis=1))
        out[valid] = d - bg
        return out, valid


def _ring_mean(pat, centre, r_min, r_max):
    f = np.asarray(pat, np.float32)
    H, W = f.shape
    yy, xx = np.ogrid[:H, :W]
    r = np.hypot(yy - centre[0], xx - centre[1])
    m = (r >= r_min) & (r <= min(r_max, min(H, W) / 2))
    return float(max(f[m].mean(), 1e-9))


def _relation(Bf, Bc, atol=0.12):
    """Integer matrix M with Bc ≈ Bf·M (every reflection of Bc is one of
    Bf), or None."""
    try:
        M = np.linalg.solve(Bf, Bc)
    except np.linalg.LinAlgError:
        return None
    Mi = np.rint(M)
    if np.allclose(M, Mi, atol=atol) and abs(np.linalg.det(Mi)) >= 1:
        return Mi
    return None


# ---------------------------------------------------------------------------
@dataclass
class PhaseRef:
    name: str
    vecs: np.ndarray            # (M, 2) (dy, dx) reference spots
    period_deg: float
    B: np.ndarray               # (2, 2) reference lattice, x / y-up cols
    pattern: np.ndarray = None
    centre: tuple = (0.0, 0.0)
    q_ref: float = 1.0          # fraction of its own spots it indexes


@dataclass
class Model:
    refs: list
    tol: float = 3.0
    r_min: float = 0.0
    r_max: float = np.inf
    w: float = 2.0
    min_spots: int = 5
    # a lattice "indexes" a pattern only if it explains at least this
    # fraction of what it explains in its own reference, both counted
    # ABOVE CHANCE (a wrong lattice of similar spacing still hits 20-50 %
    # of the spots by coincidence)
    min_index_frac: float = 0.5
    # "spots": lattices fitted to the detected peaks (strain_core) - the
    # default.  "intensity": fitted to the pattern's intensity
    # (lattice_fit) - experimental: on EuInAs it misjudged two different
    # lattices as one and was 20x slower, so it is not used by the GUI.
    fit_mode: str = "spots"
    min_z: float = 5.0          # lattice evidence vs random rotations
    n_bright: int = 20          # off-axis rule: the N brightest spots
    pairs: dict = field(default_factory=dict)     # (a, b) -> calibration
    notes: list = field(default_factory=list)


def make_phase_ref(ref, tol=3.0, fit_mode="spots"):
    """PhaseRef from a dataref_core.DataReference (lattice from its
    sub-pixel spots)."""
    vecs = np.asarray(ref.vecs, float)
    if ref.pattern is not None and not ref.extra.get("simulated"):
        cy, cx = ref.centre
        pk = np.column_stack([vecs[:, 0] + cy, vecs[:, 1] + cx,
                              np.ones(len(vecs))])
        vecs = refine_subpixel(ref.pattern, pk)[:, :2] - np.array([cy, cx])
    B, n = lattice_basis(to_xy(vecs), tol)
    q_ref = n / max(len(vecs), 1)
    if (fit_mode == "intensity" and ref.pattern is not None
            and not ref.extra.get("simulated")):
        # refine on the reference's own intensity; q_ref then = the share
        # of its spot intensity the lattice explains (same measure as for
        # every pattern in "intensity" mode)
        from gui_app import lattice_fit as lf
        r_min = ref.extra.get("r_min", 0.0)
        r_max = ref.extra.get("r_max", np.inf)
        Bi, ti, zi, Ei = lf.fit(ref.pattern, ref.centre, B, (0.0, 0.0),
                                r_min, r_max, ref.period_deg)
        if np.all(np.isfinite(Bi)) and np.isfinite(Ei):
            B, q_ref = Bi, Ei
    return PhaseRef(ref.name, np.asarray(ref.vecs, float), ref.period_deg,
                    B, ref.pattern, tuple(ref.centre), q_ref=q_ref)


# ---------------------------------------------------------------------------
# per pattern
# ---------------------------------------------------------------------------
def _excess(q, B, tol):
    """Indexed fraction above chance: a cell of area A catches a random
    spot with probability c ~ pi tol^2 / A  ->  (q - c) / (1 - c)."""
    c = min(0.95, np.pi * tol ** 2 / max(abs(np.linalg.det(B)), 1e-9))
    return (q - c) / (1.0 - c)


def _passes(model, ref, fit):
    """Does lattice ``fit`` (B, t, q) index the pattern well enough?
    Chance-corrected and relative to how well the lattice indexes its OWN
    reference (spurious spots - detector seams, streaks - lower both)."""
    if fit is None:
        return False
    if model.fit_mode == "intensity":
        # fit = (B, t, explained, z): the lattice must stand out from
        # random rotations AND explain a good share of what it explains
        # in its own reference
        return (np.isfinite(fit[3]) and fit[3] >= model.min_z
                and fit[2] >= model.min_index_frac * max(ref.q_ref, 0.1))
    need = model.min_index_frac * max(_excess(ref.q_ref, ref.B, model.tol),
                                      0.2)
    return _excess(fit[2], fit[0], model.tol) >= need


def fit_all(model, pat, peaks, centre, gate=True):
    """Fit every reference lattice to one pattern -> (list of (B, t, q)
    or None per reference, q = fraction of the detected spots it indexes;
    the detected vectors)."""
    v = spot_vectors(peaks, centre, model.r_min, model.r_max)
    if len(v) < model.min_spots:
        return [None] * len(model.refs), v
    pk = np.column_stack([v[:, 0] + centre[0], v[:, 1] + centre[1],
                          np.ones(len(v))])
    xy = to_xy(refine_subpixel(pat, pk)[:, :2] - np.asarray(centre))
    # intensity of each ring spot (same selection as spot_vectors)
    p_ = np.asarray(peaks, float).reshape(-1, 3)
    r_ = np.hypot(*(p_[:, :2] - np.asarray(centre, float)).T)
    w = np.clip(p_[(r_ >= model.r_min) & (r_ <= model.r_max), 2], 0, None)
    if len(w) != len(xy) or w.sum() <= 0:
        w = np.ones(len(xy))
    fits = []
    imgs = None
    for ref in model.refs:
        m = match_vectors(ref.vecs, v, model.tol, ref.period_deg)
        if not np.isfinite(m["theta_deg"]):
            fits.append(None)
            continue
        if model.fit_mode == "intensity":
            from gui_app import lattice_fit as lf
            if imgs is None:
                imgs = (lf.SpotImage(pat, 3.0), lf.SpotImage(pat, 1.5))
            B0 = lf.rot2(m["theta_deg"]) @ ref.B
            B, tt, z, E = lf.fit(pat, centre, B0, (0.0, 0.0), model.r_min,
                                 model.r_max, ref.period_deg, imgs=imgs)
            f = (B, tt, E, z) if np.all(np.isfinite(B)) else None
            fits.append(f if (not gate or _passes(model, ref, f)) else None)
            continue
        s = strain_one(ref.B, xy, m["theta_deg"], model.tol, model.min_spots)
        if not np.all(np.isfinite(s["B"])):
            fits.append(None)
            continue
        # share of the BRIGHTEST spots the lattice indexes: in a noisy
        # pattern of the right phase the strong spots are lattice spots
        # (noise peaks are many but weak); a different phase leaves strong
        # spots unexplained.  (A share of ALL detected spots or of their
        # total intensity falls with dose and calls noisy patterns
        # off-axis.)
        B, tt = s["B"], s["t"]
        hk = np.rint(np.linalg.solve(B, (xy - tt).T).T)
        hit = (np.hypot(*(hk @ B.T + tt - xy).T) < model.tol) & (
            np.abs(hk).sum(1) > 0)
        top = np.argsort(-w)[:min(len(w), model.n_bright)]
        q = float(hit[top].mean())
        f = (B, tt, q)
        fits.append(f if (not gate or _passes(model, ref, f)) else None)
    return fits, v


def fold(model, meter, pat, centre, B, t):
    """Site intensities at every fractional cell position (OFFSETS) of
    lattice (B, t) -> (list of value arrays, scale).

    scale = the mean intensity of the pattern's OWN main reflections
    (offset (0, 0)), measured with the same window: a satellite and its
    main spots broaden together, so their ratio does not depend on the
    spot width (a fixed window catches less of a broad spot), nor on dose
    or thickness.  Falls back to the ring mean if the main spots are not
    above background."""
    main, _ = lattice_sites(B, t, 0.0, model.r_max + 20)
    main = np.vstack([main, t[None, :]])          # + the direct beam
    xy0, _ = lattice_sites(B, t, model.r_min, model.r_max)
    # background pixels stay clear of the main spots' tails: 3 sigma of
    # THIS pattern's spot width
    r_av = max(meter.w + 3.0, 3.0 * meter.spot_sigma(pat, centre, xy0))
    raw = []
    for f in OFFSETS:
        xy, _ = lattice_sites(B, t + B @ f, model.r_min, model.r_max)
        v, ok = meter(pat, centre, xy, avoid=main, r_avoid=r_av)
        raw.append(v[ok])
    v0, ok0 = meter(pat, centre, xy0, avoid=main, r_avoid=r_av)
    main = float(np.mean(v0[ok0])) if ok0.any() else 0.0
    ring = _ring_mean(pat, centre, model.r_min, model.r_max)
    return raw, (main if main > 0.05 * ring else ring)


def fold_stats(raw, F, ctrl=None):
    """Per position: mean site intensity relative to the scale F (the
    pattern's own main reflections), and its standard error (that
    position's spread over the cells / sqrt(n)).

    Every site already has its LOCAL background removed (SiteMeter),
    and as a ratio to the main reflections the value does not change when
    spots broaden or a reference average is blurred.  No global baseline
    is subtracted: a mean over "control" positions would include the
    tails of the main spots and shift with the spot width."""
    def _trim(r):
        # drop the most extreme 10 % at each end: sites hit by a detector
        # seam / streak / dead region must not skew the position's mean
        r = np.sort(np.asarray(r, float))
        k = int(0.1 * len(r))
        return r[k:len(r) - k] if len(r) - 2 * k >= 3 else r
    tr = [_trim(r) for r in raw]
    val = np.array([np.mean(r) / F if len(r) else np.nan for r in tr])
    noi = np.array([np.std(r) / np.sqrt(len(r)) / F if len(r) > 1
                    else np.inf for r in tr])
    return val, noi + 1e-12


def calibrate(model, patterns_by_ref, z_diff: float = 4.0):
    """Work out, from the references themselves, HOW every pair differs
    (patterns_by_ref: (pattern, peaks, centre) per reference)."""
    meter = SiteMeter(model.w)
    K = len(model.refs)
    # no indexing-fraction gate here: with strong satellites the main
    # lattice explains only part of a superstructure reference's spots,
    # and the sub-lattice relation must still be found
    fr = [fit_all(model, *patterns_by_ref[k], gate=False)[0]
          for k in range(K)]
    # how much of its OWN reference's spot intensity each lattice explains
    # (the off-axis rule is relative to this)
    if model.fit_mode == "spots":
        for k in range(K):
            if fr[k][k] is not None:
                model.refs[k].q_ref = fr[k][k][2]
    allc = np.ones(len(OFFSETS), bool)
    for a in range(K):
        for b in range(a + 1, K):
            na, nb = model.refs[a].name, model.refs[b].name
            ent = dict(mode="distinct")
            if fr[a][a] is not None and fr[a][b] is not None:
                Ba, Bb = fr[a][a][0], fr[a][b][0]
                in_a, in_b = _relation(Ba, Bb), _relation(Bb, Ba)
                if in_a is not None or in_b is not None:
                    # the coarser lattice (fewer reflections) is the cell
                    coarse = b if (in_a is not None and in_b is None) else a
                    ent = dict(mode="shared", coarse=coarse)
            if ent["mode"] == "shared":
                c = ent["coarse"]
                folds = {}
                for k in (a, b):
                    if fr[k][c] is None:
                        folds = None
                        break
                    pat, _pk, cen = patterns_by_ref[k]
                    folds[k] = fold(model, meter, pat, cen, fr[k][c][0],
                                    fr[k][c][1])
                if folds is None:
                    ent = dict(mode="distinct")
                else:
                    va, sa = fold_stats(*folds[a], allc)
                    vb, sb = fold_stats(*folds[b], allc)
                    dv = va - vb
                    z = np.abs(dv) / np.sqrt(sa ** 2 + sb ** 2)
                    disc = np.isfinite(z) & (z > z_diff)
                    # Friedel's law: a real difference at cell position f
                    # is also at -f, with the same sign -> drop the rest
                    disc = disc & disc[_FRIEDEL] & (
                        np.sign(dv) == np.sign(dv[_FRIEDEL]))
                    # a position right next to a STRONGER differing one is
                    # its tail, not a reflection of its own
                    Bc = fr[a][c][0]
                    # strongest difference first (by size, not
                    # significance: a big satellite outranks the tail
                    # artifacts next to it)
                    order = np.argsort(-np.where(disc, np.abs(dv), -np.inf))
                    kept = []
                    for i in order:
                        if not disc[i]:
                            break
                        far = True
                        for j in kept:
                            df = (OFFSETS[i] - OFFSETS[j] + 0.5) % 1.0 - 0.5
                            if np.hypot(*(Bc @ df)) < 2 * meter.w + 3:
                                far = False
                                break
                        if far:
                            kept.append(i)
                    disc = np.zeros_like(disc)
                    disc[kept] = True
                    # side effects of strong reflections (e.g. their tails
                    # in a neighbour's background) are small: keep only
                    # differences >= 10 % of the strongest one
                    if disc.any():
                        disc &= np.abs(dv) >= 0.1 * np.abs(dv[disc]).max()
                    disc = disc & disc[_FRIEDEL]
                    ctrl = ~disc
                    va, _ = fold_stats(*folds[a], ctrl)
                    vb, _ = fold_stats(*folds[b], ctrl)
                    ent.update(disc=disc, ctrl=ctrl, mu={a: va, b: vb},
                               z=z, folds=folds)
                    if disc.any():
                        pos = ", ".join(f"({f[0]:g}, {f[1]:g})"
                                        for f in OFFSETS[disc])
                        model.notes.append(
                            f"'{na}' vs '{nb}': same main lattice; they "
                            f"differ at cell position(s) {pos}.")
                    else:
                        model.notes.append(
                            f"'{na}' vs '{nb}': same lattice and no cell "
                            f"position differs by > {z_diff:g} sigma - "
                            f"they cannot be told apart.")
            if ent["mode"] == "distinct":
                why = ""
                if fr[a][a] is not None and fr[a][b] is not None:
                    M = np.linalg.solve(fr[a][a][0], fr[a][b][0])
                    why = (f" (cell relation {np.round(M, 2).tolist()} is not "
                           f"integer)")
                model.notes.append(f"'{na}' vs '{nb}': different lattices"
                                   f"{why} - decided by which indexes the "
                                   f"spots.")
            model.pairs[(a, b)] = ent
    return model


def pair_llr(model, meter, pat, centre, fits, a, b, cache, raw_fits=None):
    """log P(pattern | a) - log P(pattern | b); None if undecidable."""
    cap = 20.0
    ent = model.pairs.get((a, b), {})
    if ent.get("mode") == "shared":
        if not ent["disc"].any():
            return None
        c = ent["coarse"]
        if fits[a] is None and fits[b] is None:
            return None
        fc = (raw_fits or fits)[c]
        if fc is None:
            return None
        key = ("fold", c)
        if key not in cache:
            cache[key] = fold(model, meter, pat, centre, fc[0], fc[1])
        val, noi = fold_stats(*cache[key], ent["ctrl"])
        d = ent["disc"] & np.isfinite(val)
        if not d.any():
            return None
        s, sd = val[d], noi[d]
        mu_a, mu_b = ent["mu"][a][d], ent["mu"][b][d]
        # natural spread of each phase (from its reference's individual
        # patterns, member_levels); 0 when only the average is known
        sp = ent.get("spread", {})
        va = sd ** 2 + sp.get(a, np.zeros_like(ent["mu"][a]))[d] ** 2
        vb = sd ** 2 + sp.get(b, np.zeros_like(ent["mu"][b]))[d] ** 2
        llr = float(np.sum((s - mu_b) ** 2 / (2 * vb) + 0.5 * np.log(vb)
                           - (s - mu_a) ** 2 / (2 * va) - 0.5 * np.log(va)))
        return float(np.clip(llr, -cap, cap))
    # different lattices: the spot-match score (F1 of spot positions),
    # as a two-proportion test scaled by the number of spots
    fa, fb = fits[a], fits[b]
    if fa is None and fb is None:
        return None
    if (fa is None) != (fb is None):
        return cap if fb is None else -cap
    if model.fit_mode == "intensity":
        # share of the spot intensity each lattice explains, relative to
        # its own reference; 0.06 difference ~ log-likelihood ratio 3
        ra = fa[2] / max(model.refs[a].q_ref, 0.1)
        rb = fb[2] / max(model.refs[b].q_ref, 0.1)
        return float(np.clip(50.0 * (ra - rb), -cap, cap))
    v = cache["v"]
    sa = match_vectors(model.refs[a].vecs, v, model.tol,
                       model.refs[a].period_deg)["score"]
    sb = match_vectors(model.refs[b].vecs, v, model.tol,
                       model.refs[b].period_deg)["score"]
    sm = 0.5 * (sa + sb)
    zq = (sa - sb) * np.sqrt(cache["n_spots"]) / np.sqrt(
        max(sm * (1 - sm), 0.02))
    return float(np.clip(np.sign(zq) * zq ** 2 / 2, -cap, cap))


def classify(model, pat, peaks, centre, meter=None):
    """-> dict(post (K,), best, second, margin, status, llr (K, K), fits)."""
    meter = meter or SiteMeter(model.w)
    K = len(model.refs)
    # every lattice fitted without the indexing gate (a superstructure
    # pattern's main lattice explains only part of its spots but still
    # places the satellite positions); the gated list decides off-axis
    # and the different-lattice pairs
    raw_fits, v = fit_all(model, pat, peaks, centre, gate=False)
    fits = [f if _passes(model, r, f) else None
            for f, r in zip(raw_fits, model.refs)]
    # why a lattice was (not) accepted: (indexed share above chance, needed)
    res_gate = []
    for f, r in zip(raw_fits, model.refs):
        if f is None or model.fit_mode != "spots":
            res_gate.append(None)
            continue
        res_gate.append((_excess(f[2], f[0], model.tol),
                         model.min_index_frac * max(
                             _excess(r.q_ref, r.B, model.tol), 0.2)))
    res = dict(post=np.full(K, np.nan), best=-1, second=-1, margin=np.nan,
               status=OK, llr=np.full((K, K), np.nan), fits=fits,
               gate=res_gate, raw_fits=raw_fits)
    if len(v) < model.min_spots:
        res["status"] = TOO_FEW
        return res
    if all(f is None for f in fits):
        res["status"] = OFF_AXIS
        return res
    L = np.zeros((K, K))
    undecided = False
    cache = {"n_spots": len(v), "v": v}
    for a in range(K):
        for b in range(a + 1, K):
            x = pair_llr(model, meter, pat, centre, fits, a, b, cache,
                         raw_fits)
            if x is None:
                undecided = True
                x = 0.0
            L[a, b], L[b, a] = x, -x
    res["llr"] = L
    res["sat"] = satellite_strength(model, cache)
    s = L.sum(1) / max(K - 1, 1)
    p = np.exp(s - s.max())
    p /= p.sum()
    order = np.argsort(-p)
    res["raw_fits"] = raw_fits
    res.update(post=p, best=int(order[0]),
               second=int(order[1]) if K > 1 else -1,
               margin=float(min(L[order[0], j] for j in range(K)
                                if j != order[0])) if K > 1 else np.inf)
    if undecided and res["margin"] == 0.0:
        res["status"] = METRIC_ONLY
    return res


# ---------------------------------------------------------------------------
# satellite strength: an absolute measure, independent of reference purity
# ---------------------------------------------------------------------------
def _sat_pair(model):
    """The first shared-lattice pair with discriminating positions ->
    (pair entry, mask of the positions where the satellite-RICH reference
    is the stronger one), or (None, None)."""
    for (a, b), ent in model.pairs.items():
        if ent.get("mode") != "shared" or not ent["disc"].any():
            continue
        d = ent["disc"]
        rich = a if np.nanmean(ent["mu"][a][d]) >= np.nanmean(
            ent["mu"][b][d]) else b
        poor = b if rich == a else a
        return ent, d & (ent["mu"][rich] > ent["mu"][poor]), rich, poor
    return None, None, None, None


def satellite_strength(model, cache):
    """Mean intensity at the satellite positions (those where the
    satellite-rich reference is stronger) relative to the pattern's own
    main reflections, and its standard error -> (S, SE) or (nan, nan).
    Uses the fold already measured for the classification (cache)."""
    ent, d, _r, _p = _sat_pair(model)
    if ent is None or not d.any():
        return (np.nan, np.nan)
    key = ("fold", ent["coarse"])
    if key not in cache:
        return (np.nan, np.nan)
    val, noi = fold_stats(*cache[key], ent["ctrl"])
    d = d & np.isfinite(val) & np.isfinite(noi)
    if not d.any():
        return (np.nan, np.nan)
    return (float(np.mean(val[d])),
            float(np.sqrt(np.sum(noi[d] ** 2)) / d.sum()))


def two_populations(S, p_min=0.95, n_iter=300):
    """Split per-pixel satellite strengths S into a weak and a strong
    population: 1-D two-Gaussian mixture (EM), compared with one Gaussian
    by BIC.  -> dict(lo / hi = (weight, mean, sd), thr = where the two
    posteriors are equal, bimodal = BIC prefers two, p_hi = function
    S -> P(strong)).  A pixel is strong / weak when that posterior is >=
    p_min, otherwise uncertain."""
    x = np.asarray(S, float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        raise ValueError("too few pixels with a satellite measurement")
    lo_, hi_ = np.percentile(x, [1, 99])
    x = x[(x >= lo_ - 0.5 * (hi_ - lo_)) & (x <= hi_ + 0.5 * (hi_ - lo_))]
    def npdf(v, m, s):
        return np.exp(-0.5 * ((v - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    w = np.array([0.5, 0.5])
    m = np.percentile(x, [25, 75]).astype(float)
    s0 = max(float(np.std(x)), 1e-9)
    sd = np.array([s0, s0]) / 2
    for _ in range(n_iter):
        p = np.stack([w[k] * npdf(x, m[k], sd[k]) for k in (0, 1)]) + 1e-300
        r = p / p.sum(0)
        nk = r.sum(1)
        w_n = nk / nk.sum()
        m_n = (r * x).sum(1) / nk
        sd_n = np.sqrt((r * (x - m_n[:, None]) ** 2).sum(1) / nk)
        sd_n = np.maximum(sd_n, 1e-3 * s0)
        done = np.allclose(m_n, m, atol=1e-7 * s0)
        w, m, sd = w_n, m_n, sd_n
        if done:
            break
    o = np.argsort(m)
    w, m, sd = w[o], m[o], sd[o]
    ll2 = np.log(sum(w[k] * npdf(x, m[k], sd[k]) for k in (0, 1))
                 + 1e-300).sum()
    ll1 = np.log(npdf(x, x.mean(), s0) + 1e-300).sum()
    bic2 = -2 * ll2 + 5 * np.log(x.size)
    bic1 = -2 * ll1 + 2 * np.log(x.size)

    def p_hi(v):
        v = np.asarray(v, float)
        a = w[0] * npdf(v, m[0], sd[0])
        b = w[1] * npdf(v, m[1], sd[1])
        out = b / np.maximum(a + b, 1e-300)
        # beyond the strong mean it stays strong, below the weak mean weak
        out = np.where(v > m[1], 1.0, np.where(v < m[0], 0.0, out))
        return np.where(np.isfinite(v), out, np.nan)
    g = np.linspace(m[0], m[1], 2001)
    ph = p_hi(g)
    thr = float(g[np.argmin(np.abs(ph - 0.5))])
    return dict(lo=(w[0], m[0], sd[0]), hi=(w[1], m[1], sd[1]), thr=thr,
                bimodal=bool(bic2 < bic1 - 10), p_hi=p_hi, p_min=p_min,
                n=int(x.size))


def member_levels(model, items, min_n=5):
    """Expected strengths from a reference's INDIVIDUAL patterns.

    An averaged reference under-measures its satellites: spots that move
    a little between patterns (rotation, strain, beam drift) smear in the
    average, and the smeared main spots spill into a weak satellite's
    local background (synthetic test: 0.07 in the average vs 0.125 in the
    patterns).  So the level of every cell position is the MEDIAN over the
    reference's own patterns, each measured on its own fitted lattice -
    exactly how every scan pattern is measured - and the natural spread
    between them (MAD) enters the likelihood.  The discriminating
    positions themselves still come from comparing the averages.

    items: {ref index: [(pattern, peaks, centre), ...]}.  Updates
    model.pairs in place; returns {ref: number of patterns used}."""
    meter = SiteMeter(model.w)
    used = {}
    for (a, b), ent in model.pairs.items():
        if ent.get("mode") != "shared":
            continue
        c = ent["coarse"]
        ent.setdefault("spread", {})
        ent.setdefault("mu_avg", {k: ent["mu"][k].copy() for k in (a, b)})
        for k in (a, b):
            vals = []
            for pat, pk, cen in items.get(k, []):
                raw, _v = fit_all(model, pat, pk, cen, gate=False)
                f = raw[c]
                if f is None:
                    continue
                val, _n = fold_stats(*fold(model, meter, pat, cen, f[0],
                                           f[1]), ent["ctrl"])
                vals.append(val)
            if len(vals) < min_n:
                continue
            V = np.array(vals)
            med = np.nanmedian(V, 0)
            mad = 1.4826 * np.nanmedian(np.abs(V - med), 0)
            ent["mu"][k] = med
            ent["spread"][k] = np.where(np.isfinite(mad), mad, 0.0)
            used[k] = len(vals)
    return used


# ---------------------------------------------------------------------------
# satellite-free domain: a reference-free test on the 1-D profile along the
# doubling direction
# ---------------------------------------------------------------------------
def _tails(x, A1, A2, w, p, C):
    """Two main-peak tails (generalised Lorentzians at x = 0 and x = 1,
    in units of the main spacing) plus a constant."""
    return (A1 / (1 + np.abs(x / w) ** p) + A2 / (1 + np.abs((x - 1) / w) ** p)
            + C)


def half_order_test(s, v, G, hw=None, k_min=1):
    """Is there a peak at the half-order positions (k + 1/2) G of a 1-D
    profile v(s) along a lattice direction (s in px, G = main spacing)?

    Between two main peaks the profile is the sum of their tails, so a
    straight or polynomial baseline is biased (it hides or invents a weak
    satellite).  The stretch between the peaks (0.15 G in from each) is
    fitted with the physical shape - two generalised-Lorentzian tails and
    a constant - leaving out +-1.5 hw around the half-order position; the
    satellite excess is measured against that fit and the noise from the
    residuals of the same fit.  -> dict(z = combined z-score (Stouffer
    over the orders), ratio = mean excess / mean main-peak excess,
    orders = [(k, excess, z)], sigma)."""
    from scipy.optimize import curve_fit
    s = np.asarray(s, float); v = np.asarray(v, float)
    hw = max(1.0, min(0.12 * G, hw if hw else 0.12 * G))
    orders, resid, main = [], [], []
    k = k_min
    while (k + 1) * G - 0.15 * G < s.max():
        c = (k + 0.5) * G
        reg = (s >= k * G + 0.15 * G) & (s <= (k + 1) * G - 0.15 * G)
        pk = np.abs(s - c) <= hw
        fitm = reg & (np.abs(s - c) > 1.5 * hw)
        if fitm.sum() < 8 or pk.sum() < 1:
            k += 1
            continue
        x = (s - k * G) / G
        vmax = max(v[fitm].max(), 1e-12)
        try:
            coef, _ = curve_fit(
                _tails, x[fitm], v[fitm] / vmax,
                p0=[1.0, 1.0, 0.1, 2.0, 0.0],
                bounds=([0, 0, 0.02, 1.0, 0], [50, 50, 1.0, 4.0, 1.0]),
                maxfev=4000)
            base = vmax * _tails(x, *coef)
        except Exception:
            # fall back: quadratic in log v
            A = np.vstack([x ** 2, x, np.ones_like(x)]).T
            lv = np.log(np.clip(v, 1e-12, None))
            cq = np.linalg.lstsq(A[fitm], lv[fitm], rcond=None)[0]
            base = np.exp(A @ cq)
        ex = float(np.mean(v[pk] - base[pk]))
        resid.extend((v - base)[fitm].tolist())
        orders.append([k, ex, int(pk.sum()), int(fitm.sum())])
        # main peak k (for the ratio): height over its flanks
        cm = k * G
        fl = ((np.abs(s - cm) >= 0.15 * G) & (np.abs(s - cm) <= 0.3 * G))
        pm = np.abs(s - cm) <= hw
        if fl.sum() >= 4 and pm.sum() >= 1:
            main.append(float(np.mean(v[pm]) - np.mean(v[fl])))
        k += 1
    if not orders or len(resid) < 8:
        return dict(z=np.nan, ratio=np.nan, orders=[], sigma=np.nan)
    sigma = float(np.std(resid))
    out = []
    for (k, ex, npk, nfl) in orders:
        se = sigma * np.sqrt(1.0 / npk + 1.0 / nfl)
        out.append((k, ex, ex / max(se, 1e-12)))
    z = float(sum(o[2] for o in out) / np.sqrt(len(out)))
    m = float(np.mean(main)) if main else np.nan
    ratio = float(np.mean([o[1] for o in out]) / m) if m and m > 0         else np.nan
    return dict(z=z, ratio=ratio, orders=out, sigma=sigma)


def satfree_stop(N, Z, z_thr=3.0, run=3):
    """First N at which the half-order z-score reaches z_thr and stays
    there for `run` consecutive checkpoints -> the number of weakest
    frames that are satellite-free (None when it never does)."""
    Z = np.asarray(Z, float)
    for i in range(len(Z)):
        if i + run > len(Z):
            return None
        if np.all(Z[i:i + run] >= z_thr):
            return int(N[i])
    return None
