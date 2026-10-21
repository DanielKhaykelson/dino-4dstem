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


@dataclass
class Model:
    refs: list
    tol: float = 3.0
    r_min: float = 0.0
    r_max: float = np.inf
    w: float = 2.0
    min_spots: int = 5
    # a lattice "indexes" a pattern only if it explains at least this
    # fraction of its detected spots (a wrong lattice of similar spacing
    # still hits 20-50% by coincidence)
    min_index_frac: float = 0.6
    pairs: dict = field(default_factory=dict)     # (a, b) -> calibration
    notes: list = field(default_factory=list)


def make_phase_ref(ref, tol=3.0):
    """PhaseRef from a dataref_core.DataReference (lattice from its
    sub-pixel spots)."""
    vecs = np.asarray(ref.vecs, float)
    if ref.pattern is not None and not ref.extra.get("simulated"):
        cy, cx = ref.centre
        pk = np.column_stack([vecs[:, 0] + cy, vecs[:, 1] + cx,
                              np.ones(len(vecs))])
        vecs = refine_subpixel(ref.pattern, pk)[:, :2] - np.array([cy, cx])
    B, _ = lattice_basis(to_xy(vecs), tol)
    return PhaseRef(ref.name, np.asarray(ref.vecs, float), ref.period_deg,
                    B, ref.pattern, tuple(ref.centre))


# ---------------------------------------------------------------------------
# per pattern
# ---------------------------------------------------------------------------
def fit_all(model, pat, peaks, centre):
    """Fit every reference lattice to one pattern -> (list of (B, t, q)
    or None per reference, q = fraction of the detected spots it indexes;
    the detected vectors)."""
    v = spot_vectors(peaks, centre, model.r_min, model.r_max)
    if len(v) < model.min_spots:
        return [None] * len(model.refs), v
    pk = np.column_stack([v[:, 0] + centre[0], v[:, 1] + centre[1],
                          np.ones(len(v))])
    xy = to_xy(refine_subpixel(pat, pk)[:, :2] - np.asarray(centre))
    fits = []
    for ref in model.refs:
        m = match_vectors(ref.vecs, v, model.tol, ref.period_deg)
        if not np.isfinite(m["theta_deg"]):
            fits.append(None)
            continue
        s = strain_one(ref.B, xy, m["theta_deg"], model.tol, model.min_spots)
        if not np.all(np.isfinite(s["B"])):
            fits.append(None)
            continue
        q = s["n"] / len(xy)
        fits.append((s["B"], s["t"], q) if q >= model.min_index_frac
                    else None)
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
    val = np.array([np.mean(r) / F if len(r) else np.nan for r in raw])
    noi = np.array([np.std(r) / np.sqrt(len(r)) / F if len(r) > 1
                    else np.inf for r in raw])
    return val, noi + 1e-12


def calibrate(model, patterns_by_ref, z_diff: float = 4.0):
    """Work out, from the references themselves, HOW every pair differs
    (patterns_by_ref: (pattern, peaks, centre) per reference)."""
    meter = SiteMeter(model.w)
    K = len(model.refs)
    fr = [fit_all(model, *patterns_by_ref[k])[0] for k in range(K)]
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
                model.notes.append(f"'{na}' vs '{nb}': different lattices "
                                   f"- decided by which indexes the spots.")
            model.pairs[(a, b)] = ent
    return model


def pair_llr(model, meter, pat, centre, fits, a, b, cache):
    """log P(pattern | a) - log P(pattern | b); None if undecidable."""
    cap = 20.0
    ent = model.pairs.get((a, b), {})
    if ent.get("mode") == "shared":
        if not ent["disc"].any():
            return None
        c = ent["coarse"]
        fc = fits[c] if fits[c] is not None else fits[b if c == a else a]
        if fc is None:
            return None
        key = ("fold", c if fits[c] is not None else -1 - c)
        if key not in cache:
            cache[key] = fold(model, meter, pat, centre, fc[0], fc[1])
        val, noi = fold_stats(*cache[key], ent["ctrl"])
        d = ent["disc"] & np.isfinite(val)
        if not d.any():
            return None
        s, sd = val[d], noi[d]
        mu_a, mu_b = ent["mu"][a][d], ent["mu"][b][d]
        llr = float(np.sum(((s - mu_b) ** 2 - (s - mu_a) ** 2)
                           / (2 * sd ** 2)))
        return float(np.clip(llr, -cap, cap))
    # different lattices: the spot-match score (F1 of spot positions),
    # as a two-proportion test scaled by the number of spots
    fa, fb = fits[a], fits[b]
    if fa is None and fb is None:
        return None
    if (fa is None) != (fb is None):
        return cap if fb is None else -cap
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
    fits, v = fit_all(model, pat, peaks, centre)
    res = dict(post=np.full(K, np.nan), best=-1, second=-1, margin=np.nan,
               status=OK, llr=np.full((K, K), np.nan), fits=fits)
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
            x = pair_llr(model, meter, pat, centre, fits, a, b, cache)
            if x is None:
                undecided = True
                x = 0.0
            L[a, b], L[b, a] = x, -x
    res["llr"] = L
    s = L.sum(1) / max(K - 1, 1)
    p = np.exp(s - s.max())
    p /= p.sum()
    order = np.argsort(-p)
    res.update(post=p, best=int(order[0]),
               second=int(order[1]) if K > 1 else -1,
               margin=float(min(L[order[0], j] for j in range(K)
                                if j != order[0])) if K > 1 else np.inf)
    if undecided and res["margin"] == 0.0:
        res["status"] = METRIC_ONLY
    return res
