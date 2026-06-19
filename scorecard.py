"""
scorecard.py — automated validity gate for DINO-SR runs.

Emits APPROVE / RETUNE / REJECT per trained run, plus a component breakdown
and retune hints. Calibrated so the 5 approved samples in DINO_PAPER_V2/
score APPROVE and the known-rejected runs from claude_tuning_log.md score
RETUNE or REJECT.

Two entry points:

1. Standalone CLI (recomputes everything from checkpoint + dataset):
     python scorecard.py --sample Na007b
     python scorecard.py --all
     python scorecard.py --calibrate

2. In-process function, called from eval_all.py:evaluate_sample():
     from scorecard import compute_scorecard_from_arrays
     result = compute_scorecard_from_arrays(...)
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import traceback
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import label as nd_label
from sklearn.metrics import normalized_mutual_info_score


# ─────────────────────────────────────────────────────────────────────────────
# Sample registry — extends eval_all.py:SAMPLES with type + layer info
# ─────────────────────────────────────────────────────────────────────────────
# Layered samples need the cross-layer mixing check; non-layered samples use
# connected-component compactness instead.

SAMPLE_TYPES = {
    "Na007a":       {"sample_type": "non_layered"},
    "Na007b":       {"sample_type": "non_layered"},
    "Na006a":       {"sample_type": "non_layered"},
    "EuInAs_B100":  {"sample_type": "layered", "layer_bounds": [22, 44]},
    "IMC_50nm_SI2": {"sample_type": "non_layered"},
    "IMC_150nm_SI5":{"sample_type": "non_layered"},
}

# Runs that must verdict APPROVE for calibration.
CALIB_APPROVED = [
    ("Na007b",       "heat_040_050"),
    ("Na006a",       "heat_040_070"),
    ("EuInAs_B100",  "heat_040_070"),
    ("IMC_50nm_SI2", "const_065"),
    ("IMC_150nm_SI5","const_040"),
]

# Runs that must NOT verdict APPROVE for calibration (RETUNE or REJECT).
# Derived from claude_tuning_log.md rejected runs; must have a checkpoint on disk.
CALIB_REJECTED = [
    ("IMC_150nm_SI5", "const_065"),
    ("IMC_150nm_SI5", "const_050"),
    ("Na007b",        "heat_040_060"),
]

BASE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "DINO_PAPER_V2")
WEIGHTS_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "scorecard_weights.json")


# ─────────────────────────────────────────────────────────────────────────────
# Default weights and thresholds — overwritten by scorecard_weights.json
# when --calibrate has been run.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    # v4 rebalance: user's approval signal emphasizes class *richness* +
    # confidence + internal purity more than pure spatial smoothness. Under
    # the v3 all-1.0 weights, a fragmented-but-clean run scored higher than
    # the approved runs — the opposite of user judgment. Rebalanced:
    #   ×2.0 for conf_score, class_purity, mdp_distinctness
    #   ×0.5 for coherence_score, grain_compactness
    #   ×1.0 for effk_score
    # mdp_distinctness bumped to 2.0 after Na006a showed background over-
    # split into two near-duplicate prototypes: the metric detected it
    # (0.944 → 0.820) but its weight of 1.0 didn't move 'overall' enough
    # to prevent that checkpoint from winning best.pth.
    "weights_common": {
        "conf_score":         2.0,   # ↑ user values decisive assignments
        "effk_score":         1.0,
        "coherence_score":    0.5,   # ↓ user OK with local noise
        "mdp_distinctness":   2.0,   # ↑ strong penalty for over-split
        "class_purity":       2.0,   # ↑ user values clean class membership
    },
    "weights_non_layered": {
        "grain_compactness":  0.5,   # ↓ user OK with fragmented appearance
        "no_speckle":         1.0,
    },
    "weights_layered": {
        "layer_purity":          1.0,   # MIN across classes
        "no_cross_layer_mix":    1.0,
        "intra_layer_distinct":  1.0,   # per-layer over-segmentation
    },
    "threshold_approve": 0.80,
    "threshold_retune":  0.60,
    # Critical-failure gates (v2: tightened layered gates)
    "effk_collapse_margin":  0.1,
    "min_coherence":         0.30,
    "min_conf":              0.50,
    "min_layer_purity":      0.70,   # any class with <70% in its home layer → REJECT
    "min_no_cross_mix":      0.80,
    "min_intra_layer_dist":  0.50,
    "min_class_purity":      0.40,
}


def load_settings() -> dict:
    """Load calibrated weights if available, else defaults."""
    if os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH) as f:
            return json.load(f)
    return DEFAULTS


# ─────────────────────────────────────────────────────────────────────────────
# Component metric computations
# ─────────────────────────────────────────────────────────────────────────────

def _score_effk(eff_k: float, num_prototypes: int) -> float:
    """Flat-top score: 1.0 across [2, K-1], ramps to 0 only at extremes.

    User guidance: K should happen naturally. The old bell curve peaked at K/2
    penalized true 3-phase and 8-phase samples equally. This flat-top only
    flags the genuine failure modes: collapse (effK→1) and over-saturation
    (effK→K, all prototypes used uniformly → model used every slot)."""
    K = float(num_prototypes)
    if eff_k <= 1.0 or eff_k >= K:
        return 0.0
    # Ramp up from collapse: 1.0 → 2.0 maps 0 → 1
    if eff_k < 2.0:
        return float(eff_k - 1.0)
    # Ramp down from saturation: K-1 → K maps 1 → 0
    if eff_k > K - 1.0:
        return float(K - eff_k)
    # Flat top across the wide middle
    return 1.0


def _score_coherence(class_map: np.ndarray) -> float:
    """Fraction of pixels whose class matches >=5 of their 8 neighbors.
    High = spatially coherent. Low = salt-and-pepper."""
    H, W = class_map.shape
    if H < 3 or W < 3:
        return 1.0
    # 8-neighbor agreement count
    pad = np.pad(class_map, 1, mode='edge')
    agree = np.zeros((H, W), dtype=np.int32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            agree += (pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] == class_map).astype(np.int32)
    return float((agree >= 5).mean())


def _score_mdp_distinctness(cos_sim: np.ndarray, class_map: np.ndarray,
                             class_ids: list, sim_thresh: float = 0.95,
                             centroid_dist_frac: float = 0.20) -> float:
    """Over-segmentation detector.

    For each pair of classes with MDP cosine similarity > sim_thresh, check
    whether their spatial centroids are far apart (> centroid_dist_frac of the
    scan diagonal). Near-duplicate MDPs in distinct scan regions are scored
    as over-segmentation — "same phase split into multiple classes".

    Near-duplicate MDPs whose pixels coexist in the same region are treated
    as numerical noise and NOT penalized.

    Returns: 1 - (# bad duplicate pairs / # total pairs)
    """
    n = cos_sim.shape[0]
    if n < 2:
        return 1.0
    n_pairs = n * (n - 1) // 2
    H, W = class_map.shape
    diag = float(np.hypot(H, W))

    centroids = {}
    for cid in class_ids:
        ys, xs = np.where(class_map == cid)
        centroids[cid] = (float(ys.mean()), float(xs.mean())) if len(ys) else None

    bad = 0
    for ii in range(n):
        for jj in range(ii + 1, n):
            if cos_sim[ii, jj] <= sim_thresh:
                continue
            ca, cb = class_ids[ii], class_ids[jj]
            cta, ctb = centroids[ca], centroids[cb]
            if cta is None or ctb is None:
                continue
            d = np.hypot(cta[0] - ctb[0], cta[1] - ctb[1]) / max(diag, 1e-9)
            if d > centroid_dist_frac:
                bad += 1
    return float(1.0 - bad / n_pairs)


def _score_haadf_nmi(class_map: np.ndarray, haadf: np.ndarray,
                     n_bins: int = 5) -> float:
    """Normalized mutual information between class_map and quantile-binned HAADF."""
    if haadf is None:
        return 0.5  # neutral when unavailable
    flat_cls = class_map.ravel()
    # Quantile binning of HAADF into n_bins, ignoring NaN/inf
    h = haadf.ravel().astype(np.float64)
    finite = np.isfinite(h)
    if not finite.all():
        h = np.where(finite, h, np.nanmedian(h[finite]) if finite.any() else 0.0)
    edges = np.quantile(h, np.linspace(0, 1, n_bins + 1))
    edges[0]  -= 1e-9
    edges[-1] += 1e-9
    h_bin = np.digitize(h, edges[1:-1])
    return float(normalized_mutual_info_score(flat_cls, h_bin))


def _score_grain_compactness(class_map: np.ndarray) -> float:
    """Average over classes of (largest-CC-size / total-class-size).
    High = each class is one coherent region. Low = class is scattered."""
    classes = np.unique(class_map)
    if len(classes) == 0:
        return 1.0
    fracs = []
    for c in classes:
        mask = (class_map == c)
        total = int(mask.sum())
        if total == 0:
            continue
        lbl, _ = nd_label(mask)
        if lbl.max() == 0:
            continue
        _, counts = np.unique(lbl[lbl > 0], return_counts=True)
        fracs.append(int(counts.max()) / total)
    return float(np.mean(fracs)) if fracs else 1.0


def _score_no_speckle(class_map: np.ndarray) -> float:
    """1 - (count of 1-pixel connected components / total pixels).
    Penalizes isolated speckles."""
    total_pixels = class_map.size
    ones = 0
    for c in np.unique(class_map):
        mask = (class_map == c)
        lbl, nlab = nd_label(mask)
        if nlab == 0:
            continue
        _, counts = np.unique(lbl[lbl > 0], return_counts=True)
        ones += int((counts == 1).sum())
    return float(np.clip(1.0 - ones / max(total_pixels, 1), 0.0, 1.0))


def _per_pix_layer(class_map_shape, bounds):
    """Build a (Ny, Nx) array of layer indices (0/1/2)."""
    b1, b2 = bounds
    Ny, Nx = class_map_shape
    layer_ids = np.concatenate([
        np.full(b1, 0), np.full(b2 - b1, 1), np.full(Ny - b2, 2)
    ])
    return np.repeat(layer_ids[:, None], Nx, axis=1)


def _layer_dominant(class_map: np.ndarray, bounds: list[int]) -> dict:
    """For each class, return (dominant_layer_idx, dominant_fraction)."""
    pix_layer = _per_pix_layer(class_map.shape, bounds)
    out = {}
    for c in np.unique(class_map):
        mask = (class_map == c)
        total = int(mask.sum())
        if total == 0:
            continue
        counts = np.bincount(pix_layer[mask], minlength=3)
        dom = int(np.argmax(counts))
        out[int(c)] = (dom, int(counts[dom]) / total)
    return out


def _score_layer_purity(class_map: np.ndarray, bounds: list[int]) -> float:
    """Score = MIN over classes of (dominant-layer fraction).

    One leaky class drags the score down — this is the behaviour needed to
    catch single-class layer leakage that mean-over-classes averaging hides.
    """
    dom = _layer_dominant(class_map, bounds)
    fracs = [frac for _, frac in dom.values()]
    return float(min(fracs)) if fracs else 1.0


def _score_intra_layer_distinct(class_means: dict, class_map: np.ndarray,
                                 bounds: list[int], sim_thresh: float = 0.90
                                 ) -> float:
    """Per-layer over-segmentation.

    Classes are grouped by their dominant layer. Within each layer, count
    pairs of classes with cos-sim > sim_thresh (on flattened MDPs). These
    are "same-layer duplicates" — the geometric over-segmentation detector
    can't catch them because centroids within one layer are always close.

    Score = 1 − (bad within-layer pairs / total within-layer pairs).
    """
    dom = _layer_dominant(class_map, bounds)
    # classes per layer
    groups = {0: [], 1: [], 2: []}
    for c, (lyr, _) in dom.items():
        groups[lyr].append(c)

    bad = 0
    total = 0
    for lyr, cls in groups.items():
        if len(cls) < 2:
            continue
        vecs = np.stack([class_means[c].ravel().astype(np.float64) for c in cls])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(1e-12)
        vecs_n = vecs / norms
        cs = vecs_n @ vecs_n.T
        for ii in range(len(cls)):
            for jj in range(ii + 1, len(cls)):
                total += 1
                if cs[ii, jj] > sim_thresh:
                    bad += 1
    if total == 0:
        return 1.0  # e.g., 1 class per layer → no over-seg possible
    return float(1.0 - bad / total)


def _score_class_purity(class_means: dict, class_indices: dict,
                        dataset, sample_cap: int = 200) -> float:
    """Intra-class MDP spread — under-segmentation detector.

    For each class, sample up to `sample_cap` member patterns, cosine-similarity
    each against the class MDP, and take the median. Average across classes.

    Low score = at least one class has internally heterogeneous members and
    is probably under-segmented (e.g., the "middle layer with no domains"
    case). This uses raw diffraction patterns only — no HAADF.
    """
    if not class_means:
        return 1.0
    per_class_med = []
    for c, idxs in class_indices.items():
        mdp = class_means[c].ravel().astype(np.float64)
        mdp_n = mdp / (np.linalg.norm(mdp) + 1e-12)
        if len(idxs) > sample_cap:
            rng = np.random.default_rng(42)
            sub = rng.choice(idxs, size=sample_cap, replace=False)
        else:
            sub = np.asarray(idxs)
        sims = np.empty(len(sub), dtype=np.float64)
        for k, i in enumerate(sub):
            pat = dataset.get_raw(int(i)).astype(np.float64).ravel()
            pat_n = pat / (np.linalg.norm(pat) + 1e-12)
            sims[k] = float(np.dot(mdp_n, pat_n))
        per_class_med.append(float(np.median(sims)))
    # Return the MIN across classes — a single under-segmented class drags
    # the score down, matching the design of min-over-classes layer_purity.
    return float(min(per_class_med))


def _score_no_cross_layer_mix(class_map: np.ndarray, bounds: list[int],
                              min_frac: float = 0.05) -> float:
    """1 - (fraction of classes that appear in both top AND bottom layer above min_frac)."""
    b1, b2 = bounds
    top = class_map[:b1, :]
    bot = class_map[b2:, :]
    classes = np.unique(class_map)
    if len(classes) == 0:
        return 1.0
    offenders = 0
    for c in classes:
        top_frac = (top == c).mean() if top.size else 0.0
        bot_frac = (bot == c).mean() if bot.size else 0.0
        if top_frac > min_frac and bot_frac > min_frac:
            offenders += 1
    return float(1.0 - offenders / len(classes))


# ─────────────────────────────────────────────────────────────────────────────
# Core scoring: takes pure arrays, returns a result dict.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScorecardResult:
    sample: str
    label: str
    sample_type: str
    components: dict
    weights: dict
    overall: float
    verdict: str
    critical_failures: list
    weakest_component: Optional[str]
    retune_hint: Optional[str]
    ranked_components: list  # [(name, score), ...] low -> high
    ckpt_metrics_used: dict

    def to_json(self):
        return json.dumps(asdict(self), indent=2, default=float)


RETUNE_HINTS = {
    "coherence_score":       "Lower Tfin (assignments too soft → salt-and-pepper).",
    "effk_score":            "Lower num_prototypes or lower Tfin (over-clustered).",
    "mdp_distinctness":      "Lower num_prototypes (duplicate MDPs in disjoint regions).",
    "conf_score":            "Extend training (model under-confident).",
    "grain_compactness":     "Lower num_prototypes (same phase split across regions).",
    "no_speckle":            "Lower Tfin (isolated class flips).",
    "layer_purity":          "Lower Tfin + raise center_mask_radius (a class leaks into the wrong layer; force focus on Bragg spots).",
    "no_cross_layer_mix":    "Raise vmax and lower Tfin (class spans upper+lower layer).",
    "intra_layer_distinct":  "Lower num_prototypes (duplicate classes within one layer).",
    "class_purity":          "Raise num_prototypes or lower vmax (a class has heterogeneous members — under-segmented; give it room to split).",
}


# ─────────────────────────────────────────────────────────────────────────────
# Phase C: scorecard-driven T_fin controller
# ─────────────────────────────────────────────────────────────────────────────

def scorecard_t_adjustment(components: dict, ckpt_metrics: dict,
                            current_tfin: float, num_prototypes: int,
                            min_tfin: float = 0.02, max_tfin: float = 0.15,
                            step_cool: float = 0.85,
                            step_warm: float = 1.15) -> tuple:
    """Scorecard-driven T_fin adjustment rule (replaces the EffK-only
    hard-threshold rule in dino_sr_adaptive.py).

    Only ONE rule fires per call to avoid oscillation. Priority order:
      1. EffK extreme (collapse near 1 → warm; saturation near K → cool)
      2. Under-segmentation (class_purity low) → warm
      3. Over-segmentation (mdp_distinctness low with healthy EffK) → cool
      4. Fragmented class map (low coherence + confident) → cool

    Returns: (new_tfin, rule_name_or_None, reason_string).

    Parameters
    ----------
    components : dict
        ScorecardResult.components — normalized [0, 1] scores.
    ckpt_metrics : dict
        Raw checkpoint metrics incl. `eff_k` (needed to distinguish
        collapse from saturation since effk_score is 0 in both cases).
    current_tfin : float
        Current model.Tfin (the final teacher temperature).
    """
    eff_k = float(ckpt_metrics.get("eff_k", num_prototypes / 2.0))
    coh   = float(components.get("coherence_score", 1.0))
    sep   = float(components.get("mdp_distinctness", 1.0))
    conf  = float(components.get("conf_score", 1.0))
    pur   = float(components.get("class_purity", 1.0))

    def clamp(x):
        return max(min_tfin, min(max_tfin, float(x)))

    # Priority 1: EffK extreme. Use raw eff_k to tell collapse from saturation.
    if eff_k <= 1.5:
        return (clamp(current_tfin * step_warm), "effk_collapse",
                f"effK={eff_k:.2f} (collapsed) -> warm T to spread prototypes")
    if eff_k >= num_prototypes - 0.5:
        return (clamp(current_tfin * step_cool), "effk_saturated",
                f"effK={eff_k:.2f} (saturated at K={num_prototypes}) -> cool T to consolidate")

    # Priority 2: under-segmentation (class has internally heterogeneous members)
    if pur < 0.5:
        return (clamp(current_tfin * step_warm), "low_purity",
                f"class_purity={pur:.2f} (under-seg) -> warm T to allow splitting")

    # Priority 3: over-segmentation (near-duplicate classes in disjoint regions).
    # Threshold 0.95: mdp_distinctness saturates at 1.0 when all cluster-mean
    # pairs are well-separated; anything below 0.95 means at least one pair
    # is cos-sim > 0.95 across disjoint regions — textbook over-split signal
    # (e.g., background chopped into two near-identical prototypes).
    if sep < 0.95 and eff_k > 2.0:
        return (clamp(current_tfin * step_cool), "low_separability",
                f"mdp_distinctness={sep:.2f} (near-duplicate classes) -> cool T to merge")

    # Priority 4: fragmented class map while confident
    if coh < 0.5 and conf > 0.5:
        return (clamp(current_tfin * step_cool), "fragmented",
                f"coherence={coh:.2f} but conf={conf:.2f} -> cool T to sharpen")

    return current_tfin, None, "no rule fired — scorecard already balanced"


def compute_scorecard_from_arrays(
    *,
    sample: str,
    label: str,
    sample_type: str,
    class_map: np.ndarray,             # shape (Ny, Nx)
    class_means: dict,                 # {class_id: (H, W) float32}
    ckpt_metrics: dict,                # from load_ablation_checkpoint
    num_prototypes: int,
    class_indices: Optional[dict] = None,   # {class_id: [flat_idx, ...]} for class_purity
    dataset=None,                            # LoadPRZ-like; for class_purity
    virtual_haadf: Optional[np.ndarray] = None,   # kept in signature for callers
    layer_bounds: Optional[list[int]] = None,
    settings: Optional[dict] = None,
) -> ScorecardResult:
    """Compute the full scorecard from in-memory arrays.

    This is the single source of truth for the scoring logic; both the CLI
    and the in-eval_all integration call it.
    """
    S = settings or load_settings()

    # ── Common components ───────────────────────────────────────────────────
    avg_conf = float(ckpt_metrics.get("avg_conf", 0.0))
    eff_k    = float(ckpt_metrics.get("eff_k",    0.0))

    conf_score      = float(np.clip(avg_conf, 0.0, 1.0))
    effk_score      = _score_effk(eff_k, num_prototypes)
    coherence_score = _score_coherence(class_map)

    # Cosine sim matrix from class MDPs (L2-normalised, flattened)
    cids = sorted(class_means.keys())
    if len(cids) >= 2:
        vecs = np.stack([class_means[c].ravel().astype(np.float64) for c in cids])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(1e-12)
        vecs_n = vecs / norms
        cos_sim = vecs_n @ vecs_n.T
    else:
        cos_sim = np.array([[1.0]])
    mdp_distinctness = _score_mdp_distinctness(cos_sim, class_map, cids)

    # class_purity (under-segmentation detector) — needs dataset + indices.
    # Silently skip (score=1.0) when caller didn't supply them so the older
    # call-sites keep working; the callback plumbs these in.
    if dataset is not None and class_indices is not None:
        class_purity_val = _score_class_purity(class_means, class_indices, dataset)
    else:
        class_purity_val = 1.0

    components = {
        "conf_score":       conf_score,
        "effk_score":       effk_score,
        "coherence_score":  coherence_score,
        "mdp_distinctness": mdp_distinctness,
        "class_purity":     class_purity_val,
    }
    weights = {
        k: v for k, v in S["weights_common"].items() if k in components
    }

    # ── Sample-type-specific components ─────────────────────────────────────
    if sample_type == "layered":
        if layer_bounds is None:
            raise ValueError(f"Layered sample {sample} requires layer_bounds.")
        # v2: layer_purity is now MIN over classes (a single leaky class
        # drags the score down, catching single-class layer leakage).
        components["layer_purity"]          = _score_layer_purity(class_map, layer_bounds)
        components["no_cross_layer_mix"]    = _score_no_cross_layer_mix(class_map, layer_bounds)
        # Per-layer over-segmentation: duplicate MDPs within a single layer.
        components["intra_layer_distinct"]  = _score_intra_layer_distinct(
            class_means, class_map, layer_bounds)
        weights.update(S["weights_layered"])
    else:
        components["grain_compactness"] = _score_grain_compactness(class_map)
        components["no_speckle"]        = _score_no_speckle(class_map)
        weights.update(S["weights_non_layered"])

    # ── Weighted overall ────────────────────────────────────────────────────
    wsum = sum(weights.values())
    overall = sum(weights[c] * components[c] for c in components) / max(wsum, 1e-9)

    # ── Critical-failure gates ──────────────────────────────────────────────
    critical = []
    if eff_k >= num_prototypes - S["effk_collapse_margin"]:
        critical.append(f"effK={eff_k:.2f} at num_prototypes={num_prototypes} (over-clustered)")
    if coherence_score < S["min_coherence"]:
        critical.append(f"coherence_score={coherence_score:.2f} < {S['min_coherence']}")
    if conf_score < S["min_conf"]:
        critical.append(f"conf_score={conf_score:.2f} < {S['min_conf']}")
    if components["class_purity"] < S.get("min_class_purity", 0.0):
        critical.append(f"class_purity={components['class_purity']:.2f} < {S['min_class_purity']} (under-segmentation)")
    if sample_type == "layered":
        if components["layer_purity"] < S["min_layer_purity"]:
            critical.append(f"layer_purity={components['layer_purity']:.2f} < {S['min_layer_purity']} (class leaks layer)")
        if components["no_cross_layer_mix"] < S["min_no_cross_mix"]:
            critical.append(f"no_cross_layer_mix={components['no_cross_layer_mix']:.2f} < {S['min_no_cross_mix']}")
        if components["intra_layer_distinct"] < S.get("min_intra_layer_dist", 0.0):
            critical.append(f"intra_layer_distinct={components['intra_layer_distinct']:.2f} < {S['min_intra_layer_dist']} (duplicate classes within a layer)")

    # ── Verdict ─────────────────────────────────────────────────────────────
    if critical:
        verdict = "REJECT"
    elif overall >= S["threshold_approve"]:
        verdict = "APPROVE"
    elif overall >= S["threshold_retune"]:
        verdict = "RETUNE"
    else:
        verdict = "REJECT"

    # ── Retune hint based on weakest component ──────────────────────────────
    ranked = sorted(components.items(), key=lambda kv: kv[1])
    weakest = ranked[0][0] if ranked else None
    retune_hint = RETUNE_HINTS.get(weakest) if (verdict != "APPROVE" and weakest) else None

    return ScorecardResult(
        sample=sample, label=label, sample_type=sample_type,
        components=components,
        weights=weights,
        overall=float(overall),
        verdict=verdict,
        critical_failures=critical,
        weakest_component=weakest,
        retune_hint=retune_hint,
        ranked_components=[(k, float(v)) for k, v in ranked],
        ckpt_metrics_used={"avg_conf": avg_conf, "eff_k": eff_k,
                           "num_prototypes": num_prototypes},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard PNG
# ─────────────────────────────────────────────────────────────────────────────

def save_scorecard_png(result: ScorecardResult, out_path: str,
                       thresh_approve: float = 0.80,
                       thresh_retune: float = 0.60) -> None:
    names  = [k for k, _ in result.ranked_components]
    scores = [v for _, v in result.ranked_components]
    colors = ["#c0392b" if s < thresh_retune
              else "#e67e22" if s < thresh_approve
              else "#27ae60" for s in scores]

    fig, ax = plt.subplots(figsize=(9, 0.35 * len(names) + 2.5))
    y = np.arange(len(names))
    ax.barh(y, scores, color=colors, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.axvline(thresh_retune, color="gray", linestyle=":", linewidth=1)
    ax.axvline(thresh_approve, color="gray", linestyle="--", linewidth=1)
    ax.text(thresh_retune, -0.8, "retune", fontsize=7, color="gray",
            ha="center", va="top")
    ax.text(thresh_approve, -0.8, "approve", fontsize=7, color="gray",
            ha="center", va="top")

    for yi, s in zip(y, scores):
        ax.text(s + 0.01, yi, f"{s:.2f}", va="center", fontsize=8)

    verdict_color = {"APPROVE": "#27ae60", "RETUNE": "#e67e22",
                     "REJECT": "#c0392b"}.get(result.verdict, "gray")
    title = f"{result.sample}/{result.label}  [{result.sample_type}]"
    subtitle = f"overall={result.overall:.3f}   verdict={result.verdict}"
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    fig.text(0.99, 0.99, result.verdict, fontsize=16, fontweight="bold",
             color=verdict_color, ha="right", va="top")
    fig.text(0.01, 0.01, subtitle, fontsize=9, ha="left", va="bottom")

    if result.critical_failures:
        fig.text(0.01, 0.06,
                 "Critical: " + " | ".join(result.critical_failures),
                 fontsize=7, color="#c0392b", ha="left", va="bottom")
    if result.retune_hint:
        fig.text(0.99, 0.01, f"Hint: {result.retune_hint}",
                 fontsize=8, ha="right", va="bottom", style="italic")

    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone path: load checkpoint, run inference, compute, save.
# ─────────────────────────────────────────────────────────────────────────────

def _load_sample_config(sample: str) -> dict:
    """Pull sample paths/scan/vmax/etc. from eval_all.py:SAMPLES."""
    # Import locally so scorecard.py remains usable without torch when only
    # the scoring math is needed.
    from eval_all import SAMPLES as EVAL_SAMPLES
    if sample not in EVAL_SAMPLES:
        raise KeyError(f"{sample} not in eval_all.SAMPLES")
    cfg = dict(EVAL_SAMPLES[sample])
    cfg.update(SAMPLE_TYPES.get(sample, {"sample_type": "non_layered"}))
    return cfg


def _run_inference_and_components(sample: str, label: str, cfg: dict,
                                  device: str = "cuda",
                                  ckpt_path: Optional[str] = None):
    """Load checkpoint, run inference, compute class_means + virtual_haadf.

    Reads training-time preprocessing context (`vmax`, `center_mask_radius`,
    `center_crop_size`) from `ckpt['train_config']` when present so the
    dataset + eval transform match exactly what the model was trained on.
    Falls back to `cfg` (eval_all.SAMPLES) with a loud warning for older
    checkpoints that don't carry the context.

    Returns (class_map, class_means, class_indices, ckpt_metrics,
             num_prototypes, dataset).
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2 as T

    from eval_all import (LoadPRZ, compute_virtual_images, compute_class_means,
                          run_inference, build_eval_transform)
    from dino_sr_ablation import load_ablation_checkpoint

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    if ckpt_path is None:
        ckpt_dir = os.path.join(BASE_OUTPUT_DIR, sample, "L1", label)
        ckpt_path = os.path.join(ckpt_dir, "best_auto.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ── Peek at the checkpoint to get training-time preprocessing ──
    print(f"  Peeking at checkpoint metadata ...", flush=True)
    _ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_cfg = _ck.get("train_config") or {}
    K = int(_ck.get("num_prototypes", cfg.get("num_prototypes", 10)))

    if train_cfg:
        vmax = float(train_cfg.get("vmax", cfg["vmax"]))
        mask_r = int(train_cfg.get("center_mask_radius",
                                    cfg.get("center_mask_radius", 15)))
        crop_size = int(train_cfg.get("center_crop_size", 135))
        resize = int(train_cfg.get("resize", 192))
        print(f"  Using training-time preprocessing from ckpt: "
              f"vmax={vmax}, mask_r={mask_r}, crop={crop_size}")
    else:
        vmax = float(cfg["vmax"])
        mask_r = int(cfg.get("center_mask_radius", 15))
        crop_size = 135
        resize = 192
        print(f"  [WARN] ckpt has no train_config; falling back to SAMPLES: "
              f"vmax={vmax}, mask_r={mask_r}. Only valid if these match "
              f"training. For overridden runs, re-train to persist train_config.")
    del _ck

    print(f"  Loading dataset {sample} ...", flush=True)
    dataset = LoadPRZ(cfg["path"], resize=resize, vmax=vmax)
    scan_shape = cfg["scan_shape"]

    # Inference
    print(f"  Loading checkpoint ...", flush=True)
    model, eval_temp, ckpt_metrics = load_ablation_checkpoint(ckpt_path, device=dev)
    model.eval()
    eval_tf = build_eval_transform(mask_r, crop_size)
    eval_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

    print(f"  Running inference on {len(dataset)} patterns ...", flush=True)
    assigns, eff_k, avg_conf, n_active, _all_probs, _remap = run_inference(
        model, eval_loader, eval_tf, eval_temp, dev
    )
    ckpt_metrics = dict(ckpt_metrics or {})
    # Overwrite the inference-derived values so we aren't at the mercy of
    # whatever was snapshotted at checkpoint time.
    ckpt_metrics["eff_k"]    = float(eff_k)
    ckpt_metrics["avg_conf"] = float(avg_conf)

    class_map = assigns.reshape(scan_shape)

    print(f"  Computing class means ({n_active} classes) ...", flush=True)
    class_means, class_indices = compute_class_means(assigns, dataset)

    # Free GPU/dataset before scoring (keep dataset; class_purity needs it)
    del model, eval_loader
    try:
        import torch as _t
        _t.cuda.empty_cache()
    except Exception:
        pass

    return class_map, class_means, class_indices, ckpt_metrics, K, dataset


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: in-training evaluation callback.
#
# Usage pattern:
#   from scorecard import make_scorecard_callback
#   cb = make_scorecard_callback(dataset, scan_shape, sample_type, ...)
#   train_ablation(model, dataset, ..., eval_callback=cb, eval_every=10)
#
# The callback caches virtual HAADF (sample-level, model-independent) so
# repeated invocations are cheap. Each call runs inference, computes class
# means, builds a ScorecardResult, and returns it.
# ─────────────────────────────────────────────────────────────────────────────

def make_scorecard_callback(dataset, scan_shape, sample_type, *,
                             layer_bounds=None, num_prototypes=10,
                             center_mask_radius=15, center_crop_size=135,
                             batch_size=128, device=None,
                             sample_name="run", run_label="run",
                             settings=None, use_polar: bool = False,
                             ellipse_affine=None,
                             polar_mask_cols: int = 30):
    """Return a closure `cb(model, epoch, teacher_temp) -> ScorecardResult`.

    Caches the virtual HAADF once (it doesn't depend on model weights) so
    mid-training eval overhead per call is just: 1 inference pass + 1 class-
    means computation ≈ 15–30 s depending on sample size.

    use_polar: if True, append a PolarTransform to the eval transform so the
    inference path matches what the model saw during training.
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2 as T
    from eval_all import (compute_virtual_images, compute_class_means,
                           run_inference, build_eval_transform)

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  [scorecard] precomputing virtual HAADF for {sample_name} ...",
          flush=True)
    haadf, _bf, _com, _cx, _cy = compute_virtual_images(
        dataset, scan_shape, center_mask_radius
    )

    eval_tf = build_eval_transform(center_mask_radius, center_crop_size)
    # Eval pipeline must match training exactly. When polar is on, we also
    # skip the Cartesian CenterMask (which build_eval_transform inherits
    # from the training aug) and instead mask post-polar.
    eval_ops = list(eval_tf.transforms)
    if ellipse_affine is not None:
        from elliptical_correction import EllipticalCorrection
        eval_ops = [EllipticalCorrection(ellipse_affine)] + eval_ops
    if use_polar:
        # Drop any Cartesian CenterMask from build_eval_transform — we mask
        # post-polar instead.
        from dino_sr_fixed import CenterMask
        from dino_sr_ablation import PolarTransform, PolarMaskLeft
        eval_ops = [op for op in eval_ops if not isinstance(op, CenterMask)]
        eval_ops = eval_ops + [PolarTransform(output_size=192)]
        if polar_mask_cols > 0:
            eval_ops = eval_ops + [PolarMaskLeft(k_cols=polar_mask_cols)]
    eval_tf = T.Compose(eval_ops)
    eval_loader = DataLoader(dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)

    def cb(model, epoch, teacher_temp):
        was_training = model.training
        model.eval()
        try:
            assigns, eff_k, avg_conf, _n_active, _ap, _remap = run_inference(
                model, eval_loader, eval_tf, teacher_temp, dev
            )
            class_map = assigns.reshape(scan_shape)
            class_means, class_indices = compute_class_means(assigns, dataset)
            ckpt_metrics = {"avg_conf": float(avg_conf), "eff_k": float(eff_k)}
            result = compute_scorecard_from_arrays(
                sample=sample_name, label=run_label, sample_type=sample_type,
                class_map=class_map, class_means=class_means,
                class_indices=class_indices, dataset=dataset,
                ckpt_metrics=ckpt_metrics, num_prototypes=num_prototypes,
                virtual_haadf=haadf, layer_bounds=layer_bounds,
                settings=settings,
            )
            return result
        finally:
            if was_training:
                model.train()

    return cb


def score_run(sample: str, label: Optional[str] = None,
              device: str = "cuda", save_outputs: bool = True,
              ckpt_path: Optional[str] = None
              ) -> ScorecardResult:
    cfg = _load_sample_config(sample)
    if label is None:
        label = cfg.get("approved_label")
        if label is None:
            raise ValueError(f"No label given and no approved_label for {sample}")

    class_map, class_means, class_indices, ckpt_metrics, K, dataset = \
        _run_inference_and_components(sample, label, cfg, device=device,
                                       ckpt_path=ckpt_path)

    result = compute_scorecard_from_arrays(
        sample=sample, label=label,
        sample_type=cfg["sample_type"],
        class_map=class_map,
        class_means=class_means,
        class_indices=class_indices,
        dataset=dataset,
        ckpt_metrics=ckpt_metrics,
        num_prototypes=K,
        layer_bounds=cfg.get("layer_bounds"),
    )

    if save_outputs:
        eval_dir = os.path.join(BASE_OUTPUT_DIR, sample, "L1", label, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        json_path = os.path.join(eval_dir, "scorecard.json")
        png_path  = os.path.join(eval_dir, "scorecard.png")
        with open(json_path, "w") as f:
            f.write(result.to_json())
        S = load_settings()
        save_scorecard_png(result, png_path,
                           thresh_approve=S["threshold_approve"],
                           thresh_retune=S["threshold_retune"])
        print(f"  Wrote {json_path}\n  Wrote {png_path}")

    _print_result_line(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Ranking + calibration
# ─────────────────────────────────────────────────────────────────────────────

def _print_result_line(r: ScorecardResult) -> None:
    print(f"  [{r.verdict:<7}] {r.sample}/{r.label:<18} overall={r.overall:.3f}"
          f"  weakest={r.weakest_component}"
          + (f"  ({len(r.critical_failures)} critical)" if r.critical_failures else ""))


def rank_all(device: str = "cuda") -> list[ScorecardResult]:
    """Score all approved runs, write per-run outputs, return sorted list."""
    results = []
    from eval_all import SAMPLES as EVAL_SAMPLES
    for sample, cfg in EVAL_SAMPLES.items():
        label = cfg.get("approved_label")
        if label is None:
            continue
        try:
            print(f"\n── {sample}/{label} ──")
            results.append(score_run(sample, label, device=device))
        except Exception as exc:
            print(f"  FAILED: {exc}")
            traceback.print_exc()

    results.sort(key=lambda r: r.overall, reverse=True)
    _write_ranking_csv(results)
    print("\n──────── ranked ────────")
    for r in results:
        _print_result_line(r)
    return results


def _write_ranking_csv(results: list[ScorecardResult]) -> None:
    if not results:
        return
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "scorecard_ranking.csv")
    import csv
    all_keys = []
    for r in results:
        for k in r.components:
            if k not in all_keys:
                all_keys.append(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "label", "sample_type", "overall", "verdict"] + all_keys)
        for r in results:
            w.writerow([r.sample, r.label, r.sample_type,
                        round(r.overall, 4), r.verdict]
                       + [round(r.components.get(k, float("nan")), 4) for k in all_keys])
    print(f"\nWrote ranking → {csv_path}")


def calibrate(device: str = "cuda") -> None:
    """Compute scorecards for all calibration runs, then grid-search weights
    and thresholds for best agreement. Persist to scorecard_weights.json."""
    # Step 1: compute raw component vectors for all calibration runs ONCE.
    raw = {}
    for (sample, label) in CALIB_APPROVED + CALIB_REJECTED:
        cfg = _load_sample_config(sample)
        try:
            class_map, class_means, class_indices, ckpt_metrics, K, dataset = \
                _run_inference_and_components(sample, label, cfg, device=device)
        except FileNotFoundError as e:
            print(f"  Skipping {sample}/{label}: {e}")
            continue
        print(f"  Scored {sample}/{label}")
        raw[(sample, label)] = dict(
            cfg=cfg, class_map=class_map, class_means=class_means,
            class_indices=class_indices, dataset=dataset,
            ckpt_metrics=ckpt_metrics, K=K
        )

    approved_keys = [k for k in raw if k in CALIB_APPROVED]
    rejected_keys = [k for k in raw if k in CALIB_REJECTED]
    print(f"\nCalibration set: {len(approved_keys)} approved + "
          f"{len(rejected_keys)} rejected")

    # Step 2: compute component scores once per run using DEFAULT weights — we
    # only need the component values (weights are applied post-hoc).
    per_run_components = {}
    for key, data in raw.items():
        sample, label = key
        r = compute_scorecard_from_arrays(
            sample=sample, label=label,
            sample_type=data["cfg"]["sample_type"],
            class_map=data["class_map"],
            class_means=data["class_means"],
            class_indices=data["class_indices"],
            dataset=data["dataset"],
            ckpt_metrics=data["ckpt_metrics"],
            num_prototypes=data["K"],
            layer_bounds=data["cfg"].get("layer_bounds"),
            settings=DEFAULTS,
        )
        per_run_components[key] = (r, data["cfg"]["sample_type"])

    # Step 3: grid-search over weights (step 0.1) on the common simplex and
    # one overall threshold (step 0.05). Keep type-specific weights equal for
    # simplicity — the gate already separates layered vs non_layered logic.
    best = None
    common_names = list(DEFAULTS["weights_common"].keys())  # 5 names
    # 5-simplex at 0.2 granularity → ~126 combos. With threshold grid 0.5–0.9
    # at 0.05 step (9 values) → ~1134 total. Fast enough.
    STEP = 0.2
    choices = [i * STEP for i in range(int(1.0 / STEP) + 1)]

    def simplex5(step):
        acc = []
        for a in choices:
            for b in choices:
                for c in choices:
                    for d in choices:
                        e = 1.0 - (a + b + c + d)
                        if -1e-9 <= e <= 1.0 + 1e-9:
                            acc.append((a, b, c, d, max(0.0, e)))
        return acc

    combos = simplex5(STEP)
    thresh_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    for w in combos:
        weights_common = dict(zip(common_names, w))
        # Type-specific weights: each one equals the mean of common weights
        avg = sum(w) / len(w)
        settings = dict(DEFAULTS)
        settings["weights_common"] = weights_common
        settings["weights_non_layered"] = {k: avg for k in DEFAULTS["weights_non_layered"]}
        settings["weights_layered"]     = {k: avg for k in DEFAULTS["weights_layered"]}

        for tA in thresh_grid:
            for tR in [t for t in thresh_grid if t < tA]:
                settings["threshold_approve"] = tA
                settings["threshold_retune"]  = tR

                correct = 0
                for key, (_, stype) in per_run_components.items():
                    data = raw[key]
                    r = compute_scorecard_from_arrays(
                        sample=key[0], label=key[1],
                        sample_type=stype,
                        class_map=data["class_map"],
                        class_means=data["class_means"],
                        ckpt_metrics=data["ckpt_metrics"],
                        num_prototypes=data["K"],
                        virtual_haadf=data["haadf"],
                        layer_bounds=data["cfg"].get("layer_bounds"),
                        settings=settings,
                    )
                    want_approve = key in CALIB_APPROVED
                    got_approve  = (r.verdict == "APPROVE")
                    if want_approve == got_approve:
                        correct += 1

                if best is None or correct > best[0]:
                    best = (correct, settings)

    best_correct, best_settings = best
    total = len(per_run_components)
    print(f"\nBest calibration: {best_correct}/{total} correct")
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(best_settings, f, indent=2)
    print(f"Wrote calibrated weights → {WEIGHTS_PATH}")

    # Final confusion table with persisted weights
    print("\n── Final verdicts with calibrated weights ──")
    S = best_settings
    for key, (_, stype) in per_run_components.items():
        data = raw[key]
        r = compute_scorecard_from_arrays(
            sample=key[0], label=key[1],
            sample_type=stype,
            class_map=data["class_map"],
            class_means=data["class_means"],
            class_indices=data["class_indices"],
            dataset=data["dataset"],
            ckpt_metrics=data["ckpt_metrics"],
            num_prototypes=data["K"],
            layer_bounds=data["cfg"].get("layer_bounds"),
            settings=S,
        )
        want = "APPROVE" if key in CALIB_APPROVED else "not-APPROVE"
        mark = "OK" if (want == "APPROVE") == (r.verdict == "APPROVE") else "XX"
        print(f"  [{mark}] {key[0]}/{key[1]:<18}  want={want:<11}  got={r.verdict}  "
              f"overall={r.overall:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", help="Sample name from eval_all.SAMPLES")
    ap.add_argument("--label",  help="Checkpoint label (default: approved_label)")
    ap.add_argument("--ckpt",   help="Explicit checkpoint path (overrides --label resolution)")
    ap.add_argument("--all",    action="store_true",
                    help="Score all approved runs and print ranking")
    ap.add_argument("--calibrate", action="store_true",
                    help="Tune weights + thresholds against approved/rejected runs")
    ap.add_argument("--device", default="cuda",
                    help="torch device (default: cuda, falls back to cpu)")
    args = ap.parse_args()

    if args.calibrate:
        calibrate(device=args.device)
        return 0
    if args.all:
        rank_all(device=args.device)
        return 0
    if not args.sample:
        ap.error("provide --sample, --all, or --calibrate")
    score_run(args.sample, args.label, device=args.device, ckpt_path=args.ckpt)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
