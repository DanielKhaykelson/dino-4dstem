"""pair_proposers.py — pair-sampling strategies for active-learning
fine-tune labelling.

Each `make_*_proposer(...)` returns a CALLABLE that, when invoked
with no arguments, returns the next `(idx_a, idx_b, source)` triple
or `None` once the internal queue is exhausted.  The
`PairLabelerWindow` consumes this callable in `mode='active'`.

Strategies shipped today:

  cross_class — for each pair of distinct prototypes (i, j), pick
                high-confidence representatives of i and j and ask
                "actually same physical phase?".  Useful for spotting
                clusters that the model split but the user thinks
                should merge.  +1 from the user means "merge these
                two prototypes' meanings"; -1 confirms the split.

  scan_edge   — pairs of NEIGHBOURING (4-connected) scan pixels
                assigned to DIFFERENT prototypes.  Useful for
                refining real-space phase boundaries.  +1 says
                "this boundary doesn't make physical sense"; -1
                confirms it.

Both shuffle their internal queues so the user sees the full diversity
before exhausting any single pair-class combination.

API contract for the proposer callable:
  call() -> (idx_a, idx_b, source) | None
  attribute `.total : int`        — total queue size (for progress)
  attribute `.position : int`     — current position in the queue
  attribute `.remaining : int`    — total - position
"""
from __future__ import annotations
import numpy as np


def _wrap(queue, label="proposer"):
    """Wrap a list of (a, b, source) tuples into a stateful callable
    with .total/.position/.remaining counters."""
    state = {"i": 0}

    def _next():
        if state["i"] >= len(queue):
            return None
        item = queue[state["i"]]
        state["i"] += 1
        return item

    def _get_total(): return len(queue)
    def _get_pos():   return state["i"]
    def _get_rem():   return max(0, len(queue) - state["i"])
    # Use a class to attach attributes that update live
    class _Prop:
        @property
        def total(self): return len(queue)
        @property
        def position(self): return state["i"]
        @property
        def remaining(self): return max(0, len(queue) - state["i"])
        @property
        def label(self): return label
        def __call__(self): return _next()

    return _Prop()


def cross_class_proposer(soft_probs, K, *,
                            n_per_pair=4, max_pairs=200,
                            pool_size=50, seed=0,
                            restrict_pair=None,
                            randomize_order=True):
    """Build a queue of cross-class pairs.

    Parameters
    ----------
    soft_probs : (N, K) array
        Per-pattern soft assignment to each prototype.
    K : int
        Number of prototypes.
    n_per_pair : int
        How many random (i-class, j-class) pairs to draw per ordered
        prototype pair (i, j) with i < j.  Total pairs ≈ K*(K-1)/2 *
        n_per_pair (without `restrict_pair`).
    max_pairs : int
        Hard cap on total queue length.  Longer queues = more user
        labelling but more signal.
    pool_size : int
        For each prototype c, treat the top `pool_size` patterns by
        confidence as "high-confidence reps of c".  Keeps each pair a
        meaningful "really class i" vs "really class j" comparison.
    seed : int
        RNG seed for reproducibility.
    restrict_pair : (int, int) | None
        If set, ONLY surface pairs where one member is from class
        `restrict_pair[0]` and the other from `restrict_pair[1]`.
        Useful when the user already suspects two specific
        prototypes are the same physical phase.  Self-pairs
        (i, i) are accepted and behave like `intra_class_proposer`
        for that class.  Out-of-range ids raise ValueError.
    randomize_order : bool
        If True (default), for each (a, b) randomly swap (b, a) so
        the user doesn't always see class i on the LEFT and class
        j on the RIGHT.  Removes a position-bias the user might
        otherwise pick up on subconsciously.

    Returns
    -------
    callable    proposer with .total / .position / .remaining
    """
    soft = np.asarray(soft_probs)
    if soft.ndim != 2 or soft.shape[1] != int(K):
        raise ValueError(
            f"soft_probs must be (N, K); got shape {soft.shape}, K={K}")
    N = soft.shape[0]
    rng = np.random.default_rng(int(seed))

    pool = {}
    for c in range(int(K)):
        order = np.argsort(-soft[:, c])
        n_take = min(int(pool_size), N)
        pool[c] = order[:n_take]

    if restrict_pair is not None:
        ri, rj = int(restrict_pair[0]), int(restrict_pair[1])
        if not (0 <= ri < int(K)) or not (0 <= rj < int(K)):
            raise ValueError(
                f"restrict_pair {restrict_pair} out of range for K={K}")
        # Build only this single (i, j) -- iter once.
        # Take min/max so the source label is canonical.
        i_lo, i_hi = (min(ri, rj), max(ri, rj))
        ij_iter = [(i_lo, i_hi)]
    else:
        ij_iter = [(i, j)
                    for i in range(int(K))
                    for j in range(i + 1, int(K))]

    queue = []
    for (i, j) in ij_iter:
        if pool[i].size == 0 or pool[j].size == 0:
            continue
        # When restricted to a single pair, we want MORE samples than
        # the default n_per_pair (which is per (i,j) and assumes the
        # user is browsing many pairs). Bump to max_pairs so the
        # restrict-pair queue is full-length.
        n_take = (int(max_pairs) if restrict_pair is not None
                  else int(n_per_pair))
        for _ in range(n_take):
            a = int(pool[i][rng.integers(0, pool[i].size)])
            b = int(pool[j][rng.integers(0, pool[j].size)])
            if a == b:  # rare but possible (i == j case)
                continue
            if randomize_order and bool(rng.integers(0, 2)):
                a, b = b, a
            queue.append((a, b, f"cross_class:p{i}-p{j}"))

    rng.shuffle(queue)
    if max_pairs is not None and len(queue) > int(max_pairs):
        queue = queue[: int(max_pairs)]
    return _wrap(queue, label=("cross_class"
                                if restrict_pair is None
                                else f"cross_class:p{ri}-p{rj}"))


def intra_class_proposer(soft_probs, K, *,
                            n_per_class=8, max_pairs=200,
                            pool_size=50, seed=0,
                            restrict_class=None):
    """Build a queue of WITHIN-prototype pairs.

    For each prototype c, pick `n_per_class` random pairs from c's
    high-confidence pool. User answers "actually same physical phase?".

    Use case: if the model put two physically-different phases into
    the same cluster, intra-class labels of "different" tell the
    model to split them. Conversely, "same" labels reinforce a
    correctly-merged cluster.

    See `cross_class_proposer` for parameter semantics; everything
    works the same except pairs are within a class instead of
    across classes.

    Parameters
    ----------
    restrict_class : int | None
        If set, ONLY produce pairs from this prototype.  Useful when
        the user suspects a specific cluster contains two phases.
        Out-of-range raises ValueError.
    """
    soft = np.asarray(soft_probs)
    if soft.ndim != 2 or soft.shape[1] != int(K):
        raise ValueError(
            f"soft_probs must be (N, K); got {soft.shape}, K={K}")
    N = soft.shape[0]
    rng = np.random.default_rng(int(seed))

    pool = {}
    for c in range(int(K)):
        order = np.argsort(-soft[:, c])
        n_take = min(int(pool_size), N)
        pool[c] = order[:n_take]

    if restrict_class is not None:
        rc = int(restrict_class)
        if not (0 <= rc < int(K)):
            raise ValueError(
                f"restrict_class {rc} out of range for K={K}")
        classes = [rc]
    else:
        classes = list(range(int(K)))

    queue = []
    for c in classes:
        if pool[c].size < 2:
            continue
        # When restricted, pull max_pairs from this single class so the
        # queue is full-length.
        n_take = (int(max_pairs) if restrict_class is not None
                  else int(n_per_class))
        for _ in range(n_take):
            a = int(pool[c][rng.integers(0, pool[c].size)])
            b = int(pool[c][rng.integers(0, pool[c].size)])
            if a == b:
                continue
            queue.append((a, b, f"intra_class:p{c}"))
    rng.shuffle(queue)
    if max_pairs is not None and len(queue) > int(max_pairs):
        queue = queue[: int(max_pairs)]
    return _wrap(queue, label=("intra_class"
                                if restrict_class is None
                                else f"intra_class:p{int(restrict_class)}"))


def mixed_intra_inter_proposer(soft_probs, K, *,
                                  intra_per_class=4,
                                  inter_per_pair=2,
                                  max_pairs=200,
                                  pool_size=50, seed=0,
                                  restrict_intra_class=None,
                                  restrict_inter_pair=None,
                                  randomize_order=True):
    """Interleaved intra-class + cross-class pairs.

    Default mode for active labelling — probes both 'are these two
    really the same class?' and 'are these two really different?',
    so a single labelling pass tests both kinds of model error.

    Builds the two queues separately, interleaves them so the user
    sees a balanced mix as they label, then truncates to `max_pairs`.

    Optional filters:
      restrict_intra_class : int | None  — restrict the intra side
        to one prototype.  When set, the intra queue ONLY draws from
        that class.
      restrict_inter_pair  : (int, int) | None  — restrict the inter
        side to one prototype pair.  When set, the inter queue ONLY
        draws across those two prototypes.
      randomize_order      : bool — passed to cross_class_proposer
        so the user doesn't always see (i, j) with i on the LEFT.
    """
    intra = intra_class_proposer(soft_probs, K,
        n_per_class=intra_per_class, max_pairs=10**6,
        pool_size=pool_size, seed=seed,
        restrict_class=restrict_intra_class)
    inter = cross_class_proposer(soft_probs, K,
        n_per_pair=inter_per_pair, max_pairs=10**6,
        pool_size=pool_size, seed=seed + 1,
        restrict_pair=restrict_inter_pair,
        randomize_order=randomize_order)
    # Drain both into lists so we can interleave deterministically.
    intra_list = []
    while True:
        item = intra()
        if item is None: break
        intra_list.append(item)
    inter_list = []
    while True:
        item = inter()
        if item is None: break
        inter_list.append(item)
    rng = np.random.default_rng(int(seed) + 2)
    rng.shuffle(intra_list)
    rng.shuffle(inter_list)
    # Interleave round-robin
    queue = []
    i_intra = i_inter = 0
    while i_intra < len(intra_list) or i_inter < len(inter_list):
        if i_intra < len(intra_list):
            queue.append(intra_list[i_intra]); i_intra += 1
        if i_inter < len(inter_list):
            queue.append(inter_list[i_inter]); i_inter += 1
    if max_pairs is not None and len(queue) > int(max_pairs):
        queue = queue[: int(max_pairs)]
    return _wrap(queue, label="cross+intra")


def scan_edge_proposer(assigns, scan_shape, *,
                          max_pairs=200, seed=0,
                          include_diag=False):
    """Pairs of neighbour scan pixels assigned to DIFFERENT prototypes.

    Parameters
    ----------
    assigns : (N,) int array
        argmax(soft_probs, axis=1) reshaped flat. Equivalent to
        `inference['assigns']` from `infer_scan`.
    scan_shape : (Ny, Nx)
    max_pairs : int
    seed : int
    include_diag : bool
        If True, also include the two diagonal neighbours (so pairs
        across 8-connected boundaries). Default False (4-connected).
    """
    Ny, Nx = int(scan_shape[0]), int(scan_shape[1])
    a2 = np.asarray(assigns).reshape(Ny, Nx).astype(np.int64)
    rng = np.random.default_rng(int(seed))

    pairs = set()
    # Right neighbour
    for ry in range(Ny):
        for rx in range(Nx - 1):
            if a2[ry, rx] != a2[ry, rx + 1]:
                ia = ry * Nx + rx
                ib = ry * Nx + (rx + 1)
                pairs.add((min(ia, ib), max(ia, ib),
                            int(a2[ry, rx]), int(a2[ry, rx + 1])))
    # Down neighbour
    for ry in range(Ny - 1):
        for rx in range(Nx):
            if a2[ry, rx] != a2[ry + 1, rx]:
                ia = ry * Nx + rx
                ib = (ry + 1) * Nx + rx
                pairs.add((min(ia, ib), max(ia, ib),
                            int(a2[ry, rx]), int(a2[ry + 1, rx])))
    if include_diag:
        for ry in range(Ny - 1):
            for rx in range(Nx - 1):
                if a2[ry, rx] != a2[ry + 1, rx + 1]:
                    ia = ry * Nx + rx
                    ib = (ry + 1) * Nx + (rx + 1)
                    pairs.add((min(ia, ib), max(ia, ib),
                                int(a2[ry, rx]), int(a2[ry + 1, rx + 1])))
                if rx + 1 < Nx and a2[ry, rx + 1] != a2[ry + 1, rx]:
                    ia = ry * Nx + (rx + 1)
                    ib = (ry + 1) * Nx + rx
                    pairs.add((min(ia, ib), max(ia, ib),
                                int(a2[ry, rx + 1]), int(a2[ry + 1, rx])))

    pairs_list = list(pairs)
    rng.shuffle(pairs_list)
    if max_pairs is not None and len(pairs_list) > int(max_pairs):
        pairs_list = pairs_list[: int(max_pairs)]
    queue = [(a, b, f"scan_edge:p{ca}-p{cb}")
             for (a, b, ca, cb) in pairs_list]
    return _wrap(queue, label="scan_edge")


def low_margin_proposer(soft_probs, *,
                          max_pairs=200, K_top=2,
                          margin_threshold=0.20, seed=0):
    """Pick patterns where top-1 and top-2 softmax are close (model
    is unsure), and pair each with a high-confidence rep of its
    second-place prototype. User labels resolve which class the
    ambiguous pattern actually belongs to."""
    soft = np.asarray(soft_probs)
    N, K = soft.shape
    rng = np.random.default_rng(int(seed))

    # margin = top1 - top2
    sorted_p = np.sort(soft, axis=1)[:, ::-1]
    top1 = sorted_p[:, 0]
    top2 = sorted_p[:, 1] if K >= 2 else np.zeros(N)
    margin = top1 - top2
    am_idx = np.argsort(margin)
    am_idx = am_idx[margin[am_idx] < float(margin_threshold)]
    if am_idx.size == 0:
        return _wrap([], label="low_margin")
    # For each ambiguous pattern, pair it with a top-1 rep of its
    # SECOND choice prototype.
    top2_class = np.argsort(-soft, axis=1)[:, 1]
    # Reps per class
    reps = {}
    for c in range(K):
        order = np.argsort(-soft[:, c])
        reps[c] = order[: max(20, N // 100)]
    queue = []
    for amb in am_idx:
        c2 = int(top2_class[amb])
        if reps[c2].size == 0:
            continue
        partner = int(reps[c2][rng.integers(0, reps[c2].size)])
        if partner == int(amb):
            continue
        queue.append((int(amb), partner,
                       f"low_margin:p{int(top2_class[amb])}"))
    rng.shuffle(queue)
    if max_pairs is not None and len(queue) > int(max_pairs):
        queue = queue[: int(max_pairs)]
    return _wrap(queue, label="low_margin")
