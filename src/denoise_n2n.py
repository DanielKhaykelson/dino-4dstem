"""denoise_n2n.py -- self-supervised Noise2Noise denoising by binomial splitting.

Vendored, self-contained implementation of the "Noise2Noise, binomial" recipe
developed in the cellulose low-dose 4D-STEM sandbox (celldenoise).  Nothing here
imports that project or the parent GUI, so it ships with the app.

The idea in one paragraph
-------------------------
A single low-dose diffraction pattern is a Poisson draw: each pixel holds a
count k ~ Poisson(lambda), and lambda is the noise-free pattern we want.  Deal
those counts into two piles by flipping a fair coin for every electron --
h1 ~ Binomial(k, 1/2), h2 = k - h1.  By Poisson thinning h1 and h2 are two
*independent* half-dose exposures of the SAME underlying pattern.  That is
exactly the pair Noise2Noise needs (Lehtinen et al. 2018): a network trained to
map one noisy realisation to another independent noisy realisation of the same
signal converges to the signal itself, because the noise has zero conditional
mean.  No clean reference and no repeated acquisition are required -- one
exposure is enough.

Why the network is deliberately tiny
------------------------------------
The npj Comput. Mater. 10:243 (2024) result that matters here is a negative
one: a high-capacity U-Net fits the individual electron arrivals instead of the
structure.  Capacity is the regulariser -- base=16, depth=3 (~35k params) is on
purpose.  Going bigger makes low-dose denoising worse.

Dose-equivariance
------------------
The per-pattern normalisation and the input transform live INSIDE the model, so
a network trained on half-dose piles applies to the full exposure without a
scale mismatch.  Getting this wrong makes the denoiser worse than the raw data.

Windows note: no DataLoader / no worker processes are used anywhere, so the
machine's multiprocessing deadlock (see CLAUDE.md) cannot occur here.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================================
# Loss -- Poisson negative log-likelihood, the right one for counting data
# ==========================================================================

def poisson_nll(pred_rate: torch.Tensor, counts: torch.Tensor,
                eps: float = 1e-6) -> torch.Tensor:
    """-log p(counts | Poisson(pred_rate)), dropping the constant log(k!).

    ``pred_rate`` must be positive (the model uses a softplus head).  Unlike
    MSE this treats a pixel with zero counts as informative rather than as
    "no signal", which is the common case at a few electrons per pattern.
    """
    lam = pred_rate.clamp_min(eps)
    return (lam - counts * torch.log(lam)).mean()


# ==========================================================================
# A deliberately low-capacity U-Net that predicts a non-negative rate
# ==========================================================================

class _Block(nn.Module):
    def __init__(self, cin, cout, norm=True):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=not norm)]
        if norm:
            layers.append(nn.GroupNorm(min(8, cout), cout))
        layers.append(nn.GELU())
        layers.append(nn.Conv2d(cout, cout, 3, padding=1, bias=not norm))
        if norm:
            layers.append(nn.GroupNorm(min(8, cout), cout))
        layers.append(nn.GELU())
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class SmallUNet(nn.Module):
    """Low-capacity U-Net taking raw counts, returning an expected-counts map.

    The input transform and the per-pattern dose normalisation both live
    inside the module: the input is divided by its own total counts before the
    network sees it and the output is multiplied back, so the network only ever
    learns the pattern's *shape* and the result is dose-equivariant.  Train on
    half-dose piles, apply to the full exposure.
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 16,
                 depth: int = 3, norm: bool = True,
                 input_transform: str = "anscombe"):
        super().__init__()
        self.depth = depth
        self.input_transform = input_transform
        chs = [base * (2 ** i) for i in range(depth + 1)]
        self.inc = _Block(in_ch, chs[0], norm)
        self.downs = nn.ModuleList(
            [_Block(chs[i], chs[i + 1], norm) for i in range(depth)])
        self.ups = nn.ModuleList(
            [_Block(chs[i + 1] + chs[i], chs[i], norm)
             for i in reversed(range(depth))])
        self.outc = nn.Conv2d(chs[0], out_ch, 1)

    def _normalise(self, x):
        npx = x.shape[-1] * x.shape[-2]
        s = x.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
        return x / s * npx, s / npx

    def _transform(self, x):
        if self.input_transform == "anscombe":
            return 2.0 * torch.sqrt(x.clamp_min(0.0) + 0.375)
        if self.input_transform == "log1p":
            return torch.log1p(x.clamp_min(0.0))
        return x

    def _backbone(self, x):
        feats = []
        h = self.inc(x)
        for d in self.downs:
            feats.append(h)
            h = d(F.max_pool2d(h, 2))
        for u, skip in zip(self.ups, reversed(feats)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = u(torch.cat([h, skip], dim=1))
        return self.outc(h)

    def forward(self, x):
        xn, scale = self._normalise(x)
        h = self._backbone(self._transform(xn))
        out = F.softplus(h)
        return out * scale

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ==========================================================================
# Binomial split -- Poisson thinning of one exposure into two halves
# ==========================================================================

def binomial_split(counts: torch.Tensor,
                   generator: Optional[torch.Generator] = None):
    """Deal integer counts into two independent halves by a fair coin per event.

    ``counts`` is clamped to >= 0 and rounded, then h1 ~ Binomial(counts, 1/2)
    and h2 = counts - h1.  The two halves are independent half-dose exposures of
    the same underlying pattern (Poisson thinning).
    """
    c = counts.clamp_min(0.0).round()
    p = torch.full_like(c, 0.5)
    h1 = torch.binomial(c, p, generator=generator)
    return h1, c - h1


def estimate_count_quantum(frames, max_frames: int = 48) -> float:
    """Value of a SINGLE detected electron, which is often not 1.0.

    Counting detectors are usually stored gain/flat-field corrected, so one
    electron lands at, say, 0.70 rather than 1 (and varies a little per pixel).
    Poisson thinning needs genuine integer counts: rounding gain-corrected data
    directly maps one event (0.70) to 1 but *two* events (1.40) also to 1,
    which silently destroys the split.  So recover the quantum and divide by it.

    Returns 1.0 when the data already are integer counts.
    """
    vals = []
    for f in list(frames)[:max_frames]:
        a = np.asarray(f, dtype=np.float64).ravel()
        p = a[a > 0]
        if p.size:
            vals.append(p)
    if not vals:
        return 1.0
    p = np.concatenate(vals)
    # already integral?  (check before anything else)
    s = p[:200000]
    if float(np.mean(np.abs(s - np.round(s)) < 1e-4)) > 0.98:
        return 1.0
    lo = float(np.percentile(p, 1.0))
    if not np.isfinite(lo) or lo <= 0:
        return 1.0
    band = p[(p >= 0.6 * lo) & (p <= 1.6 * lo)]
    q = float(np.median(band)) if band.size >= 32 else lo
    if not (np.isfinite(q) and q > 0):
        return 1.0
    # within a few percent of unity: treat as counts.  Per-pixel gain scatter
    # puts a few events below 1, but rescaling by ~1% buys nothing and would
    # cost a full-array division on every row of a large cube.
    return 1.0 if abs(q - 1.0) < 0.05 else q


def looks_like_counts(sample: np.ndarray) -> bool:
    """True if the data look like raw integer counts (the valid N2N regime).

    Binomial splitting is only exact for Poisson counts.  If the cube has
    already been vmax-normalised / blurred / otherwise turned into small floats
    the split becomes a heuristic; the caller should warn.
    """
    a = np.asarray(sample, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return False
    if finite.min() < -1e-6:
        return False
    frac = np.abs(finite - np.round(finite))
    return float(np.mean(frac < 1e-4)) > 0.98 and float(finite.max()) >= 1.0


# ==========================================================================
# Training
# ==========================================================================

class Cancelled(Exception):
    """Raised when a ``cancel()`` callback asks the run to stop."""


def _sample_training_stack(cube, max_patterns: int, seed: int = 0,
                           progress: Optional[Callable[[int, int], None]] = None,
                           cancel: Optional[Callable[[], bool]] = None):
    """Even-ish random subsample of scan positions as an (n, H, W) float32 stack.

    Read in **bulk, one scan row at a time**, not frame by frame.  On a lazy or
    compressed cube (HDF5, .mib, memmap) a single ``cube[y, x]`` can pull and
    decompress a whole chunk, so gathering a few thousand scattered frames is
    the slowest part of the entire run -- minutes of apparent hang.  Reading
    whole rows turns that into a few dozen sequential bulk reads.

    Coverage is kept by spreading the sampled rows evenly down the scan and the
    sampled columns evenly across each one, rather than taking one contiguous
    block.  Works for 4-D .npy cubes, ``read_block`` lazy wrappers and
    3-D-backed h5 wrappers, falling back to per-frame indexing if needed.
    """
    Ny, Nx = cube.shape[:2]
    H, W = cube.shape[-2], cube.shape[-1]

    # up to 64 bulk row reads, evenly spaced down the scan
    rows = np.unique(np.linspace(0, Ny - 1, min(Ny, 64)).astype(int))
    per_row = int(min(Nx, max(1, int(np.ceil(max_patterns / len(rows))))))
    cols = np.unique(np.linspace(0, Nx - 1, per_row).astype(int))

    reader = getattr(cube, "read_block", None)
    chunks = []
    n_tot = len(rows) * len(cols)
    got = 0
    for k, y in enumerate(rows):
        if cancel is not None and cancel():
            raise Cancelled("cancelled while loading training patterns")
        y = int(y)
        if reader is not None:
            take = np.asarray(reader(y, 1, 0, Nx)[0], dtype=np.float32)[cols]
        else:
            try:
                # fancy-index the columns we actually want: on a memmap this
                # reads only those frames instead of the whole scan row
                take = np.asarray(cube[y, cols], dtype=np.float32)
            except Exception:
                try:
                    take = np.asarray(cube[y], dtype=np.float32)[cols]
                except Exception:
                    take = np.stack([np.asarray(cube[y, x], dtype=np.float32)
                                     for x in cols], 0)
        chunks.append(take)
        got += len(cols)
        if progress is not None:
            progress(got, n_tot)
    return np.concatenate(chunks, 0).astype(np.float32, copy=False)


def train_binomial_n2n(cube, *, target_steps: int = 10000,
                       min_steps: int = 1500,
                       min_epochs: int = 20, max_epochs: int = 250,
                       lr: float = 2e-3, crop: int = 192,
                       max_seconds: float = 600.0,
                       batch_size: int = 16, base: int = 16, depth: int = 3,
                       input_transform: str = "anscombe",
                       max_patterns: int = 4096, val_frac: float = 0.15,
                       auto_quantum: bool = True,
                       device: Optional[str] = None, seed: int = 0,
                       progress: Optional[Callable[[int, int, float, float],
                                                   None]] = None,
                       load_progress: Optional[Callable[[int, int],
                                                        None]] = None,
                       step_progress: Optional[Callable[[int, int],
                                                        None]] = None,
                       cancel: Optional[Callable[[], bool]] = None):
    """Train a SmallUNet on binomially-split pairs drawn from ``cube``.

    A fresh binomial split is sampled every step, so the network never sees the
    same pair twice and cannot memorise a noise realisation.  Both directions
    (h1->h2 and h2->h1) are used.  The validation loss is the Poisson NLL of the
    prediction against a held-out independent half -- it has a noise floor and
    will not go to zero; a steadily falling value means the fit is real.

    Training length is set by a fixed **gradient-step budget** (``target_steps``)
    rather than a fixed epoch count, so a small scan (few patterns, few steps
    per epoch) trains for more epochs and a large scan for fewer — both see a
    comparable number of weight updates and converge equally.  The epoch count
    is clamped to ``[min_epochs, max_epochs]``.

    Two things keep the cost bounded on a large detector:

    * ``crop`` — training runs on random crops of this size (the network is
      fully convolutional, so inference still uses whole frames).  Step cost
      then does not grow with detector size.  A third of the crops sit near the
      frame centre so the direct beam's intensity regime is represented.
    * ``max_seconds`` — a few real steps are timed first, and the run is sized
      to fit that wall-clock budget, never dropping below ``min_steps``.

    Progress callbacks (all optional):
    ``load_progress(i, n)`` while the training patterns are read off the cube,
    ``step_progress(done, total)`` during training (throttled to ~200 calls),
    ``progress(epoch, n_epochs, train_loss, val_loss)`` once per epoch.
    ``cancel()`` is polled throughout; returning True raises :class:`Cancelled`.

    Returns ``(model, history)``.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available()
                                  else "cpu"))
    torch.manual_seed(seed)
    # Bound the training stack's memory: at 512x512 a 4096-frame stack is
    # 4.3 GB of RAM (and 4.3 GB to read).  Random crops mean far fewer whole
    # frames still give plenty of distinct training views.
    _H, _W = int(cube.shape[-2]), int(cube.shape[-1])
    max_patterns = max(64, min(int(max_patterns),
                               int(1.5e9 / max(_H * _W * 4, 1))))
    stack = _sample_training_stack(cube, max_patterns, seed=seed,
                                   progress=load_progress, cancel=cancel)
    # Convert gain-corrected intensities to genuine counts; the split is only
    # valid on integers.  quantum == 1.0 when the data already are counts.
    quantum = estimate_count_quantum(stack) if auto_quantum else 1.0
    if quantum != 1.0:
        stack = stack / np.float32(quantum)

    n = len(stack)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(batch_size, int(val_frac * n)) if n > 2 * batch_size else 0
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    # The training stack stays in CPU memory; only the current batch goes to
    # the device.  Preloading thousands of full frames onto the GPU costs GB
    # and leaves too little room for activations -- measured on an RTX 4080, a
    # batch-32 step at 256x256 takes 5.0 s that way versus 0.13 s at 192x192.
    tr_cpu = torch.from_numpy(np.ascontiguousarray(stack[tr_idx]))
    va_cpu = (torch.from_numpy(np.ascontiguousarray(stack[val_idx]))
              if n_val else None)
    del stack
    gen = torch.Generator(device=dev).manual_seed(seed)

    # Train on random CROPS when the detector is large.  The network is fully
    # convolutional, so it still applies to the full frame at inference, and
    # the cost per step becomes independent of detector size.  A third of the
    # crops are centred near the frame middle so the direct beam's very
    # different intensity regime is represented too.
    H, W = int(tr_cpu.shape[-2]), int(tr_cpu.shape[-1])
    ch, cw = min(int(crop), H), min(int(crop), W)
    cropping = (ch < H) or (cw < W)

    def _draw(src, idx):
        b = src[idx]
        if cropping:
            if rng.random() < 0.34:
                y0 = int(np.clip((H - ch) // 2 + rng.integers(-ch // 4,
                                                              ch // 4 + 1),
                                 0, H - ch))
                x0 = int(np.clip((W - cw) // 2 + rng.integers(-cw // 4,
                                                              cw // 4 + 1),
                                 0, W - cw))
            else:
                y0 = int(rng.integers(0, H - ch + 1))
                x0 = int(rng.integers(0, W - cw + 1))
            b = b[..., y0:y0 + ch, x0:x0 + cw]
        return b.unsqueeze(1).to(dev, non_blocking=True)

    steps = max(1, len(tr_idx) // batch_size)
    model = SmallUNet(base=base, depth=depth,
                      input_transform=input_transform).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def _one_step(idx):
        batch = _draw(tr_cpu, idx)
        h1, h2 = binomial_split(batch, generator=gen)
        # use both directions in one batch for twice the pairs
        x = torch.cat([h1, h2], 0)
        y = torch.cat([h2, h1], 0)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = poisson_nll(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        return float(loss), int(x.shape[0])

    # Probe with a few real steps to measure this machine's step time, then
    # size the run to the wall-clock budget.  Without this, a large detector
    # silently turns a 7-minute job into a 14-hour one.
    probe_idx = torch.arange(min(batch_size, len(tr_cpu)))
    for _ in range(3):                       # warm-up: lazy init, autotune
        _one_step(probe_idx)                 # (timing these overestimates ~20x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_probe = time.time()
    for _ in range(5):
        _one_step(probe_idx)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    step_s = max((time.time() - t_probe) / 5.0, 1e-6)

    affordable = int(max_seconds / step_s)
    planned = int(np.clip(min(target_steps, affordable), min_steps,
                          target_steps))
    epochs = int(np.clip(int(np.ceil(planned / steps)), min_epochs,
                         max_epochs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Remember the training geometry on the model so inference can match it
    # automatically -- applying a crop-trained net to a whole large frame
    # costs ~32% of the counts and distorts the radial profile.
    model._train_crop = int(min(ch, cw)) if cropping else 0

    hist = {"train_loss": [], "val_loss": [], "n_params": model.n_params,
            "n_train": int(len(tr_idx)), "device": str(dev),
            "epochs": epochs, "steps_per_epoch": steps,
            "crop": [ch, cw] if cropping else None,
            "quantum": quantum,
            "step_seconds": step_s,
            "est_minutes": epochs * steps * step_s / 60.0}

    total_steps = epochs * steps
    gstep = 0
    t_last = 0.0

    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(tr_idx))
        tl, seen = 0.0, 0
        for s in range(steps):
            if cancel is not None and cancel():
                raise Cancelled("cancelled during training")
            sel = order[s * batch_size:(s + 1) * batch_size]
            if sel.numel() == 0:
                continue
            loss_v, nb = _one_step(sel)
            tl += loss_v * nb
            seen += nb
            gstep += 1
            # time-throttled so the very first step already reports
            now = time.time()
            if step_progress is not None and (now - t_last > 0.3
                                              or gstep == total_steps):
                t_last = now
                step_progress(gstep, total_steps)
        sched.step()

        vloss = float("nan")
        if va_cpu is not None:
            model.eval()
            with torch.no_grad():
                vsel = torch.arange(min(batch_size, len(va_cpu)))
                vb = _draw(va_cpu, vsel)
                vh1, vh2 = binomial_split(vb, generator=gen)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    vloss = float(poisson_nll(model(vh1), vh2))
                # Baseline: predict the other half with THIS half, i.e. do
                # nothing.  A denoiser that cannot beat it is not earning its
                # keep -- usually a run that was cut too short.  Under-training
                # produces output measurably worse than the raw data, so this
                # is reported and surfaced rather than left silent.
                hist["val_loss_identity"] = float(
                    poisson_nll(vh1.clamp_min(1e-6).float(), vh2.float()))
            model.train()
        hist["train_loss"].append(tl / max(seen, 1))
        hist["val_loss"].append(vloss)
        if progress is not None:
            progress(ep + 1, epochs, hist["train_loss"][-1], vloss)

    return model, hist


# ==========================================================================
# Inference over a whole cube (streamed row by row into a writable target)
# ==========================================================================

def _tile_windows(n: int, tile: int, overlap: int):
    """Start offsets covering ``n`` with ``tile``-sized windows."""
    if tile >= n:
        return [0]
    step = max(1, tile - overlap)
    starts = list(range(0, max(n - tile, 0) + 1, step))
    if starts[-1] != n - tile:
        starts.append(n - tile)
    return starts


def _blend_window(h: int, w: int, ramp: int, device) -> torch.Tensor:
    """Separable linear ramp: 1 in the interior, tapering over ``ramp`` px."""
    def ax(m):
        v = torch.ones(m, device=device)
        r = min(ramp, m // 2)
        if r > 0:
            e = torch.linspace(1.0 / (r + 1), 1.0, r, device=device)
            v[:r] = e
            v[-r:] = e.flip(0)
        return v
    return ax(h)[:, None] * ax(w)[None, :]


@torch.no_grad()
def _apply_tiled(model, x, tile: int, overlap: int, use_amp: bool):
    """Run the model over ``x`` (b,1,H,W) in ``tile``-sized windows.

    The model normalises by its input's own total, so a whole 512x512 frame
    (dominated by the direct beam) is a different input distribution from the
    crops it trained on -- which shows up as a systematic intensity deficit.
    Applying it tile-by-tile at the training crop size keeps inference and
    training on identical geometry.  Overlapping tiles are blended with a
    linear ramp so no seam appears.
    """
    b, _, H, W = x.shape
    acc = torch.zeros((b, 1, H, W), device=x.device, dtype=torch.float32)
    wsum = torch.zeros((1, 1, H, W), device=x.device, dtype=torch.float32)
    for y0 in _tile_windows(H, tile, overlap):
        for x0 in _tile_windows(W, tile, overlap):
            th, tw = min(tile, H), min(tile, W)
            patch = x[..., y0:y0 + th, x0:x0 + tw]
            with torch.amp.autocast("cuda", enabled=use_amp):
                p = model(patch)
            win = _blend_window(th, tw, overlap // 2, x.device)
            acc[..., y0:y0 + th, x0:x0 + tw] += p.float() * win
            wsum[..., y0:y0 + th, x0:x0 + tw] += win
    return acc / wsum.clamp_min(1e-8)


@torch.no_grad()
def denoise_cube_into(model, cube, out, *, device: Optional[str] = None,
                      batch_size: int = 32, quantum: float = 1.0,
                      tile: int = 0, overlap: int = 48,
                      progress: Optional[Callable[[int, int], None]] = None,
                      cancel: Optional[Callable[[], bool]] = None):
    """Apply ``model`` to every pattern of ``cube`` -> ``out`` (Ny,Nx,H,W).

    ``out`` is any array-like that supports ``out[y, x0:x1] = block`` (e.g. a
    ``np.lib.format.open_memmap``).  Streams a scan row at a time so an
    arbitrarily large cube never has to be materialised in memory.

    Feeding the FULL exposure (not a half) is correct: the model is
    dose-equivariant, so a network trained on half-dose piles returns the
    full-dose expected pattern here.  ``progress(row, Ny)`` per row.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available()
                                  else "cpu"))
    model = model.to(dev).eval()
    # Default to the geometry the model was trained on.  Measured on a 512x512
    # Dectris cube, applying a 192-crop-trained net to whole frames keeps only
    # 67.6% of the counts and skews the radial profile (ratio 0.58-0.88, worst
    # across the Bragg region); tiled at 192 it is 99.6% and 0.89-1.06.
    if not tile:
        tile = int(getattr(model, "_train_crop", 0) or 0)
    Ny, Nx = cube.shape[:2]
    H, W = cube.shape[-2], cube.shape[-1]
    use_amp = dev.type == "cuda"
    # keep batch*pixels bounded so a large detector cannot exhaust device
    # memory (which degrades to thrashing rather than a clean OOM)
    batch_size = max(1, min(int(batch_size), int(4.0e6 / max(H * W, 1))))
    for y in range(Ny):
        if cancel is not None and cancel():
            raise Cancelled("cancelled while applying to the cube")
        try:
            row = np.asarray(cube[y], dtype=np.float32)
        except Exception:
            row = np.stack([np.asarray(cube[y, x], dtype=np.float32)
                            for x in range(Nx)], 0)
        if quantum != 1.0:
            row = row / np.float32(quantum)          # -> counts
        res = np.empty((Nx, H, W), dtype=np.float32)
        for s in range(0, Nx, batch_size):
            t = torch.from_numpy(row[s:s + batch_size, None]).to(dev)
            if tile and (tile < H or tile < W):
                p = _apply_tiled(model, t, tile, overlap, use_amp)
            else:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    p = model(t)
            res[s:s + batch_size] = p.float().cpu().numpy()[:, 0]
        if quantum != 1.0:
            res *= np.float32(quantum)               # back to input units
        out[y] = res
        if progress is not None:
            progress(y + 1, Ny)


@torch.no_grad()
def denoise_frame(model, frame: np.ndarray,
                  device: Optional[str] = None) -> np.ndarray:
    """Denoise a single (H, W) pattern -- for the before/after preview."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available()
                                  else "cpu"))
    model = model.to(dev).eval()
    t = torch.from_numpy(np.asarray(frame, dtype=np.float32)[None, None]).to(dev)
    return model(t).float().cpu().numpy()[0, 0]
