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
            row = np.asarray(reader(y, 1, 0, Nx)[0], dtype=np.float32)
        else:
            try:
                row = np.asarray(cube[y], dtype=np.float32)
            except Exception:
                row = np.stack([np.asarray(cube[y, x], dtype=np.float32)
                                for x in cols], 0)
                chunks.append(row); got += len(cols)
                if progress is not None:
                    progress(got, n_tot)
                continue
        chunks.append(row[cols])
        got += len(cols)
        if progress is not None:
            progress(got, n_tot)
    return np.concatenate(chunks, 0).astype(np.float32, copy=False)


def train_binomial_n2n(cube, *, target_steps: int = 10000,
                       min_epochs: int = 20, max_epochs: int = 250,
                       lr: float = 2e-3,
                       batch_size: int = 16, base: int = 16, depth: int = 3,
                       input_transform: str = "anscombe",
                       max_patterns: int = 4096, val_frac: float = 0.15,
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
    stack = _sample_training_stack(cube, max_patterns, seed=seed,
                                   progress=load_progress, cancel=cancel)
    n = len(stack)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(batch_size, int(val_frac * n)) if n > 2 * batch_size else 0
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    tr = torch.from_numpy(stack[tr_idx]).to(dev)
    va = torch.from_numpy(stack[val_idx]).to(dev) if n_val else None
    gen = torch.Generator(device=dev).manual_seed(seed)

    steps = max(1, len(tr_idx) // batch_size)
    # fixed step budget -> adaptive epoch count (clamped)
    epochs = int(np.clip(int(np.ceil(target_steps / steps)),
                         min_epochs, max_epochs))

    model = SmallUNet(base=base, depth=depth,
                      input_transform=input_transform).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hist = {"train_loss": [], "val_loss": [], "n_params": model.n_params,
            "n_train": int(len(tr_idx)), "device": str(dev),
            "epochs": epochs, "steps_per_epoch": steps}

    total_steps = epochs * steps
    report_every = max(1, total_steps // 200)
    gstep = 0

    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(tr_idx), device=dev)
        tl, seen = 0.0, 0
        for s in range(steps):
            if cancel is not None and cancel():
                raise Cancelled("cancelled during training")
            sel = order[s * batch_size:(s + 1) * batch_size]
            if sel.numel() == 0:
                continue
            batch = tr[sel][:, None]                       # (b,1,H,W) counts
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
            tl += float(loss) * x.shape[0]
            seen += x.shape[0]
            gstep += 1
            if step_progress is not None and (gstep % report_every == 0
                                              or gstep == total_steps):
                step_progress(gstep, total_steps)
        sched.step()

        vloss = float("nan")
        if va is not None:
            model.eval()
            with torch.no_grad():
                vh1, vh2 = binomial_split(va[:, None], generator=gen)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    vloss = float(poisson_nll(model(vh1), vh2))
        hist["train_loss"].append(tl / max(seen, 1))
        hist["val_loss"].append(vloss)
        if progress is not None:
            progress(ep + 1, epochs, hist["train_loss"][-1], vloss)

    return model, hist


# ==========================================================================
# Inference over a whole cube (streamed row by row into a writable target)
# ==========================================================================

@torch.no_grad()
def denoise_cube_into(model, cube, out, *, device: Optional[str] = None,
                      batch_size: int = 32,
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
    Ny, Nx = cube.shape[:2]
    H, W = cube.shape[-2], cube.shape[-1]
    use_amp = dev.type == "cuda"
    for y in range(Ny):
        if cancel is not None and cancel():
            raise Cancelled("cancelled while applying to the cube")
        try:
            row = np.asarray(cube[y], dtype=np.float32)
        except Exception:
            row = np.stack([np.asarray(cube[y, x], dtype=np.float32)
                            for x in range(Nx)], 0)
        res = np.empty((Nx, H, W), dtype=np.float32)
        for s in range(0, Nx, batch_size):
            t = torch.from_numpy(row[s:s + batch_size, None]).to(dev)
            with torch.amp.autocast("cuda", enabled=use_amp):
                p = model(t)
            res[s:s + batch_size] = p.float().cpu().numpy()[:, 0]
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
