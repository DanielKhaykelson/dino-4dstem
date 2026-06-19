"""
DINO-SR Ablation Model v4 — L2 ResNet, no CBAM, single-best checkpoint
=======================================================================
Changes from v3:
  - Single source of truth: best.pth (scorecard-driven) + latest.pth only.
    Removed 10 legacy best_*.pth variants that were drowning the "which
    epoch actually won" signal. Scorecard overall is the one criterion.
  - Optional prototype-orthogonality regularizer (lam_sep > 0):
    L_sep = mean_{k!=k'} [cos(c_k, c_k') - margin]^2_+
    Penalizes near-duplicate cluster centers.
  - Optional local-Potts spatial regularizer (lam_spatial > 0):
    L_spatial = mean over 4-neighbor pairs of (1 - <p_i, p_j>)
    Encourages scan-adjacent positions to share a label WITHOUT any
    long-range grouping pressure. Requires scan_shape to be passed.
  - Optional scorecard-driven T_fin controller (adjust_t=True).
  - Per-epoch reason line printed when scorecard fires.
  - w_ent defaults to 0 at call sites.

Total loss:
  L = L_DINO_CE  +  lam_sep * L_separation  +  lam_spatial * L_spatial

Unchanged from v3:
  - Quiet training: tqdm progress bar, no per-step prints.
  - Deterministic DataLoader via torch.Generator per run.
  - num_prototypes exposed as a sweep axis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, math, os, csv, json, time, random
import numpy as np
from torchvision import models
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode

try:
    from tqdm.auto import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from dino_sr_fixed import (
    CenterMask, ProjectionHead, Prototypes,
    get_augmentation_transforms, get_teacher_temp, get_teacher_momentum,
    _is_bn_or_bias_param, _compute_spatial_metrics, _compute_confidence_stats,
    _compute_mdp_cross_correlation, _compute_per_class_flip_rates,
    _compute_auto_score, _compute_total_score,
)

# =========================================================================
# 1. ENCODER
# =========================================================================
LAYER_OUT_DIMS = {1: 64, 2: 128, 3: 256, 4: 512}

class PlainSequentialEncoder(nn.Sequential):
    def __init__(self, *args, use_maxpool=True, out_dim=128):
        super().__init__(*args)
        self.use_maxpool = use_maxpool
        self.out_dim = out_dim
        self.gradcam_target_idx = None

    def forward(self, x):
        for m in self:
            x = m(x)
        if self.use_maxpool:
            return F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)
        else:
            return F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)

def create_encoder_resnet18_variableN(n_layers=2, use_maxpool=True):
    assert n_layers in (1, 2, 3, 4)
    resnet = models.resnet18(weights=None)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    layer_map = {1: resnet.layer1, 2: resnet.layer2, 3: resnet.layer3, 4: resnet.layer4}
    out_dim = LAYER_OUT_DIMS[n_layers]
    modules = [resnet.conv1, resnet.bn1, resnet.relu]
    for i in range(1, n_layers + 1):
        modules.append(layer_map[i])
    encoder = PlainSequentialEncoder(*modules, use_maxpool=use_maxpool, out_dim=out_dim)
    encoder.gradcam_target_idx = len(modules) - 1
    return encoder

# =========================================================================
# 2. AUGMENTATION FACTORY
# =========================================================================
_AUG_NAMES = {"rotation", "hflip", "vflip", "colorjitter", "blur", "centermask"}


class PolarMaskLeft(nn.Module):
    """Zero the first `k_cols` columns of a polar-transformed tensor.

    In our PolarTransform the column axis spans radii r ∈ [0, max_r_px]. With
    default max_radius_frac=1.0 and W=192, one column ≈ 0.5 input-pixel
    radius. So k_cols=30 masks the inner ~15 px (matches center_mask_radius=15)
    with a few extra columns of cushion for small beam-center offsets that
    otherwise produce a wave at small r in polar.

    Why this replaces the pre-rotation CenterMask when use_polar=True:
      - Pre-rotation CenterMask has a sharp disk edge. Rotation interpolates
        that edge, producing a soft fuzzy disc that leaks nonzero intensity
        into the polar output at small r.
      - Post-polar masking is a clean rectangular zero — no rotation-
        smoothing, no wave artifacts from any residual beam offset at those
        small r.
    """
    def __init__(self, k_cols: int = 30):
        super().__init__()
        self.k_cols = int(max(0, k_cols))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.k_cols == 0:
            return x
        out = x.clone()
        if out.dim() == 3:
            out[:, :, :self.k_cols] = 0
        elif out.dim() == 4:
            out[:, :, :, :self.k_cols] = 0
        return out


class CenterOnCOM(nn.Module):
    """Shift each pattern so the constrained center-of-mass falls at the
    image center, BEFORE the polar transform. The COM is computed only over
    intensity within a search disk of radius `search_radius` from the
    geometric center, so shadowing or off-center bright pixels can't drag
    the COM far away. Shift is applied via grid_sample (bilinear, zero pad).
    Operates on (C, H, W) or (N, C, H, W) tensors.
    """
    def __init__(self, search_radius: float):
        super().__init__()
        self.search_radius = float(search_radius)

    def forward(self, x):
        single = (x.dim() == 3)
        if single:
            x = x.unsqueeze(0)
        N, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        cy0 = (H - 1) / 2.0
        cx0 = (W - 1) / 2.0
        yy = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W)
        xx = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)
        d2 = (yy - cy0) ** 2 + (xx - cx0) ** 2
        search_mask = (d2 <= self.search_radius ** 2).to(dtype)        # (H, W)

        # Per-image intensity (sum across channels); apply search mask.
        intensity = x.sum(dim=1)                                       # (N, H, W)
        weighted = intensity * search_mask
        wsum = weighted.sum(dim=(-1, -2)).clamp_min(1e-12)             # (N,)
        cy_com = (weighted * yy).sum(dim=(-1, -2)) / wsum              # (N,)
        cx_com = (weighted * xx).sum(dim=(-1, -2)) / wsum              # (N,)

        # Translation to recenter COM at (cy0, cx0). grid_sample uses
        # normalized coords [-1, 1], so divide pixel shift by half-extent.
        ty = (cy0 - cy_com) * 2.0 / max(H - 1, 1)                       # (N,)
        tx = (cx0 - cx_com) * 2.0 / max(W - 1, 1)                       # (N,)
        theta = torch.zeros(N, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        # affine_grid with align_corners=False: signs follow grid_sample
        # convention where positive translation in theta moves the SOURCE
        # sampling location, so we negate to physically shift the image.
        theta[:, 0, 2] = -tx
        theta[:, 1, 2] = -ty
        grid = F.affine_grid(theta, [N, C, H, W], align_corners=False)
        out = F.grid_sample(x, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=False)
        return out.squeeze(0) if single else out


class PolarTransform(nn.Module):
    """Cartesian → polar warp, as the last step in the aug pipeline.

    After CenterMask + rotation/flip/etc. operate on the Cartesian pattern,
    this maps (C, H, W) → (C, H_out, W_out) where:
      - row    = azimuthal angle (0 .. 2π), top to bottom
      - column = radial distance from image center (0 .. max_radius)

    Assumes the beam is at the image center (which the preprocessing and
    CenterMask already assume). Input in Cartesian; output in polar.

    Runs on-device via F.grid_sample — no CPU round-trip, works on batched
    or single tensors.

    Physics alignment:
      - Peak d-spacing ↔ column position  (user's stated separability axis)
      - Orientation / rotation ↔ row shift  (equivariant if we later use
        circular-θ convolutions; for a plain ResNet it's a data-aug-style
        easier invariance to learn than image-space rotation)
    """
    def __init__(self, output_size=192, max_radius_frac=1.0):
        super().__init__()
        self.output_size = int(output_size)
        self.max_radius_frac = float(max_radius_frac)

    def forward(self, x):
        single = (x.dim() == 3)
        if single:
            x = x.unsqueeze(0)
        B = x.size(0)
        device, dtype = x.device, x.dtype
        H = W = self.output_size
        # Radii in input-normalized coords [-1, 1]; image center → (0, 0).
        # max_radius_frac=1.0 reaches the inscribed-circle edge.
        theta = torch.linspace(0.0, 2.0 * math.pi, H, dtype=dtype, device=device)
        r     = torch.linspace(0.0, self.max_radius_frac, W, dtype=dtype, device=device)
        cos_t = torch.cos(theta).unsqueeze(1)    # (H, 1)
        sin_t = torch.sin(theta).unsqueeze(1)    # (H, 1)
        r_    = r.unsqueeze(0)                    # (1, W)
        gx = r_ * cos_t                           # (H, W)
        gy = r_ * sin_t                           # (H, W)
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        out = F.grid_sample(x, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=True)
        return out.squeeze(0) if single else out


def get_ablation_transforms(disable=None, center_mask_radius=15, center_crop_size=140,
                             use_polar: bool = False, polar_size: int = 192,
                             ellipse_affine=None,
                             polar_mask_cols: int = 30):
    """Build student / teacher aug pipelines.

    Pipeline order when all options are enabled:
      EllipticalCorrection (per-sample affine, undoes ring eccentricity)
        → rotation/flip/etc (Cartesian aug)
        → Resize
        → PolarTransform   (θ = rows, r = cols)
        → PolarMaskLeft    (zero first polar_mask_cols cols, equivalent to
                            a beam mask of radius ~= polar_mask_cols * max_r / W)

    For polar pipelines, CenterMask is SKIPPED in Cartesian space — its
    sharp disk edge gets smoothed by rotation interpolation (creating a
    fuzzy disc that leaks into polar low-r). PolarMaskLeft does the same
    job cleanly post-polar.

    polar_mask_cols=0 disables the polar mask entirely (useful for ablations).

    ellipse_affine: optional 2x2 array. None = identity.
    """
    if disable is None: disable = set()
    elif isinstance(disable, str): disable = {disable}
    else: disable = set(disable)
    unknown = disable - _AUG_NAMES
    if unknown: raise ValueError(f"Unknown: {unknown}")

    student_ops, teacher_ops = [], []

    # Elliptical correction first — so rings are circular BEFORE any aug.
    # Rotation aug is only a true symmetry on circular rings; otherwise it
    # introduces orientation-sensitive features the CNN can latch onto.
    if ellipse_affine is not None:
        from elliptical_correction import EllipticalCorrection
        student_ops.append(EllipticalCorrection(ellipse_affine))
        teacher_ops.append(EllipticalCorrection(ellipse_affine))

    # CenterMask in Cartesian: only when NOT doing polar. For polar pipelines
    # we mask post-polar via PolarMaskLeft (cleaner — no rotation-smoothed
    # disc edge, no sensitivity to small beam-center offsets).
    if not use_polar:
        if center_mask_radius > 0 and "centermask" not in disable:
            student_ops.append(CenterMask(radius=center_mask_radius))
            teacher_ops.append(CenterMask(radius=center_mask_radius))
        elif center_mask_radius > 0:
            teacher_ops.append(CenterMask(radius=center_mask_radius))

    if "rotation" not in disable:
        student_ops.append(T.RandomRotation(360, interpolation=InterpolationMode.BILINEAR, expand=True))
    student_ops.append(T.CenterCrop(center_crop_size))
    if "hflip" not in disable: student_ops.append(T.RandomHorizontalFlip(p=0.5))
    if "vflip" not in disable: student_ops.append(T.RandomVerticalFlip(p=0.5))
    if "colorjitter" not in disable: student_ops.append(T.ColorJitter(brightness=0.2, contrast=0.2))
    if "blur" not in disable: student_ops.append(T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 0.3)))
    student_ops.append(T.Resize(192, interpolation=InterpolationMode.BILINEAR, antialias=True))

    teacher_ops += [
        T.RandomRotation(15, interpolation=InterpolationMode.BILINEAR, expand=True),
        T.CenterCrop(center_crop_size),
        T.Resize(192, interpolation=InterpolationMode.BILINEAR, antialias=True),
    ]
    # Polar transform (if enabled) is the LAST geometric step on both views.
    # Cartesian augs first so they're physically well-defined; model sees
    # polar patterns throughout. PolarMaskLeft follows immediately to
    # block the direct-beam region in polar space.
    if use_polar:
        student_ops.append(PolarTransform(output_size=polar_size))
        teacher_ops.append(PolarTransform(output_size=polar_size))
        if polar_mask_cols > 0:
            student_ops.append(PolarMaskLeft(k_cols=polar_mask_cols))
            teacher_ops.append(PolarMaskLeft(k_cols=polar_mask_cols))
    return T.Compose(student_ops), T.Compose(teacher_ops)

# =========================================================================
# 3. MODEL
# =========================================================================
class AblationDINOModelSR(nn.Module):
    def __init__(self, n_layers=2, num_prototypes=10, proj_out_dim=256,
                 use_maxpool=True, w_ent=0.0, center_momentum=0.9,
                 T0=0.07, Tfin=0.07, EMA0=0.990, EMAfin=0.999, warmup_frac=0.2):
        super().__init__()
        self.architecture = "resnet"
        self.n_layers = n_layers
        self.student_encoder = create_encoder_resnet18_variableN(n_layers=n_layers, use_maxpool=use_maxpool)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        in_dim = self.student_encoder.out_dim
        self.student_projector = ProjectionHead(in_dim, 512, proj_out_dim)
        self.teacher_projector = ProjectionHead(in_dim, 512, proj_out_dim)
        self.prototypes = Prototypes(num_prototypes, proj_out_dim)
        self.register_buffer("center", torch.zeros(1, num_prototypes))
        self.center_momentum = float(center_momentum)
        self.T0, self.Tfin = T0, Tfin
        self.EMA0, self.EMAfin = EMA0, EMAfin
        self.warmup_frac, self.w_ent = warmup_frac, w_ent
        for p in self.teacher_encoder.parameters(): p.requires_grad = False
        for p in self.teacher_projector.parameters(): p.requires_grad = False

    def forward(self, x1, x2, temp_student=0.1, temp_teacher=0.06):
        s = self.student_projector(self.student_encoder(x1))
        logits_s = self.prototypes(s)
        self.teacher_encoder.eval(); self.teacher_projector.eval()
        with torch.no_grad():
            t = self.teacher_projector(self.teacher_encoder(x2))
            raw_t = self.prototypes(t)
        tl = raw_t - self.center
        pt = F.softmax(tl / temp_teacher, dim=-1)
        lps = F.log_softmax(logits_s / temp_student, dim=-1)
        loss = (-(pt.detach() * lps).sum(dim=-1)).mean()
        if self.w_ent > 0:
            ps = F.softmax(logits_s / temp_student, dim=-1).mean(dim=0)
            loss = loss - self.w_ent * -(ps * ps.clamp_min(1e-6).log()).sum()
        return loss, pt, raw_t

    @torch.no_grad()
    def update_teacher(self, momentum=0.996):
        for sp, tp in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            tp.data.mul_(momentum).add_(sp.data, alpha=1-momentum)
        for sp, tp in zip(self.student_projector.parameters(), self.teacher_projector.parameters()):
            tp.data.mul_(momentum).add_(sp.data, alpha=1-momentum)

    @torch.no_grad()
    def sync_teacher_bn_buffers(self):
        for (_, ms), (_, mt) in zip(self.student_encoder.named_modules(), self.teacher_encoder.named_modules()):
            if isinstance(ms, nn.modules.batchnorm._BatchNorm):
                mt.running_mean.copy_(ms.running_mean); mt.running_var.copy_(ms.running_var)
        for (_, ms), (_, mt) in zip(self.student_projector.named_modules(), self.teacher_projector.named_modules()):
            if isinstance(ms, nn.modules.batchnorm._BatchNorm):
                mt.running_mean.copy_(ms.running_mean); mt.running_var.copy_(ms.running_var)

    @torch.no_grad()
    def update_center(self, teacher_logits):
        bc = teacher_logits.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + bc * (1 - self.center_momentum)

    def _num_prototypes(self):
        return self.prototypes.prototypes.size(0)

# =========================================================================
# 4. TRAINING LOOP — quiet, full checkpoints
# =========================================================================
def train_ablation(model, dataset, epochs=50, lr=3e-4, weight_decay=1e-6,
                   device=None, outdir='./', save_every=5, seed=42,
                   unfreeze_epoch=0, grad_clip=1.0, batch_size=128,
                   student_aug=None, teacher_aug=None, scan_shape=None,
                   center_mask_radius=15, center_crop_size=140, disable_aug=None,
                   eval_callback=None, eval_every=10, eval_min_epoch=10,
                   lam_sep: float = 0.0, sep_margin: float = 0.3,
                   lam_spatial: float = 0.0,
                   adjust_t: bool = False):
    # v4: L_spatial (local Potts on scan neighbors) requires a spatial tile,
    # which needs scan_shape. Fail loudly if caller forgot to pass it.
    if lam_spatial > 0.0 and scan_shape is None:
        raise ValueError("lam_spatial > 0 requires scan_shape to be passed "
                         "(the training loop samples a spatial tile from it).")
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); torch.backends.cudnn.deterministic = True

    g = torch.Generator(); g.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=True, generator=g)

    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(outdir, exist_ok=True); model = model.to(device)

    no_decay, decay = _is_bn_or_bias_param(model)
    optimizer = torch.optim.AdamW([
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay}], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    if student_aug is None or teacher_aug is None:
        student_aug, teacher_aug = get_ablation_transforms(
            disable=disable_aug, center_mask_radius=center_mask_radius,
            center_crop_size=center_crop_size)
    K = model._num_prototypes()

    ag = torch.Generator(); ag.manual_seed(seed)
    al = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=ag)
    ad = next(iter(al))
    anchor_x = (ad[0] if isinstance(ad, (list, tuple)) else ad).to(device).float()
    with torch.no_grad(): anchor_x = teacher_aug(anchor_x)
    prev_assignments = None
    _, C_in, H_in, W_in = anchor_x.shape
    running_mdp = torch.zeros(K, C_in, H_in, W_in).to(device)
    lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32, device=device).view(1,1,3,3)

    use_spatial = (scan_shape is not None)
    spx = sps = psa = None
    if use_spatial:
        Ny, Nx = scan_shape; pa = min(batch_size, Ny*Nx)
        ph = int(math.sqrt(pa*Ny/max(Nx,1))); pw = pa//max(ph,1)
        ph, pw = min(ph, Ny), min(pw, Nx)
        y0, x0 = max(0,(Ny-ph)//2), max(0,(Nx-pw)//2)
        si = [r*Nx+c for r in range(y0,y0+ph) for c in range(x0,x0+pw) if r*Nx+c < len(dataset)]
        sb = torch.stack([(dataset[i][0] if isinstance(dataset[i],(list,tuple)) else dataset[i]) for i in si]).to(device).float()
        if sb.dim()==3: sb=sb.unsqueeze(1)
        with torch.no_grad(): spx = teacher_aug(sb)
        an = len(si); sps = (ph, min(pw, an//max(ph,1)))
        spx = spx[:sps[0]*sps[1]]

    # v4: single best tracker, driven by scorecard overall.
    # Legacy trackers (best_sharpness, best_diversity, best_distinct,
    # best_balanced, best_high_conf, best_stable_div, best_spatial, best_total,
    # best_auto) removed — they produced competing "best" checkpoints with
    # no principled way to pick among them.
    # v5 addition: best_loss.pth restored for comparison purposes (user asks
    # "does the lowest-loss epoch match the scorecard-best epoch?" — usually
    # no, and that's the whole point of having a scorecard). Only two best-*
    # saves total (best.pth + best_loss.pth) instead of the old ten.
    best_overall = -float('inf')
    best_loss_val = float('inf')
    scorecard_log = []

    if unfreeze_epoch > 0:
        for p in model.prototypes.parameters(): p.requires_grad = False
    pu = (unfreeze_epoch == 0)

    log_path = os.path.join(outdir, "training_log.csv")
    cols = ["epoch","avg_loss","effK","active_classes","flip_rate","max_class_flip",
            "sharpness","max_sim","mdp_max_xcorr","avg_conf","conf_median","conf_p10",
            "conf_p90","conf_bimodality","intensity_corr","spatial_coherence",
            "isolated_disagree","boundary_frac","spatial_flip","auto_score","total_score"]
    cf = open(log_path, "w", newline=''); wr = csv.writer(cf); wr.writerow(cols); cf.flush()

    t_start = time.perf_counter()
    eiter = range(epochs)
    if _HAS_TQDM: eiter = _tqdm(eiter, desc=os.path.basename(outdir), ncols=100, leave=False, mininterval=5)

    for epoch in eiter:
        if not pu and epoch >= unfreeze_epoch:
            for p in model.prototypes.parameters(): p.requires_grad = True
            pu = True
        model.train()
        tl = ts = 0; sp_sum = None; tcw = tccw = tcb = 0.0; acv = []
        tt = get_teacher_temp(epoch, epochs, model.T0, model.Tfin, model.warmup_frac)
        mm = get_teacher_momentum(epoch, epochs, model.EMA0, model.EMAfin)

        for batch in dataloader:
            x = (batch[0] if isinstance(batch,(list,tuple)) else batch).to(device, non_blocking=True).float()
            bs = x.size(0)
            with torch.no_grad():
                xs = student_aug(x.clone()); xt = teacher_aug(x.clone()); bi_ = x.sum(dim=(1,2,3))
            optimizer.zero_grad(set_to_none=True)
            loss, pt, rtl = model(xs, xt, temp_student=0.1, temp_teacher=tt)
            # v4: prototype-orthogonality regularizer. Penalizes pairs of
            # prototype vectors whose cosine similarity exceeds `sep_margin`.
            # Encourages distinct cluster centers without touching DINO CE.
            # Cheap: O(K^2) with K=10.
            if lam_sep > 0.0:
                p_norm = F.normalize(model.prototypes.prototypes, dim=-1)
                sim = p_norm @ p_norm.t()
                off = 1.0 - torch.eye(K, device=sim.device, dtype=sim.dtype)
                sep_loss = (F.relu(sim - sep_margin) ** 2 * off).sum() / max(K * (K - 1), 1)
                loss = loss + lam_sep * sep_loss

            # v4: local-Potts spatial regularizer. Encourages 4-neighbor scan
            # positions to share a label WITHOUT imposing any long-range
            # grouping pressure (guardrail: same phase far apart must remain
            # same class on its own — satisfied because only adjacent pairs
            # are penalized).
            # Student forward on the fixed spatial tile (teacher-augmented,
            # light aug only → preserves physical pattern content so neighbor
            # agreement is meaningful). Gradient flows to student params.
            # Cost: ph*pw ≈ 120 extra patterns/step for Na007b.
            if lam_spatial > 0.0 and use_spatial and spx is not None:
                sp_feat   = model.student_projector(model.student_encoder(spx))
                sp_logits = model.prototypes(sp_feat)
                sp_probs  = F.softmax(sp_logits / 0.1, dim=-1)  # student temp
                sp_probs_2d = sp_probs.view(sps[0], sps[1], K)
                # 4-neighbor Potts: 1 - <p_i, p_j> for right and down pairs
                right = (sp_probs_2d[:, :-1] * sp_probs_2d[:, 1:]).sum(-1)   # (ph, pw-1)
                down  = (sp_probs_2d[:-1, :] * sp_probs_2d[1:, :]).sum(-1)   # (ph-1, pw)
                spatial_loss = ((1.0 - right).mean() + (1.0 - down).mean()) * 0.5
                loss = loss + lam_spatial * spatial_loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step(); model.update_teacher(momentum=mm)
            model.sync_teacher_bn_buffers(); model.update_center(rtl)
            with torch.no_grad():
                tl += loss.item(); ts += bs
                sp_sum = pt.sum(dim=0).cpu() if sp_sum is None else sp_sum + pt.sum(dim=0).cpu()
                bmc = pt.max(dim=-1).values; tcw += bmc.mean().item()*bs; acv.extend(bmc.cpu().numpy().tolist())
                pr = pt.argmax(dim=-1).float()
                if pr.var()>1e-5 and bi_.var()>1e-5:
                    tccw += torch.corrcoef(torch.stack([bi_, pr]))[0,1].item()*bs; tcb += bs
                cm = bmc > 0.6
                if cm.any():
                    cxt = xt[cm]; cpr = pr[cm].long()
                    for ki in range(K):
                        mk = (cpr==ki)
                        if mk.any(): running_mdp[ki] = 0.9*running_mdp[ki] + 0.1*cxt[mk].mean(dim=0)
        scheduler.step()

        # Metrics
        avg_loss = tl/len(dataloader); avg_corr = tccw/max(tcb,1); avg_conf = tcw/max(ts,1)
        conf_median, conf_p10, conf_p90, conf_bimodality = _compute_confidence_stats(acv)
        with torch.no_grad():
            _, ap, _ = model(anchor_x, anchor_x, temp_teacher=tt)
            ca = ap.argmax(dim=-1)
            flip_rate = (ca!=prev_assignments).float().mean().item() if prev_assignments is not None else 1.0
            _, max_class_flip = _compute_per_class_flip_rates(ca, prev_assignments, K); prev_assignments = ca

        ch, cw_ = H_in//2, W_in//2; mc = running_mdp.clone(); p_ = 5
        mc[:,:,max(ch-p_,0):min(ch+p_,H_in),max(cw_-p_,0):min(cw_+p_,W_in)] = 0
        sl = [F.conv2d(mc[ki].unsqueeze(0), lap).var().item() for ki in range(K) if running_mdp[ki].sum()>0]
        avg_sharp = np.mean(sl) if sl else 0.0
        fp = running_mdp.view(K,-1); np_ = F.normalize(fp, p=2, dim=1, eps=1e-8)
        sm = torch.mm(np_, np_.t()); sm.masked_fill_(torch.eye(K,device=device).bool(),0); max_sim = sm.max().item()
        mdp_max_xcorr = _compute_mdp_cross_correlation(running_mdp, K)
        pb = (sp_sum/ts).clamp(1e-12,1.0); effK = math.exp(-(pb*pb.log()).sum().item()); active_classes = int((sp_sum>0).sum().item())

        sp_coh = sp_iso = sp_bnd = 0.0; sp_flip = 1.0
        if use_spatial and spx is not None:
            with torch.no_grad():
                _, spp, _ = model(spx, spx, temp_teacher=tt)
                spd = spp.argmax(dim=-1).cpu().numpy(); sh, sw = sps
                sp_coh, sp_iso, sp_bnd = _compute_spatial_metrics(spd[:sh*sw].reshape(sh,sw))
                if psa is not None: sp_flip = float((spd!=psa).mean())
                psa = spd.copy()

        mse = max(10, unfreeze_epoch+5)
        auto_score = _compute_auto_score(epoch, effK, K, flip_rate, avg_conf, avg_sharp, max_sim, avg_loss, min_epoch=mse)
        total_score = _compute_total_score(epoch, effK, K, flip_rate, max_class_flip, avg_conf, conf_bimodality, avg_sharp, max_sim, mdp_max_xcorr, sp_coh, sp_iso, avg_loss, min_epoch=mse, use_spatial=use_spatial)

        wr.writerow([epoch+1,f"{avg_loss:.4f}",f"{effK:.2f}",active_classes,f"{flip_rate:.4f}",f"{max_class_flip:.4f}",
            f"{avg_sharp:.6f}",f"{max_sim:.4f}",f"{mdp_max_xcorr:.4f}",f"{avg_conf:.4f}",f"{conf_median:.4f}",f"{conf_p10:.4f}",
            f"{conf_p90:.4f}",f"{conf_bimodality:.4f}",f"{avg_corr:.4f}",f"{sp_coh:.4f}",f"{sp_iso:.4f}",f"{sp_bnd:.4f}",f"{sp_flip:.4f}",
            f"{auto_score:.4f}",f"{total_score:.4f}"]); cf.flush()

        # Training-time preprocessing context — saved in every checkpoint so
        # downstream scoring can reproduce the exact dataset / transform the
        # model was trained on. Without this, standalone scoring defaults to
        # eval_all.SAMPLES values, which silently give wrong results for runs
        # that overrode vmax / mask from the command line.
        train_config_snapshot = {
            "vmax": getattr(dataset, "vmax", None),
            "center_mask_radius": center_mask_radius,
            "center_crop_size": center_crop_size,
            "resize": getattr(dataset, "resize", 192),
            "batch_size": batch_size,
            "seed": seed,
        }

        def mkc():
            return {'epoch':epoch+1,'architecture':'resnet','n_layers':model.n_layers,
                'num_prototypes':K,
                'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'teacher_temp':tt,'center':model.center.detach().cpu(),
                'train_config': dict(train_config_snapshot),
                'metrics':{'loss':avg_loss,'eff_k':effK,'active_classes':active_classes,'sharpness':avg_sharp,
                    'flip_rate':flip_rate,'max_class_flip':max_class_flip,'max_sim':max_sim,'mdp_max_xcorr':mdp_max_xcorr,
                    'avg_conf':avg_conf,'conf_median':conf_median,'conf_p10':conf_p10,'conf_p90':conf_p90,
                    'conf_bimodality':conf_bimodality,'intensity_corr':avg_corr,'spatial_coherence':sp_coh,
                    'isolated_disagree':sp_iso,'boundary_frac':sp_bnd,'spatial_flip':sp_flip,
                    'auto_score':auto_score,'total_score':total_score}}

        # v4/v5: keep the one scorecard-driven best.pth + best_loss.pth (for
        # comparison) + latest.pth + periodic ckpt_ep*.pth. The 10 legacy
        # best_*.pth checkpoints stay removed — they competed without a
        # principled picker and left the user guessing.
        torch.save(mkc(), os.path.join(outdir, "latest.pth"))
        if (epoch+1) % save_every == 0:
            torch.save(mkc(), os.path.join(outdir, f"ckpt_ep{epoch+1}.pth"))
        # best_loss.pth — lowest-avg-loss checkpoint. Typically NOT the best
        # class map (training loss keeps falling while the clustering quality
        # peaks earlier). Saved so users can A/B it vs best.pth.
        if avg_loss < best_loss_val:
            best_loss_val = avg_loss
            torch.save(mkc(), os.path.join(outdir, "best_loss.pth"))

        # ── Scorecard-driven checkpoint selection (single best.pth) ──
        # Runs every `eval_every` epochs starting from `eval_min_epoch`.
        # best.pth is overwritten whenever the scorecard overall improves.
        if eval_callback is not None and (epoch + 1) >= eval_min_epoch \
                and ((epoch + 1) % eval_every == 0 or (epoch + 1) == epochs):
            try:
                sc = eval_callback(model, epoch + 1, tt)
                scorecard_log.append({
                    "epoch": epoch + 1,
                    "overall": float(sc.overall),
                    "verdict": sc.verdict,
                    "components": {k: float(v) for k, v in sc.components.items()},
                })
                # Richer one-line reason: show the components that drove the
                # score so "why epoch N wins" is legible without extra tooling.
                comp = sc.components
                comp_bits = []
                for k in ("effk_score", "coherence_score", "mdp_distinctness",
                          "class_purity", "grain_compactness", "no_speckle"):
                    if k in comp:
                        comp_bits.append(f"{k.split('_')[0]}={comp[k]:.2f}")
                marker = "*BEST*" if sc.overall > best_overall else "      "
                print(f"  [sc ep{epoch+1:>3}] {marker} overall={sc.overall:.3f} "
                      f"{sc.verdict:<7} {' '.join(comp_bits)} "
                      f"weakest={sc.weakest_component}",
                      flush=True)
                if sc.overall > best_overall:
                    best_overall = sc.overall
                    ckpt = mkc()
                    ckpt["scorecard"] = {
                        "overall": float(sc.overall),
                        "verdict": sc.verdict,
                        "components": {k: float(v) for k, v in sc.components.items()},
                        "weakest_component": sc.weakest_component,
                        "retune_hint": sc.retune_hint,
                        "critical_failures": sc.critical_failures,
                    }
                    torch.save(ckpt, os.path.join(outdir, "best.pth"))

                # ── Phase C: scorecard-driven T_fin adjustment ──
                # Fires only one rule per scorecard call. Modifies model.Tfin
                # which get_teacher_temp() reads on subsequent epochs.
                if adjust_t:
                    from scorecard import scorecard_t_adjustment
                    new_tfin, rule, reason = scorecard_t_adjustment(
                        components=sc.components,
                        ckpt_metrics=sc.ckpt_metrics_used,
                        current_tfin=float(model.Tfin),
                        num_prototypes=K,
                    )
                    if rule is not None and abs(new_tfin - model.Tfin) > 1e-6:
                        # Update Tfin FIRST so the print can't prevent the
                        # adjustment (v4_fix regression: Unicode '->' arrow
                        # crashed the print on Windows cp1252 console, and
                        # the assignment after the print never ran).
                        old_tfin = float(model.Tfin)
                        model.Tfin = float(new_tfin)
                        print(f"  [T-ctrl ep{epoch+1}] Tfin "
                              f"{old_tfin:.4f} -> {new_tfin:.4f}  "
                              f"rule={rule}  ({reason})", flush=True)
            except Exception as exc:
                print(f"  [sc ep{epoch+1}] FAILED: {exc}", flush=True)

    cf.close()

    # Persist the scorecard trajectory alongside the training log.
    if scorecard_log:
        with open(os.path.join(outdir, "scorecard_log.json"), "w") as f:
            json.dump(scorecard_log, f, indent=2)

    return model, {'train_time_s': time.perf_counter()-t_start,
                   'train_time_per_epoch_s': (time.perf_counter()-t_start)/max(epochs,1),
                   'scorecard_log': scorecard_log,
                   'best_scorecard_overall': float(best_overall) if best_overall > -float('inf') else None}

# =========================================================================
# 5. CHECKPOINT LOADING
# =========================================================================
def load_ablation_checkpoint(path, device=None, num_prototypes=None, **kw):
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    nl = ckpt.get("n_layers", 2)
    # Auto-detect num_prototypes from checkpoint if not specified
    if num_prototypes is None:
        num_prototypes = ckpt.get("num_prototypes", 10)
    model = AblationDINOModelSR(n_layers=nl, num_prototypes=num_prototypes, **kw)
    model = model.to(device)
    r = model.load_state_dict(sd, strict=False)
    if r.missing_keys: print(f"  WARNING: Missing keys: {r.missing_keys}")
    if r.unexpected_keys: print(f"  WARNING: Unexpected keys: {r.unexpected_keys}")
    model.eval()
    et = ckpt.get("teacher_temp", 0.06)
    if torch.is_tensor(et): et = et.item()
    return model, et, ckpt.get("metrics", {})
