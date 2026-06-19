"""
DINO-SR Unified — Fixed & Combined (v3 — comprehensive metrics)
=================================================================
Single-file drop-in replacement for both:
    utils_DINOSR_2layers_self_valdiating_Maxpool_attn.py   (ResNet-CBAM)
    utils_DINOSR_2layers_self_valdiating_MaxpoolVit.py     (ViT-Tiny)

Toggle via the `architecture` parameter:
    model = DINOModelSR(architecture="resnet", ...)
    model = DINOModelSR(architecture="vit", ...)

Bug fixes:
    [FIX  1]  Prototype unfreeze_epoch is now a variable (default 0)
    [FIX  2]  ViT teacher head BN buffers are synced (was missing)
    [FIX  3]  ViT training uses linear LR warmup + cosine decay
    [FIX  4]  Unified training loop — one function for both architectures
    [FIX  5]  Checkpoint saves architecture tag for safe reload
    [FIX  6]  Anchor batch is pre-augmented consistently for both archs
    [FIX  7]  Entropy regularisation carried over from ResNet to unified forward
    [FIX  8]  ResNet forward sets teacher to eval mode (preserved)
    [FIX  9]  Intensity correlation tracking unified
    [FIX 10]  safe_load_state_dict logs missing/unexpected keys
    [FIX 11]  CenterMask grid cache now checks device
    [FIX 12]  running_mdp update: removed keepdim=True shape mismatch
    [FIX 13]  avg_conf / avg_corr now sample-weighted
    [FIX 14]  effK computed with explicit .item()
    [FIX 15]  ResNet optimizer: BN weight+bias both get zero weight decay

New in v3 (comprehensive training metrics):
    - Spatial coherence from a fixed center patch of the scan grid
    - Isolated disagreement fraction (noise vs real grain boundaries)
    - Spatial flip rate (per-epoch change on the spatial patch)
    - Per-class flip rates on the anchor batch
    - Confidence distribution stats (median, P10, P90, bimodality)
    - MDP cross-correlation (duplicate cluster detection)
    - New checkpoints: best_spatial, best_total
    - Expanded auto-score incorporates all new metrics

Usage (training):
    from dino_sr_fixed import DINOModelSR, train_dino, get_augmentation_transforms

    model = DINOModelSR(architecture="resnet", num_prototypes=10)
    model = train_dino(model, dataloader, dataset=dataset, scan_shape=(Ny, Nx),
                       epochs=80, outdir="./run1")

Usage (evaluation / loading):
    from dino_sr_fixed import DINOModelSR, load_checkpoint

    model, eval_temp, metrics = load_checkpoint("./run1/best_auto.pth", device)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import csv
import random
import numpy as np
from torchvision import models
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode
try:
    from tqdm.auto import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# =========================================================================
# 1. TRANSFORMS (shared by both architectures)
# =========================================================================

class CenterMask(nn.Module):
    """
    Blocks out the central beam using a Center of Mass (Centroid) calculation.
    Automatically handles both Batches (4D) and Single Images (3D).
    """
    def __init__(self, radius=25, exponent=2.0):
        super().__init__()
        self.radius = radius
        self.exponent = exponent
        self.grid_y = None
        self.grid_x = None

    def forward(self, img):
        is_batch = img.dim() == 4
        if not is_batch:
            img = img.unsqueeze(0)

        B, C, H, W = img.shape
        device = img.device

        # [FIX 11] Also check device
        if (self.grid_y is None
                or self.grid_y.shape[-2:] != (H, W)
                or self.grid_y.device != device):
            y = torch.arange(H, device=device)
            x = torch.arange(W, device=device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
            self.grid_y = grid_y.unsqueeze(0)
            self.grid_x = grid_x.unsqueeze(0)

        img_flat = img.mean(dim=1)
        weights = img_flat ** self.exponent
        total_mass = weights.sum(dim=(1, 2)).view(B, 1, 1).clamp(min=1e-6)

        cy = (weights * self.grid_y).sum(dim=(1, 2)).view(B, 1, 1) / total_mass
        cx = (weights * self.grid_x).sum(dim=(1, 2)).view(B, 1, 1) / total_mass

        dist_sq = (self.grid_x - cx)**2 + (self.grid_y - cy)**2
        mask_binary = (dist_sq > self.radius**2).float().unsqueeze(1)

        out = img * mask_binary

        if not is_batch:
            return out.squeeze(0)
        return out


def get_augmentation_transforms(center_mask_radius=13, center_crop_size=150):
    """Augmentation pipeline shared by both ResNet and ViT.
    
    Args:
        center_mask_radius: radius of the center beam mask. Set to 0 to disable.
        center_crop_size: size of the center crop after rotation.
    """
    student_ops = []
    teacher_ops = []

    if center_mask_radius > 0:
        student_ops.append(CenterMask(radius=center_mask_radius))
        teacher_ops.append(CenterMask(radius=center_mask_radius))

    student_ops += [
        T.RandomRotation(360, interpolation=InterpolationMode.BILINEAR, expand=True),
        T.CenterCrop(center_crop_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 0.3)),
        T.Resize(192, interpolation=InterpolationMode.BILINEAR, antialias=True),
    ]

    teacher_ops += [
        T.RandomRotation(15, interpolation=InterpolationMode.BILINEAR, expand=True),
        T.CenterCrop(center_crop_size),
        T.Resize(192, interpolation=InterpolationMode.BILINEAR, antialias=True),
    ]

    return T.Compose(student_ops), T.Compose(teacher_ops)


# =========================================================================
# 2. CBAM ATTENTION MODULES (ResNet path)
# =========================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        spatial_attn_map = self.sa(x)
        x = x * spatial_attn_map
        return x, spatial_attn_map


# =========================================================================
# 3. RESNET ENCODER
# =========================================================================

class CBAMSequential(nn.Sequential):
    """Custom Sequential that handles the CBAM tuple output."""
    def __init__(self, *args, use_maxpool=True):
        super().__init__(*args)
        self.use_maxpool = use_maxpool
        self.last_spatial_attn = None
        self.out_dim = 128

    def forward(self, x):
        for m in self:
            if isinstance(m, CBAMBlock):
                x, spatial_attn = m(x)
                self.last_spatial_attn = spatial_attn
            else:
                x = m(x)

        if self.use_maxpool:
            return F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)
        else:
            return F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)


def create_encoder_resnet18_layer2_1ch(use_maxpool=True):
    resnet = models.resnet18(weights=None)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    cbam = CBAMBlock(in_planes=128)
    modules = [resnet.conv1, resnet.bn1, resnet.relu, resnet.layer1, resnet.layer2, cbam]
    encoder = CBAMSequential(*modules, use_maxpool=use_maxpool)
    return encoder


# =========================================================================
# 4. VIT-TINY ENCODER
# =========================================================================

class PatchEmbed(nn.Module):
    def __init__(self, img_size=192, patch_size=16, in_chans=1, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=3, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if not return_attention:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            return x
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            return attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x, return_attention=False):
        if return_attention:
            return self.attn(self.norm1(x), return_attention=True)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerTiny(nn.Module):
    def __init__(self, img_size=192, patch_size=16, in_chans=1,
                 embed_dim=192, depth=12, num_heads=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


# =========================================================================
# 5. PROJECTION HEADS
# =========================================================================

class ProjectionHead(nn.Module):
    """ResNet projection head: Linear → BN → ReLU → Linear."""
    def __init__(self, in_dim=128, hidden_dim=512, out_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return x


class DINOHead(nn.Module):
    """ViT DINO head: MLP → L2-norm → weight-normed last layer."""
    def __init__(self, in_dim, out_dim, hidden_dim=512, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


# =========================================================================
# 6. PROTOTYPE LAYERS
# =========================================================================

class Prototypes(nn.Module):
    """ResNet-style prototypes: cosine similarity with learnable cluster centers."""
    def __init__(self, num_prototypes, proj_dim):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, proj_dim))

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        return x @ p.t()


# =========================================================================
# 7. UNIFIED DINO MODEL
# =========================================================================

class DINOModelSR(nn.Module):
    """
    Unified DINO Self-Supervised model supporting both ResNet-CBAM and ViT-Tiny.
    """
    def __init__(
        self,
        architecture="resnet",
        encoder=None,
        num_prototypes=10,
        proj_out_dim=256,
        use_maxpool=True,
        w_ent=0.0,
        center_momentum=0.995,
        T0=0.09,
        Tfin=0.08,
        EMA0=0.994,
        EMAfin=0.996,
        warmup_frac=0.30,
        vit_img_size=192,
        vit_patch_size=16,
        vit_embed_dim=192,
        vit_depth=12,
        vit_num_heads=3,
    ):
        super().__init__()
        self.architecture = architecture.lower()
        assert self.architecture in ("resnet", "vit"), \
            f"architecture must be 'resnet' or 'vit', got '{architecture}'"

        if encoder is not None:
            self.student_encoder = encoder
        elif self.architecture == "resnet":
            self.student_encoder = create_encoder_resnet18_layer2_1ch(use_maxpool=use_maxpool)
        else:
            self.student_encoder = VisionTransformerTiny(
                img_size=vit_img_size, patch_size=vit_patch_size,
                in_chans=1, embed_dim=vit_embed_dim,
                depth=vit_depth, num_heads=vit_num_heads,
            )

        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        in_dim = getattr(self.student_encoder, "out_dim", 128)

        if self.architecture == "resnet":
            self.student_projector = ProjectionHead(in_dim, 512, proj_out_dim)
            self.teacher_projector = ProjectionHead(in_dim, 512, proj_out_dim)
            self.prototypes = Prototypes(num_prototypes, proj_out_dim)
        else:
            self.student_head = DINOHead(in_dim, proj_out_dim)
            self.teacher_head = DINOHead(in_dim, proj_out_dim)
            self.prototypes = nn.Linear(proj_out_dim, num_prototypes, bias=False)
            self.prototypes.weight.data.normal_(0, 0.02)

        self.register_buffer("center", torch.zeros(1, num_prototypes))
        self.center_momentum = float(center_momentum)
        self.T0 = T0
        self.Tfin = Tfin
        self.EMA0 = EMA0
        self.EMAfin = EMAfin
        self.warmup_frac = warmup_frac
        self.w_ent = w_ent

        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        if self.architecture == "resnet":
            for p in self.teacher_projector.parameters():
                p.requires_grad = False
        else:
            for p in self.teacher_head.parameters():
                p.requires_grad = False

    def forward(self, x1, x2, temp_student=0.1, temp_teacher=0.06):
        if self.architecture == "resnet":
            s = self.student_projector(self.student_encoder(x1))
            logits_student = self.prototypes(s)
        else:
            s_feat = self.student_head(self.student_encoder(x1))
            logits_student = self.prototypes(F.normalize(s_feat, dim=-1))

        self.teacher_encoder.eval()
        if self.architecture == "resnet":
            self.teacher_projector.eval()
        else:
            self.teacher_head.eval()

        with torch.no_grad():
            if self.architecture == "resnet":
                t = self.teacher_projector(self.teacher_encoder(x2))
                raw_teacher_logits = self.prototypes(t)
            else:
                t_feat = self.teacher_head(self.teacher_encoder(x2))
                raw_teacher_logits = self.prototypes(F.normalize(t_feat, dim=-1))

            teacher_logits = raw_teacher_logits - self.center
            probs_teacher = F.softmax(teacher_logits / temp_teacher, dim=-1)

        log_probs_student = F.log_softmax(logits_student / temp_student, dim=-1)
        dino_loss = (-(probs_teacher.detach() * log_probs_student).sum(dim=-1)).mean()

        if self.w_ent > 0:
            probs_student = F.softmax(logits_student / temp_student, dim=-1)
            avg_student_probs = probs_student.mean(dim=0)
            student_entropy = -(avg_student_probs * avg_student_probs.clamp_min(1e-6).log()).sum()
            loss = dino_loss - (self.w_ent * student_entropy)
        else:
            loss = dino_loss

        return loss, probs_teacher, raw_teacher_logits

    @torch.no_grad()
    def update_teacher(self, momentum=0.996):
        for sp, tp in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            tp.data.mul_(momentum).add_(sp.data, alpha=(1 - momentum))
        if self.architecture == "resnet":
            for sp, tp in zip(self.student_projector.parameters(), self.teacher_projector.parameters()):
                tp.data.mul_(momentum).add_(sp.data, alpha=(1 - momentum))
        else:
            for sp, tp in zip(self.student_head.parameters(), self.teacher_head.parameters()):
                tp.data.mul_(momentum).add_(sp.data, alpha=(1 - momentum))

    @torch.no_grad()
    def sync_teacher_bn_buffers(self):
        for (_, m_s), (_, m_t) in zip(
            self.student_encoder.named_modules(), self.teacher_encoder.named_modules()
        ):
            if isinstance(m_s, nn.modules.batchnorm._BatchNorm):
                m_t.running_mean.copy_(m_s.running_mean)
                m_t.running_var.copy_(m_s.running_var)
        if self.architecture == "resnet":
            s_head, t_head = self.student_projector, self.teacher_projector
        else:
            s_head, t_head = self.student_head, self.teacher_head
        for (_, m_s), (_, m_t) in zip(s_head.named_modules(), t_head.named_modules()):
            if isinstance(m_s, nn.modules.batchnorm._BatchNorm):
                m_t.running_mean.copy_(m_s.running_mean)
                m_t.running_var.copy_(m_s.running_var)

    @torch.no_grad()
    def update_center(self, teacher_logits):
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    @torch.no_grad()
    def track_prototype_usage(self, probs_teacher):
        K = self._num_prototypes()
        counts = torch.bincount(probs_teacher.argmax(dim=1), minlength=K)
        active = int((counts > 0).sum().item())
        p = probs_teacher.mean(dim=0)
        entropy = float(-(p * p.clamp_min(1e-8).log()).sum().item())
        return active, entropy

    def _num_prototypes(self):
        if isinstance(self.prototypes, Prototypes):
            return self.prototypes.prototypes.size(0)
        else:
            return self.prototypes.weight.size(0)


# =========================================================================
# 8. SCHEDULES
# =========================================================================

def get_teacher_temp(epoch, total_epochs, temp_start=0.08, temp_end=0.06, warmup_frac=0.5):
    warm = max(1, int(warmup_frac * total_epochs))
    if epoch < warm:
        return temp_start + (temp_end - temp_start) * min(1.0, epoch / warm)
    return temp_end


def get_teacher_momentum(epoch, total_epochs, base_m=0.994, final_m=0.999):
    return final_m + 0.5 * (base_m - final_m) * (
        1.0 + math.cos(math.pi * epoch / max(1, total_epochs))
    )


def _get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr / optimizer.defaults['lr'],
                   0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def _is_bn_or_bias_param(model):
    no_decay, decay = [], []
    for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                no_decay.append(param)
            elif pname == 'bias':
                no_decay.append(param)
            else:
                decay.append(param)
    return no_decay, decay


# =========================================================================
# 9. SPATIAL METRICS (computed on a fixed center patch)
# =========================================================================

def _compute_spatial_metrics(assignments_2d):
    """
    Compute spatial coherence metrics from a 2D assignment map (numpy).

    Returns:
        spatial_coherence:  fraction of 4-connected neighbor pairs that agree (0-1)
        isolated_disagree:  fraction of disagreeing pixels that are isolated (0-1)
                            high = noise, low = real grain boundaries
        boundary_fraction:  fraction of all pixels on a boundary (expect 0.05-0.15)
    """
    h, w = assignments_2d.shape
    if h < 3 or w < 3:
        return 0.0, 0.0, 0.0

    # 4-connected neighbor agreement
    agree_h = (assignments_2d[:-1, :] == assignments_2d[1:, :])   # vertical
    agree_w = (assignments_2d[:, :-1] == assignments_2d[:, 1:])   # horizontal
    total_pairs = agree_h.size + agree_w.size
    total_agree = agree_h.sum() + agree_w.sum()
    spatial_coherence = float(total_agree) / max(total_pairs, 1)

    # Boundary pixels: any pixel where at least one 4-neighbor disagrees
    is_boundary = np.zeros((h, w), dtype=bool)
    is_boundary[:-1, :] |= ~agree_h   # pixel (i,j) disagrees with (i+1,j)
    is_boundary[1:, :]  |= ~agree_h   # pixel (i+1,j) disagrees with (i,j)
    is_boundary[:, :-1] |= ~agree_w
    is_boundary[:, 1:]  |= ~agree_w
    boundary_fraction = float(is_boundary.sum()) / max(h * w, 1)

    # Isolated disagreement: boundary pixel where NO neighbor is also a boundary pixel
    n_boundary = int(is_boundary.sum())
    if n_boundary == 0:
        return spatial_coherence, 0.0, 0.0

    # Pad for safe neighbor access
    bp = np.pad(is_boundary, 1, mode='constant', constant_values=False)
    isolated = np.zeros_like(is_boundary)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        isolated |= bp[1+dy:h+1+dy, 1+dx:w+1+dx]
    # isolated now has "any neighbor is also boundary" — invert for truly isolated
    truly_isolated = is_boundary & ~isolated
    # Actually, re-think: a boundary pixel is "isolated" if NONE of its neighbors
    # are also boundary pixels. Let's fix:
    has_boundary_neighbor = np.zeros_like(is_boundary)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        has_boundary_neighbor |= bp[1+dy:h+1+dy, 1+dx:w+1+dx]
    truly_isolated = is_boundary & ~has_boundary_neighbor
    isolated_frac = float(truly_isolated.sum()) / max(n_boundary, 1)

    return spatial_coherence, isolated_frac, boundary_fraction


def _compute_confidence_stats(all_conf_values):
    """
    Confidence distribution statistics from a 1D array of max-softmax probs.

    Returns:
        conf_median, conf_p10, conf_p90, conf_bimodality
    """
    if len(all_conf_values) < 10:
        return 0.0, 0.0, 0.0, 0.0

    arr = np.array(all_conf_values)
    median = float(np.median(arr))
    p10 = float(np.percentile(arr, 10))
    p90 = float(np.percentile(arr, 90))

    # Bimodality coefficient: (skewness^2 + 1) / kurtosis
    # BC > 0.555 suggests bimodal (Pfister et al. 2013)
    # For our case: bimodal = good (most confident + some uncertain boundary pixels)
    mean = arr.mean()
    std = arr.std()
    if std < 1e-8:
        return median, p10, p90, 0.0
    skew = float(((arr - mean) ** 3).mean() / (std ** 3))
    kurt = float(((arr - mean) ** 4).mean() / (std ** 4))
    if kurt < 1e-8:
        return median, p10, p90, 0.0
    bimodality = (skew ** 2 + 1.0) / kurt

    return median, p10, p90, bimodality


def _compute_mdp_cross_correlation(running_mdp, K):
    """
    Detect duplicate clusters via normalized cross-correlation of mean
    diffraction patterns.  Returns max cross-corr between any pair (0-1).
    Values > 0.95 suggest redundant clusters.
    """
    flat = running_mdp.view(K, -1)
    # Only consider active clusters
    norms = flat.norm(dim=1)
    active_mask = norms > 1e-6
    active_idx = torch.where(active_mask)[0]

    if len(active_idx) < 2:
        return 0.0

    active_flat = flat[active_idx]
    active_norms = norms[active_idx].unsqueeze(1)
    normed = active_flat / active_norms.clamp(min=1e-8)
    sim = torch.mm(normed, normed.t())
    # Zero diagonal
    sim.fill_diagonal_(0.0)
    return float(sim.max().item())


def _compute_per_class_flip_rates(curr, prev, K):
    """
    Per-class flip rate: for each class in prev, what fraction changed?
    Returns a dict {class_id: flip_rate} and the max per-class flip rate.
    """
    if prev is None:
        return {}, 1.0

    per_class = {}
    for k in range(K):
        mask = (prev == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        flipped = ((curr[mask] != k).sum().item())
        per_class[k] = flipped / n_k

    max_flip = max(per_class.values()) if per_class else 1.0
    return per_class, max_flip


# =========================================================================
# 10. UNIFIED TRAINING LOOP (v3 — comprehensive metrics)
# =========================================================================

def train_dino(
    model,
    dataloader,
    epochs=50,
    lr=None,
    weight_decay=None,
    device=None,
    outdir='./',
    save_every=5,
    seed=42,
    unfreeze_epoch=0,
    grad_clip=None,
    lr_warmup_epochs=None,
    # Spatial coherence support
    dataset=None,              # the Dataset object (for spatial patch)
    scan_shape=None,           # (rows, cols) of the scan grid
    # Center mask
    center_mask_radius=13,     # radius of center beam mask. 0 = disabled.
    center_crop_size=150,      # center crop size after rotation.
):
    """
    Unified training loop for both ResNet-CBAM and ViT-Tiny DINO models.

    New in v3:
        dataset:    pass your Dataset object to enable spatial coherence tracking.
                    If None, spatial metrics are skipped (backward compatible).
        scan_shape: (rows, cols) tuple — the 2D shape of the scan grid.
                    Required if dataset is provided.
        center_mask_radius: radius of center beam mask. Set to 0 to disable.
    """
    arch = model.architecture

    # --- Defaults per architecture ---
    if lr is None:
        lr = 3e-4 if arch == "resnet" else 5e-4
    if weight_decay is None:
        weight_decay = 1e-6 if arch == "resnet" else 0.05
    if grad_clip is None:
        grad_clip = 1.0 if arch == "resnet" else 3.0
    if lr_warmup_epochs is None:
        lr_warmup_epochs = 0 if arch == "resnet" else 5

    # --- Setup ---
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(outdir, exist_ok=True)
    model = model.to(device)

    # --- Optimizer ---
    if arch == "resnet":
        no_decay_params, decay_params = _is_bn_or_bias_param(model)
        optimizer = torch.optim.AdamW([
            {'params': no_decay_params, 'weight_decay': 0.0},
            {'params': decay_params, 'weight_decay': weight_decay}
        ], lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- LR Scheduler ---
    if lr_warmup_epochs > 0:
        scheduler = _get_cosine_schedule_with_warmup(optimizer, lr_warmup_epochs, epochs)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)

    # --- Augmentations ---
    student_aug, teacher_aug = get_augmentation_transforms(center_mask_radius=center_mask_radius, center_crop_size=center_crop_size)
    K = model._num_prototypes()

    # --- Validation anchor batch ---
    anchor_data = next(iter(dataloader))
    anchor_x = (anchor_data[0] if isinstance(anchor_data, (list, tuple)) else anchor_data).to(device).float()
    with torch.no_grad():
        anchor_x = teacher_aug(anchor_x)
    prev_assignments = None

    # Dynamic shape from augmented output
    _, C_in, H_in, W_in = anchor_x.shape
    running_mdp = torch.zeros(K, C_in, H_in, W_in).to(device)
    laplacian_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)

    # ================================================================
    # --- Spatial patch setup (center of scan grid) ---
    # ================================================================
    use_spatial = (dataset is not None and scan_shape is not None)
    spatial_patch_x = None
    spatial_patch_shape = None
    prev_spatial_assignments = None

    if use_spatial:
        Ny, Nx = scan_shape
        batch_size = dataloader.batch_size or 128

        # Pick a rectangular block from the center of the scan grid
        # that fits within one batch
        patch_area = min(batch_size, Ny * Nx)
        patch_h = int(math.sqrt(patch_area * Ny / max(Nx, 1)))
        patch_w = patch_area // max(patch_h, 1)
        patch_h = min(patch_h, Ny)
        patch_w = min(patch_w, Nx)

        # Center the patch
        y0 = max(0, (Ny - patch_h) // 2)
        x0 = max(0, (Nx - patch_w) // 2)

        # Convert 2D (y, x) to flat indices (row-major)
        spatial_indices = []
        for row in range(y0, y0 + patch_h):
            for col in range(x0, x0 + patch_w):
                flat_idx = row * Nx + col
                if flat_idx < len(dataset):
                    spatial_indices.append(flat_idx)

        # Load and pre-augment the spatial patch (once, cached)
        spatial_batch = torch.stack([
            (dataset[i][0] if isinstance(dataset[i], (list, tuple)) else dataset[i])
            for i in spatial_indices
        ]).to(device).float()
        # Ensure 4D: [B, C, H, W]
        if spatial_batch.dim() == 3:
            spatial_batch = spatial_batch.unsqueeze(1)

        with torch.no_grad():
            spatial_patch_x = teacher_aug(spatial_batch)

        actual_n = len(spatial_indices)
        # Recompute exact patch shape from actual indices loaded
        spatial_patch_shape = (patch_h, min(patch_w, actual_n // max(patch_h, 1)))
        # Trim to exact rectangle
        rect_n = spatial_patch_shape[0] * spatial_patch_shape[1]
        spatial_patch_x = spatial_patch_x[:rect_n]

        print(f"--> Spatial patch: {spatial_patch_shape[0]}x{spatial_patch_shape[1]} "
              f"= {rect_n} samples from center of ({Ny},{Nx}) grid")
    else:
        if dataset is not None and scan_shape is None:
            print("WARNING: dataset provided but scan_shape is None — "
                  "spatial metrics disabled. Pass scan_shape=(Ny, Nx).")

    # --- Trackers ---
    best_loss = float('inf')
    best_sharpness = 0.0
    best_effK = 0.0
    min_max_sim = 1.0
    best_balanced_score = 0.0
    best_conf_score = 0.0
    best_stable_K = 0.0
    best_auto_score = -float('inf')
    best_spatial_coherence = 0.0
    best_total_score = -float('inf')

    # [FIX 1] Freeze prototypes
    if unfreeze_epoch > 0:
        for p in model.prototypes.parameters():
            p.requires_grad = False
        print(f"--> Prototypes FROZEN until epoch {unfreeze_epoch}")
    prototypes_unfrozen = (unfreeze_epoch == 0)

    # --- CSV log (expanded columns) ---
    log_path = os.path.join(outdir, "training_log.csv")
    csv_columns = [
        "epoch", "avg_loss", "effK", "active_classes",
        "flip_rate", "max_class_flip",
        "sharpness", "max_sim", "mdp_max_xcorr",
        "avg_conf", "conf_median", "conf_p10", "conf_p90", "conf_bimodality",
        "intensity_corr",
        "spatial_coherence", "isolated_disagree", "boundary_frac", "spatial_flip",
        "auto_score", "total_score",
    ]

    with open(log_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)

        epoch_iter = range(epochs)
        if _HAS_TQDM:
            epoch_iter = _tqdm(epoch_iter, desc="Training", ncols=80,
                               leave=False, mininterval=5)

        for epoch in epoch_iter:

            if not prototypes_unfrozen and epoch >= unfreeze_epoch:
                for p in model.prototypes.parameters():
                    p.requires_grad = True
                prototypes_unfrozen = True
                print(f"--> Epoch {epoch}: Unfreezing Prototypes")

            model.train()
            total_loss = 0.0
            total_samples = 0
            sum_probs = None
            total_corr_weighted = 0.0
            total_conf_weighted = 0.0
            total_corr_batches = 0
            all_conf_values = []   # for confidence distribution

            temp_t = get_teacher_temp(epoch, epochs, model.T0, model.Tfin, model.warmup_frac)
            m = get_teacher_momentum(epoch, epochs, model.EMA0, model.EMAfin)

            for bi, batch in enumerate(dataloader):
                x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device, non_blocking=True).float()
                bs = x.size(0)

                with torch.no_grad():
                    xs = student_aug(x.clone())
                    xt = teacher_aug(x.clone())
                    batch_intensity = x.sum(dim=(1, 2, 3))

                optimizer.zero_grad(set_to_none=True)
                loss, probs_t, raw_t_logits = model(xs, xt, temp_student=0.1, temp_teacher=temp_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                model.update_teacher(momentum=m)
                model.sync_teacher_bn_buffers()
                model.update_center(raw_t_logits)

                with torch.no_grad():
                    total_loss += loss.item()
                    total_samples += bs
                    sum_probs = (probs_t.sum(dim=0).cpu() if sum_probs is None
                                 else sum_probs + probs_t.sum(dim=0).cpu())

                    batch_max_conf = probs_t.max(dim=-1).values
                    batch_conf = batch_max_conf.mean().item()
                    total_conf_weighted += batch_conf * bs

                    # Collect per-sample confidence for distribution analysis
                    all_conf_values.extend(batch_max_conf.cpu().numpy().tolist())

                    preds = probs_t.argmax(dim=-1).float()
                    if preds.var() > 1e-5 and batch_intensity.var() > 1e-5:
                        total_corr_weighted += torch.corrcoef(
                            torch.stack([batch_intensity, preds]))[0, 1].item() * bs
                        total_corr_batches += bs

                    conf_mask = (batch_max_conf > 0.6)
                    if conf_mask.any():
                        confident_xt = xt[conf_mask]
                        confident_preds = preds[conf_mask].long()
                        for k_idx in range(K):
                            mask_k = (confident_preds == k_idx)
                            if mask_k.any():
                                running_mdp[k_idx] = (0.9 * running_mdp[k_idx]
                                    + 0.1 * confident_xt[mask_k].mean(dim=0))

            scheduler.step()

            # ================================================================
            # END-OF-EPOCH VALIDATION — all metrics
            # ================================================================
            avg_loss = total_loss / len(dataloader)
            avg_corr = total_corr_weighted / max(total_corr_batches, 1)
            avg_conf = total_conf_weighted / max(total_samples, 1)

            # -- Confidence distribution --
            conf_median, conf_p10, conf_p90, conf_bimodality = \
                _compute_confidence_stats(all_conf_values)

            # -- Anchor batch: flip rate + per-class flip rates --
            with torch.no_grad():
                _, anchor_probs, _ = model(anchor_x, anchor_x, temp_teacher=temp_t)
                curr_assignments = anchor_probs.argmax(dim=-1)
                flip_rate = ((curr_assignments != prev_assignments).float().mean().item()
                             if prev_assignments is not None else 1.0)
                per_class_flips, max_class_flip = _compute_per_class_flip_rates(
                    curr_assignments, prev_assignments, K)
                prev_assignments = curr_assignments

            # -- Sharpness --
            center_h, center_w = H_in // 2, W_in // 2
            mdp_check = running_mdp.clone()
            pad = 5
            h_lo = max(center_h - pad, 0)
            h_hi = min(center_h + pad, H_in)
            w_lo = max(center_w - pad, 0)
            w_hi = min(center_w + pad, W_in)
            mdp_check[:, :, h_lo:h_hi, w_lo:w_hi] = 0
            sharpness_list = [
                F.conv2d(mdp_check[k_idx].unsqueeze(0), laplacian_kernel).var().item()
                for k_idx in range(K) if running_mdp[k_idx].sum() > 0
            ]
            avg_sharp = np.mean(sharpness_list) if sharpness_list else 0.0

            # -- Inter-prototype similarity --
            flat_protos = running_mdp.view(K, -1)
            norm_protos = F.normalize(flat_protos, p=2, dim=1, eps=1e-8)
            sim_matrix = torch.mm(norm_protos, norm_protos.t())
            sim_matrix.masked_fill_(torch.eye(K, device=device).bool(), 0)
            max_sim = sim_matrix.max().item()

            # -- MDP cross-correlation (duplicate cluster detection) --
            mdp_max_xcorr = _compute_mdp_cross_correlation(running_mdp, K)

            # -- Effective K --
            p_bar = (sum_probs / total_samples).clamp(1e-12, 1.0)
            effK = math.exp(-(p_bar * p_bar.log()).sum().item())
            active_classes = int((sum_probs > 0).sum().item())

            # -- Spatial coherence (from cached center patch) --
            sp_coherence = 0.0
            sp_isolated = 0.0
            sp_boundary = 0.0
            sp_flip = 1.0

            if use_spatial and spatial_patch_x is not None:
                with torch.no_grad():
                    _, sp_probs, _ = model(
                        spatial_patch_x, spatial_patch_x, temp_teacher=temp_t)
                    sp_preds = sp_probs.argmax(dim=-1).cpu().numpy()

                sp_h, sp_w = spatial_patch_shape
                sp_map = sp_preds[:sp_h * sp_w].reshape(sp_h, sp_w)
                sp_coherence, sp_isolated, sp_boundary = _compute_spatial_metrics(sp_map)

                # Spatial flip rate
                if prev_spatial_assignments is not None:
                    sp_flip = float((sp_preds != prev_spatial_assignments).mean())
                prev_spatial_assignments = sp_preds.copy()

            # -- Auto-score (original, backward compatible) --
            auto_score = _compute_auto_score(
                epoch, effK, K, flip_rate, avg_conf, avg_sharp, max_sim, avg_loss,
                min_epoch=max(10, unfreeze_epoch + 5),
            )

            # -- Total score (new, incorporates ALL metrics) --
            total_score = _compute_total_score(
                epoch, effK, K, flip_rate, max_class_flip,
                avg_conf, conf_bimodality,
                avg_sharp, max_sim, mdp_max_xcorr,
                sp_coherence, sp_isolated,
                avg_loss,
                min_epoch=max(10, unfreeze_epoch + 5),
                use_spatial=use_spatial,
            )

            # -- CSV row --
            writer.writerow([
                epoch + 1, f"{avg_loss:.4f}", f"{effK:.2f}", active_classes,
                f"{flip_rate:.4f}", f"{max_class_flip:.4f}",
                f"{avg_sharp:.6f}", f"{max_sim:.4f}", f"{mdp_max_xcorr:.4f}",
                f"{avg_conf:.4f}", f"{conf_median:.4f}", f"{conf_p10:.4f}",
                f"{conf_p90:.4f}", f"{conf_bimodality:.4f}",
                f"{avg_corr:.4f}",
                f"{sp_coherence:.4f}", f"{sp_isolated:.4f}",
                f"{sp_boundary:.4f}", f"{sp_flip:.4f}",
                f"{auto_score:.4f}", f"{total_score:.4f}",
            ])

            # -- No epoch print — tqdm handles progress --

            # ---- Checkpoint creation helper ----
            def create_ckpt():
                return {
                    'epoch': epoch + 1,
                    'architecture': arch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'teacher_temp': temp_t,
                    'center': model.center.detach().cpu(),
                    'metrics': {
                        'loss': avg_loss,
                        'eff_k': effK,
                        'active_classes': active_classes,
                        'sharpness': avg_sharp,
                        'flip_rate': flip_rate,
                        'max_class_flip': max_class_flip,
                        'max_sim': max_sim,
                        'mdp_max_xcorr': mdp_max_xcorr,
                        'avg_conf': avg_conf,
                        'conf_median': conf_median,
                        'conf_p10': conf_p10,
                        'conf_p90': conf_p90,
                        'conf_bimodality': conf_bimodality,
                        'intensity_corr': avg_corr,
                        'spatial_coherence': sp_coherence,
                        'isolated_disagree': sp_isolated,
                        'boundary_frac': sp_boundary,
                        'spatial_flip': sp_flip,
                        'auto_score': auto_score,
                        'total_score': total_score,
                    }
                }

            # ---- Saving logic ----
            torch.save(create_ckpt(), os.path.join(outdir, "latest.pth"))

            if (epoch + 1) % save_every == 0:
                torch.save(create_ckpt(), os.path.join(outdir, f"ckpt_ep{epoch+1}.pth"))

            min_save_epoch = max(10, unfreeze_epoch + 5)

            if avg_loss < best_loss and epoch >= min_save_epoch:
                best_loss = avg_loss
                torch.save(create_ckpt(), os.path.join(outdir, "best_loss.pth"))

            if epoch >= min_save_epoch and flip_rate < 0.30 and effK > 3.0:
                if avg_sharp > best_sharpness:
                    best_sharpness = avg_sharp
                    torch.save(create_ckpt(), os.path.join(outdir, "best_sharpness.pth"))

            if epoch >= min_save_epoch and effK > best_effK:
                best_effK = effK
                torch.save(create_ckpt(), os.path.join(outdir, "best_diversity.pth"))

            if epoch >= min_save_epoch and max_sim < min_max_sim and effK > 3.0:
                min_max_sim = max_sim
                torch.save(create_ckpt(), os.path.join(outdir, "best_distinct.pth"))

            curr_balanced = effK * (1.0 - max_sim)
            if epoch >= min_save_epoch and flip_rate < 0.30:
                if curr_balanced > best_balanced_score:
                    best_balanced_score = curr_balanced
                    torch.save(create_ckpt(), os.path.join(outdir, "best_balanced.pth"))

            curr_conf_score = avg_sharp * avg_conf
            if epoch >= min_save_epoch and flip_rate < 0.30 and effK > 3.0:
                if curr_conf_score > best_conf_score:
                    best_conf_score = curr_conf_score
                    torch.save(create_ckpt(), os.path.join(outdir, "best_high_conf.pth"))

            if epoch >= min_save_epoch and flip_rate < 0.10:
                if effK > best_stable_K:
                    best_stable_K = effK
                    torch.save(create_ckpt(), os.path.join(outdir, "best_stable_div.pth"))

            # -- Best spatial coherence --
            if (use_spatial and epoch >= min_save_epoch
                    and sp_coherence > best_spatial_coherence
                    and effK > 3.0 and flip_rate < 0.30):
                best_spatial_coherence = sp_coherence
                torch.save(create_ckpt(), os.path.join(outdir, "best_spatial.pth"))

            # -- Best total score --
            if total_score > best_total_score:
                best_total_score = total_score
                torch.save(create_ckpt(), os.path.join(outdir, "best_total.pth"))

            # -- Auto-select best --
            if auto_score > best_auto_score:
                best_auto_score = auto_score
                torch.save(create_ckpt(), os.path.join(outdir, "best_auto.pth"))

    return model


# =========================================================================
# 11. AUTO-SCORE (original, backward compatible)
# =========================================================================

def _compute_auto_score(epoch, effK, K, flip_rate, avg_conf, avg_sharp,
                        max_sim, avg_loss, min_epoch=10):
    """
    Composite quality score computed during training.

    v4 fixes based on empirical results across NaPHI/MgPHI/Tb:
      - Hard gate: avg_conf >= 0.5 (eliminates uniform-spread pathology)
      - Diversity uses bell curve peaking at 0.6-0.8×K (penalizes both
        collapse AND uniform spread near K)
    """
    if epoch < min_epoch:
        return 0.0
    if effK < 2.5:
        return 0.0
    if flip_rate > 0.40:
        return 0.0
    if avg_conf < 0.5:
        return 0.0

    # Bell-curve diversity: peaks at effK ≈ 0.7*K, penalizes both extremes
    # e.g. K=10: peak at 7, scores 0 at 2 and 10
    target_effK = 0.7 * K
    sigma_effK = 0.25 * K
    s_diversity = math.exp(-0.5 * ((effK - target_effK) / sigma_effK) ** 2)

    s_stability = max(1.0 - flip_rate, 0.0)
    s_confidence = min(avg_conf / 0.8, 1.0)
    s_distinct = max(1.0 - max_sim, 0.0)
    s_loss = max(1.0 - avg_loss / 5.0, 0.0)
    s_sharpness = 1.0 if avg_sharp > 1e-7 else 0.0

    return (0.25 * s_diversity + 0.25 * s_stability + 0.20 * s_confidence
            + 0.15 * s_distinct + 0.05 * s_loss + 0.10 * s_sharpness)


# =========================================================================
# 12. TOTAL SCORE (new — incorporates ALL metrics)
# =========================================================================

def _compute_total_score(
    epoch, effK, K, flip_rate, max_class_flip,
    avg_conf, conf_bimodality,
    avg_sharp, max_sim, mdp_max_xcorr,
    sp_coherence, sp_isolated,
    avg_loss,
    min_epoch=10,
    use_spatial=False,
):
    """
    Comprehensive quality score using ALL available metrics.

    v4 fixes:
      - Hard gate: avg_conf >= 0.5 (uniform-spread models always have <0.3)
      - Diversity uses bell curve peaking at 0.7×K (penalizes effK near K)
      - Confidence weight increased (the strongest discriminator empirically)
    """
    if epoch < min_epoch:
        return 0.0
    if effK < 2.5:
        return 0.0
    if flip_rate > 0.50:
        return 0.0
    if avg_conf < 0.5:
        return 0.0

    # Bell-curve diversity: peaks at effK ≈ 0.7*K
    target_effK = 0.7 * K
    sigma_effK = 0.25 * K
    s_diversity     = math.exp(-0.5 * ((effK - target_effK) / sigma_effK) ** 2)

    s_stability     = max(1.0 - flip_rate, 0.0)
    s_class_stab    = max(1.0 - max_class_flip, 0.0)
    s_confidence    = min(avg_conf / 0.8, 1.0)
    s_conf_dist     = min(conf_bimodality / 0.555, 1.0)
    s_sharpness     = 1.0 if avg_sharp > 1e-7 else 0.0
    s_distinct      = max(1.0 - max_sim, 0.0)
    s_no_duplicates = max(1.0 - mdp_max_xcorr, 0.0)
    s_loss          = max(1.0 - avg_loss / 5.0, 0.0)

    if use_spatial and sp_coherence > 0:
        s_spatial   = min(sp_coherence / 0.85, 1.0)
        s_low_noise = max(1.0 - sp_isolated, 0.0)

        score = (
            0.10 * s_diversity
            + 0.08 * s_stability
            + 0.06 * s_class_stab
            + 0.18 * s_confidence         # confidence is king
            + 0.04 * s_conf_dist
            + 0.06 * s_sharpness
            + 0.06 * s_distinct
            + 0.05 * s_no_duplicates
            + 0.02 * s_loss
            + 0.15 * s_spatial
            + 0.10 * s_low_noise
            + 0.10 * max(1.0 - sp_isolated * 3.0, 0.0)
        )
    else:
        score = (
            0.15 * s_diversity
            + 0.12 * s_stability
            + 0.08 * s_class_stab
            + 0.25 * s_confidence         # confidence is king
            + 0.05 * s_conf_dist
            + 0.08 * s_sharpness
            + 0.08 * s_distinct
            + 0.08 * s_no_duplicates
            + 0.03 * s_loss
            + 0.08 * s_sharpness
        )

    return max(score, 0.0)


# =========================================================================
# 13. CHECKPOINT LOADING (safe, architecture-aware)
# =========================================================================

def load_checkpoint(path, device=None, num_prototypes=10, **model_kwargs):
    """
    Load a checkpoint saved by train_dino().
    Auto-detects architecture. Returns (model, eval_temp, metrics).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    arch = ckpt.get("architecture", None)
    if arch is None:
        if any("blocks" in k for k in state_dict.keys()):
            arch = "vit"
        else:
            arch = "resnet"

    model = DINOModelSR(architecture=arch, num_prototypes=num_prototypes, **model_kwargs)
    model = model.to(device)

    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"  WARNING load_checkpoint: Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  WARNING load_checkpoint: Unexpected keys: {result.unexpected_keys}")

    model.eval()

    eval_temp = ckpt.get("teacher_temp", 0.06)
    if torch.is_tensor(eval_temp):
        eval_temp = eval_temp.item()

    metrics = ckpt.get("metrics", {})

    return model, eval_temp, metrics
