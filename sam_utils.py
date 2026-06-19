"""sam_utils.py -- Segment-Anything mask generation + geometric filtering
for 4D-STEM diffraction patterns.

Ported from
    https://github.com/DanielKhaykelson/NaPHI_structural-simulations_SAM
    (SAM/sam_image_utils.py)

Pipeline per pattern:
    raw 2D (1ch) -> cv2.cvtColor (3ch) -> gaussian(sigma) -> rescale_intensity
    -> rescale(downsample) -> SAM ViT-{b,l,h}.generate -> sort by area
    -> filter_masks (geometric criteria) -> list of dicts with
       {segmentation, area, bbox, angle, distance, midpoint}.

Designed so the GUI panel can call:
    proc = SamMaskProcessor(checkpoint=..., model_type="vit_h", device="cuda")
    masks = proc.run_one(class_average_2d, **filter_kwargs)
    proc.run_class(image_stack, save_dir, **filter_kwargs)

`segment_anything` is imported lazily so this module is importable on
machines without the package installed (the GUI panel surfaces a clean
"please install" message instead of crashing).
"""
from __future__ import annotations
import os
import math
import json
import time
from typing import Iterable

import numpy as np
import cv2
from skimage.filters import gaussian
from skimage.transform import rescale
from skimage import exposure

# Lazy import — see _ensure_sam().
_SAM_REGISTRY = None
_SamAutomaticMaskGenerator = None


def _ensure_sam():
    """Lazy-load segment_anything; raise a clear error if missing."""
    global _SAM_REGISTRY, _SamAutomaticMaskGenerator
    if _SAM_REGISTRY is not None:
        return
    try:
        from segment_anything import (sam_model_registry,
                                       SamAutomaticMaskGenerator)
    except ImportError as e:
        raise ImportError(
            "segment_anything is not installed.  Install with:\n"
            "    pip install git+https://github.com/facebookresearch/"
            "segment-anything.git"
        ) from e
    _SAM_REGISTRY = sam_model_registry
    _SamAutomaticMaskGenerator = SamAutomaticMaskGenerator


# ---------------------------------------------------------------------------
# Preprocessing (matches the original notebook exactly)
# ---------------------------------------------------------------------------

def preprocess_image(image: np.ndarray,
                       blur_sigma: float = 4.0,
                       rescale_lo: float = 0.0,
                       rescale_hi: float = 0.6,
                       downsample: float = 0.5) -> np.ndarray:
    """1-channel 2D pattern -> 3-channel preprocessed image ready for SAM.

    Parameters mirror the original SamMaskProcessor.preprocess_image,
    but every step is exposed as a kwarg so the GUI sliders can tune
    the pipeline without forking the function.
    """
    if image.ndim != 2:
        raise ValueError(f"preprocess_image expects 2D, got shape {image.shape}")
    # cv2 wants uint8 OR float32 in [0,1] for cvtColor. Normalise to [0,1] first.
    img = image.astype(np.float32)
    img_min, img_max = float(img.min()), float(img.max())
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    blurred = gaussian(img_3ch, sigma=float(blur_sigma), preserve_range=False)
    rescaled = exposure.rescale_intensity(
        blurred, in_range=(float(rescale_lo), float(rescale_hi)))
    if downsample != 1.0:
        rescaled = rescale(rescaled, float(downsample), channel_axis=2)
    return rescaled


# ---------------------------------------------------------------------------
# Geometric mask filtering (matches the original SamMaskProcessor.filter_masks)
# ---------------------------------------------------------------------------

def filter_masks(masks: list,
                  area_range=(50, 1000),
                  aspect_ratio_threshold: float = 1.2,
                  min_length: int = 20,
                  min_distance: float = 30,
                  max_distance: float = 130,
                  min_r2: float = 0.5,
                  skip_largest: bool = True,
                  candidate_slice=(1, 10)) -> list:
    """Keep masks that look like linear streaks at sensible distance from
    the central beam.

    The masks list must already be sorted by area descending.  The
    largest mask is typically the "no-foreground" background blob and
    is dropped via `skip_largest` (default True).  `candidate_slice`
    bounds how many of the next-biggest masks we even consider — keeps
    runtime bounded when SAM produces hundreds of small false-positive
    masks.

    Returns: list of mask dicts (each mutated to gain `angle`,
    `distance`, `midpoint` fields).
    """
    filtered = []
    if not masks:
        return filtered
    start = 1 if skip_largest else 0
    end = candidate_slice[1] if candidate_slice else len(masks)
    for mask in masks[start:end]:
        area = mask['area']
        if not (area_range[0] <= area <= area_range[1]):
            continue
        # Bounding box: cv2/SAM convention (x, y, w, h)
        bw, bh = mask['bbox'][2], mask['bbox'][3]
        if min(bw, bh) <= 0:
            continue
        aspect_ratio = max(bw, bh) / min(bw, bh)
        if aspect_ratio < aspect_ratio_threshold or max(bw, bh) < min_length:
            continue
        img = mask['segmentation'].astype(int)
        ys, xs = np.nonzero(img)
        if len(xs) < 2 or len(ys) < 2:
            continue
        try:
            line_fit = np.polyfit(xs, ys, 1)
            y_pred = np.poly1d(line_fit)(xs)
        except np.linalg.LinAlgError:
            continue
        ss_res = float(np.sum((ys - y_pred) ** 2))
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
        if ss_tot == 0:
            continue
        r2 = 1.0 - ss_res / ss_tot
        if r2 < min_r2:
            continue
        x_min, x_max = int(np.min(xs)), int(np.max(xs))
        mid_x = (x_min + x_max) / 2.0
        mid_y = float(np.poly1d(line_fit)(mid_x))
        H, W = img.shape
        xc, yc = W / 2.0, H / 2.0
        dx = mid_x - xc
        # Note: the original code uses `img.shape[1] - mid_y - yc` for
        # delta_y -- author's coordinate convention. Preserve verbatim.
        dy = float(W) - mid_y - yc
        distance = float(math.sqrt(dx * dx + dy * dy))
        if not (min_distance <= distance <= max_distance):
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        mask['angle'] = float(angle)
        mask['distance'] = distance
        mask['midpoint'] = (mid_x, mid_y)
        filtered.append(mask)
    return filtered


# ---------------------------------------------------------------------------
# SAM mask processor (per-image + per-stack)
# ---------------------------------------------------------------------------

class SamMaskProcessor:
    """Wraps SAM model loading + per-image mask generation + filtering.

    Heavy SAM checkpoint is loaded lazily on first .generate() call so
    that constructing the processor is cheap (e.g. for unit tests or
    GUI startup).
    """

    def __init__(self,
                  checkpoint_path: str,
                  model_type: str = "vit_h",
                  device: str = "cuda",
                  amg_kwargs: "dict | None" = None):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device
        # SamAutomaticMaskGenerator constructor kwargs.  None -> SAM defaults.
        self.amg_kwargs = dict(amg_kwargs) if amg_kwargs else {}
        self._sam = None
        self._mask_generator = None

    # ----- model loading (lazy) ------------------------------------
    def _load(self):
        if self._mask_generator is not None:
            return
        _ensure_sam()
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"SAM checkpoint not found: {self.checkpoint_path}\n"
                f"Download from https://github.com/facebookresearch/"
                f"segment-anything (sam_{self.model_type}_*.pth)")
        self._sam = _SAM_REGISTRY[self.model_type](
            checkpoint=self.checkpoint_path)
        self._sam.to(self.device)
        self._mask_generator = _SamAutomaticMaskGenerator(
            self._sam, **self.amg_kwargs)

    # ----- single-image mask generation ----------------------------
    def generate(self, preprocessed_3ch: np.ndarray) -> list:
        """Run SAM on a 3-channel preprocessed image; return masks
        sorted by area (descending)."""
        self._load()
        masks = self._mask_generator.generate(preprocessed_3ch)
        return sorted(masks, key=lambda x: x['area'], reverse=True)

    def run_one(self,
                  raw_2d: np.ndarray,
                  *,
                  blur_sigma: float = 4.0,
                  rescale_lo: float = 0.0,
                  rescale_hi: float = 0.6,
                  downsample: float = 0.5,
                  filter_kwargs: "dict | None" = None) -> tuple:
        """End-to-end: raw 2D pattern -> preprocessed -> SAM ->
        filtered masks.  Returns (preprocessed_image, all_masks,
        filtered_masks)."""
        prep = preprocess_image(
            raw_2d,
            blur_sigma=blur_sigma,
            rescale_lo=rescale_lo,
            rescale_hi=rescale_hi,
            downsample=downsample,
        )
        all_masks = self.generate(prep)
        flt = filter_masks(all_masks, **(filter_kwargs or {}))
        return prep, all_masks, flt

    # ----- per-class batch run -------------------------------------
    def run_stack(self,
                    image_stack: np.ndarray,
                    *,
                    blur_sigma: float = 4.0,
                    rescale_lo: float = 0.0,
                    rescale_hi: float = 0.6,
                    downsample: float = 0.5,
                    filter_kwargs: "dict | None" = None,
                    progress_cb=None) -> list:
        """Iterate over an (N, H, W) stack; return list-of-list of
        filtered mask dicts.  `progress_cb(i, N)` is called every
        iteration if provided (e.g. for tqdm or subprocess heartbeat)."""
        out = []
        N = len(image_stack)
        for i, img in enumerate(image_stack):
            _, _, flt = self.run_one(
                img,
                blur_sigma=blur_sigma,
                rescale_lo=rescale_lo,
                rescale_hi=rescale_hi,
                downsample=downsample,
                filter_kwargs=filter_kwargs,
            )
            out.append(flt)
            if progress_cb is not None:
                try: progress_cb(i + 1, N)
                except Exception: pass
        return out


# ---------------------------------------------------------------------------
# Helpers for downstream analysis
# ---------------------------------------------------------------------------

def extract_min_angle(filtered_masks_per_pattern: Iterable) -> np.ndarray:
    """For each pattern, return the MIN angle of its filtered masks
    (or NaN if none).  This is what the original notebook used to
    build the (Ny, Nx) angle map shown via imshow(cmap='hsv').
    """
    n = len(filtered_masks_per_pattern)
    arr = np.full(n, np.nan, dtype=np.float32)
    for i, mlist in enumerate(filtered_masks_per_pattern):
        if not mlist:
            continue
        angs = [m['angle'] for m in mlist if 'angle' in m]
        if angs:
            arr[i] = float(np.min(angs))
    return arr


def masks_to_rle_list(filtered_masks_per_pattern: Iterable) -> list:
    """Compact RLE-style encoding so we don't store full binary masks
    on disk.  Each mask is replaced by (H, W, indices_of_nonzero) so
    we can still reconstruct.  For a typical 12k-pattern run this
    keeps disk footprint at ~10s of MB instead of 10s of GB.
    """
    out = []
    for mlist in filtered_masks_per_pattern:
        compact = []
        for m in mlist:
            seg = m['segmentation']
            ys, xs = np.nonzero(seg)
            compact.append({
                'shape': list(seg.shape),
                'ys': ys.astype(np.int32),
                'xs': xs.astype(np.int32),
                'area': int(m['area']),
                'angle': float(m.get('angle', np.nan)),
                'distance': float(m.get('distance', np.nan)),
                'midpoint': list(m.get('midpoint', (0.0, 0.0))),
                'bbox': list(m.get('bbox', [0, 0, 0, 0])),
            })
        out.append(compact)
    return out
