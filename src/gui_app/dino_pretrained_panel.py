"""dino_pretrained_panel.py -- DINO + cluster benchmark.

Mirrors nmf_panel.py but, instead of factorising the polar matrix,
runs a pretrained DINO (v1 / v2 / v3) on every diffraction pattern,
collects the CLS-token embeddings, and feeds them through the same
clustering pipeline (K-means / Aglo / HDBSCAN / FCM, auto-K via
silhouette).

Models are pulled lazily via `torch.hub.load(...)`; the first run
downloads weights (~100 MB – 4 GB depending on size). 4DSTEM frames
are vmax-normalised, resized to the model's expected input size,
replicated 1 → 3 channels, and ImageNet-normalised before forward.

Input matrix to clustering:
    W = embeddings, shape (N, D)
    optional PCA(d) to reduce dimensionality before clustering
    (helps HDBSCAN density estimates).
"""
from __future__ import annotations
import os, json, time, threading

import numpy as np
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                 NavigationToolbar2Tk)
from matplotlib.colors import ListedColormap


# ---------------------------------------------------------------------------
# Pretrained DINO catalogue.  Each entry is what torch.hub needs +
# embedding/input metadata.  The reference column is the citation that
# gets shown in the dropdown label, mirroring the NMF tab convention.
DINO_MODELS = {
    "DINOv1 ViT-S/16  (Caron et al. 2021)":
        dict(repo="facebookresearch/dino:main",
             name="dino_vits16",
             hf_id="facebook/dino-vits16",
             input_size=224),
    "DINOv1 ViT-B/16  (Caron et al. 2021)":
        dict(repo="facebookresearch/dino:main",
             name="dino_vitb16",
             hf_id="facebook/dino-vitb16",
             input_size=224),
    "DINOv2 ViT-S/14  (Oquab et al. 2023)":
        dict(repo="facebookresearch/dinov2",
             name="dinov2_vits14",
             hf_id="facebook/dinov2-small",
             input_size=224),
    "DINOv2 ViT-B/14  (Oquab et al. 2023)":
        dict(repo="facebookresearch/dinov2",
             name="dinov2_vitb14",
             hf_id="facebook/dinov2-base",
             input_size=224),
    "DINOv2 ViT-L/14  (Oquab et al. 2023)":
        dict(repo="facebookresearch/dinov2",
             name="dinov2_vitl14",
             hf_id="facebook/dinov2-large",
             input_size=224),
    "DINOv2 ViT-g/14  (Oquab et al. 2023)":
        dict(repo="facebookresearch/dinov2",
             name="dinov2_vitg14",
             hf_id="facebook/dinov2-giant",
             input_size=224),
    "DINOv3 ViT-S/16  (Meta 2025)":
        dict(repo="facebookresearch/dinov3",
             name="dinov3_vits16",
             hf_id=None,                 # no HF mirror at time of writing
             input_size=224),
    "DINOv3 ViT-B/16  (Meta 2025)":
        dict(repo="facebookresearch/dinov3",
             name="dinov3_vitb16",
             hf_id=None,
             input_size=224),
}

CLUSTER_METHODS = ["K-means", "Aglo", "HDBSCAN", "FCM"]
AGLO_DISTANCES  = ["euclidean", "cosine"]


def _section(parent, title):
    f = ctk.CTkFrame(parent, fg_color="transparent")
    f.pack(fill="x", padx=2, pady=(8, 0))
    ctk.CTkLabel(f, text=title, font=("Segoe UI", 11, "bold")).pack(
        anchor="w")
    sep = ctk.CTkFrame(parent, height=2, fg_color=("#cccccc", "#444444"))
    sep.pack(fill="x", padx=2, pady=(0, 4))
    return parent


def _safe_name(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "+", "/"):
            out.append("_")
    return "".join(out).strip("_") or "model"


# ---------------------------------------------------------------------------
def _load_dino_model(model_cfg: dict, source: str, device,
                        pth_path: str | None = None):
    """Load a pretrained DINO from `torch.hub` or HuggingFace.

    source ∈ {"hub", "hf", "auto"}.  "auto" tries hub first, falls back
    to HF if the hub fetch fails (and an `hf_id` is registered).

    If `pth_path` is non-empty and the file exists, the loaded model's
    state_dict is **overwritten** by `torch.load(pth_path)` after the
    hub/HF download — useful for custom DINO weights without making
    the user run torch.hub at all (e.g. when offline).

    Returns (model, source_used).
    """
    import torch
    last_err = None
    loaded_model = None
    used_src = None
    if source in ("hub", "auto"):
        try:
            try:
                loaded_model = torch.hub.load(model_cfg["repo"],
                                          model_cfg["name"],
                                          trust_repo=True)
            except TypeError:
                # Old torch lacks `trust_repo`.
                loaded_model = torch.hub.load(model_cfg["repo"],
                                          model_cfg["name"])
            used_src = "hub"
        except Exception as e:
            last_err = e
            print(f"[dino] torch.hub load failed: {e!r}", flush=True)
            if source == "hub":
                raise
    if loaded_model is None and source in ("hf", "auto") \
            and model_cfg.get("hf_id"):
        try:
            from transformers import AutoModel
            loaded_model = AutoModel.from_pretrained(model_cfg["hf_id"])
            used_src = "hf"
        except Exception as e:
            print(f"[dino] HuggingFace load failed: {e!r}", flush=True)
            last_err = e
    if loaded_model is None:
        if last_err is not None:
            raise RuntimeError(
                f"could not load {model_cfg['name']!r} from any source: "
                f"{last_err!r}")
        raise RuntimeError(
            f"no source available for {model_cfg['name']!r} "
            f"(no hf_id registered)")

    # Optional custom .pth override: load state_dict on top of the
    # already-built architecture.  Tolerant of common checkpoint
    # wrapper layouts (DINO uses {'student':..., 'teacher':...}; some
    # exports wrap as {'model': ...}; many just dump a state_dict).
    if pth_path:
        try:
            sd = torch.load(pth_path, map_location=device)
            for wrap_key in ("teacher", "student", "model",
                                "state_dict"):
                if isinstance(sd, dict) and wrap_key in sd \
                        and isinstance(sd[wrap_key], dict):
                    sd = sd[wrap_key]
                    break
            # Strip common prefixes.
            sd = {k.replace("backbone.", "")
                      .replace("module.", "")
                  : v for k, v in sd.items()}
            missing, unexpected = loaded_model.load_state_dict(
                sd, strict=False)
            print(f"[dino] loaded .pth {pth_path}: "
                  f"missing={len(missing)} unexpected={len(unexpected)}",
                  flush=True)
            used_src = (used_src or "hub") + "+pth"
        except Exception as e:
            print(f"[dino] .pth override failed: {e!r}", flush=True)
            raise RuntimeError(
                f"could not load custom weights from {pth_path}: "
                f"{e!r}")

    return loaded_model.eval().to(device), used_src


def _forward_cls(model, x, source: str):
    """Get the CLS-token embedding regardless of model interface.

    HF `AutoModel`     → `BaseModelOutput.last_hidden_state[:, 0, :]`.
    DINOv2 hub         → `forward_features(x)['x_norm_clstoken']`.
    DINOv1/v3 hub      → `model(x)` returns the CLS embedding directly,
                         or `(B, T, D)` patch tokens (CLS at [:, 0]).
    """
    if source == "hf":
        out = model(x)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0, :]
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        # Some HF wrappers return a tuple.
        if isinstance(out, (tuple, list)):
            return out[0][:, 0, :]
        return out
    # source == "hub"
    ff = getattr(model, "forward_features", None)
    if callable(ff):
        try:
            o = ff(x)
            if isinstance(o, dict):
                cls = (o.get("x_norm_clstoken")
                         or o.get("x_clstoken")
                         or o.get("cls_token"))
                if cls is not None:
                    return cls
            if hasattr(o, "ndim") and o.ndim == 3:
                return o[:, 0]
            if hasattr(o, "ndim") and o.ndim == 2:
                return o
        except Exception:
            pass
    o = model(x)
    if hasattr(o, "ndim") and o.ndim == 3:
        return o[:, 0]
    return o


def _upsample_heatmap(heat: np.ndarray, target_hw: tuple) -> np.ndarray:
    """Bilinearly upsample a (p, p) attention map to (H, W)."""
    try:
        import cv2
        return cv2.resize(heat.astype(np.float32),
                            (int(target_hw[1]), int(target_hw[0])),
                            interpolation=cv2.INTER_LINEAR)
    except Exception:
        # scipy fallback (no cv2): use zoom.
        from scipy.ndimage import zoom
        zy = target_hw[0] / heat.shape[0]
        zx = target_hw[1] / heat.shape[1]
        return zoom(heat.astype(np.float32), (zy, zx), order=1)


def _extract_attention(model, x, source: str):
    """Per-pattern CLS→patches attention from the last layer.

    Returns a tensor shape (B, H_p, W_p), averaged over attention heads
    and normalised per pattern to [0, 1].  Raises NotImplementedError if
    the model doesn't expose attention.
    """
    import torch
    attn = None
    if source == "hf":
        out = model(x, output_attentions=True)
        if hasattr(out, "attentions") and out.attentions:
            attn = out.attentions[-1]            # (B, h, T, T)
    else:  # hub
        for meth in ("get_last_selfattention",
                       "get_last_self_attention"):
            f = getattr(model, meth, None)
            if callable(f):
                try:
                    attn = f(x)
                    break
                except Exception:
                    attn = None
    if attn is None:
        raise NotImplementedError(
            "this model / source doesn't expose attention weights. "
            "Try 'hf' source (HuggingFace) for DINOv2; DINOv1 hub "
            "exposes `get_last_selfattention` natively.")
    cls_attn = attn[:, :, 0, 1:]                 # (B, h, N)
    cls_attn = cls_attn.mean(dim=1)              # avg heads → (B, N)
    N = cls_attn.shape[-1]
    p = int(round(N ** 0.5))
    if p * p != N:
        raise RuntimeError(
            f"non-square patch grid: N={N} (expected p² patches)")
    cls_attn = cls_attn.reshape(-1, p, p)
    # Normalize per pattern to [0, 1] for display.
    flat = cls_attn.reshape(cls_attn.shape[0], -1)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    flat = (flat - mn) / (mx - mn).clamp_min(1e-12)
    return flat.reshape(-1, p, p)


def compute_embeddings(sample_key: str, model_cfg: dict,
                        vmax: float, batch: int = 64,
                        progress_cb=None, stop_check=None,
                        source: str = "auto",
                        pth_path: str | None = None
                        ) -> tuple[np.ndarray, "torch.nn.Module", str]:
    """Forward every pattern through a pretrained DINO and stack
    CLS-token embeddings into a (N, D) numpy array.

    Returns (embeds, model, source_used) — the model is kept so the
    caller can re-use it for attention extraction on selected patterns.
    """
    import torch
    import torch.nn.functional as F
    from data import SAMPLES, LoadPRZ
    cfg = SAMPLES[sample_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = LoadPRZ(cfg["path"], resize=192, vmax=float(vmax))
    N = len(ds)
    sz = int(model_cfg["input_size"])
    model, src_used = _load_dino_model(model_cfg, source, device,
                                              pth_path=pth_path)

    # ImageNet normalisation (the pretraining convention for all DINOs).
    mean = torch.tensor([0.485, 0.456, 0.406], device=device
                          ).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device
                          ).view(1, 3, 1, 1)

    embeds: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, N, batch):
            if stop_check is not None and stop_check():
                raise RuntimeError("stop requested")
            j = min(i + batch, N)
            xs = torch.stack([ds[k] for k in range(i, j)]
                              ).to(device).float()       # (b, 1, h, w)
            # Replicate channel 1 → 3 and resize to model input.
            xs = xs.repeat(1, 3, 1, 1)
            if xs.shape[-1] != sz or xs.shape[-2] != sz:
                xs = F.interpolate(xs, size=(sz, sz),
                                     mode="bilinear",
                                     align_corners=False)
            xs = (xs - mean) / std
            cls = _forward_cls(model, xs, src_used)
            embeds.append(cls.detach().cpu().numpy().astype(np.float32))
            if progress_cb is not None and (i // batch) % 2 == 0:
                progress_cb(j, N, "DINO inference")
    if progress_cb is not None:
        progress_cb(N, N, "DINO inference")
    return np.concatenate(embeds, axis=0), model, src_used


# ---------------------------------------------------------------------------
class DINOClusterPanel(ctk.CTkFrame):

    def __init__(self, master, app=None):
        super().__init__(master)
        self.app = app
        self.outdir = None
        self.sample = None
        self._scan_shape = None
        self._last = None
        self._compute_running = False
        self._stop_requested = False
        self._compute_progress = ""
        self._lock = threading.Lock()
        self._thread = None
        self._build()

    # ----- run linkage --------------------------------------------------
    def link_run(self, outdir, sample):
        self.outdir = outdir
        self.sample = sample
        try:
            self._vars["sample"].set(sample)
        except Exception:
            pass
        self._refresh_scan_shape()
        if self._vars["use_sample_vmax"].get():
            self._snap_vmax_to_sample()
        self._info_lbl.configure(
            text=f"sample: {sample}   scan = {self._scan_shape}   "
                  f"vmax = {self._vars['vmax'].get():g}")
        self._last = None
        self._render_idle()

    def on_runtime_sample_added(self, key):
        # A cube was loaded in the Data tab -> make it THIS panel's dataset.
        try:
            self._vars["sample"].set(key)
            self._on_sample_change()
        except Exception:
            pass

    def _sync_from_pre(self):
        """Adopt the cube currently loaded in the Data tab (app.pre)."""
        pre = getattr(self.app, "pre", None)
        try:
            k = pre.get_sample_key() if pre is not None else None
        except Exception:
            k = None
        if k and k != self.sample:
            self._vars["sample"].set(k)
            self._on_sample_change()
        return self.sample

    def _refresh_scan_shape(self):
        try:
            from data import SAMPLES
            cfg = SAMPLES.get(self.sample) or {}
            self._scan_shape = (cfg.get("scan_shape")
                                  or cfg.get("scan_size") or None)
        except Exception:
            self._scan_shape = None

    def _snap_vmax_to_sample(self):
        try:
            from data import SAMPLES
            cfg = SAMPLES.get(self.sample) or {}
            self._vars["vmax"].set(float(cfg.get("vmax", 2.0)))
        except Exception:
            pass

    # ----- UI ----------------------------------------------------------
    def _build(self):
        self._vars = {
            "sample":       ctk.StringVar(value=""),
            "vmax":         ctk.DoubleVar(value=2.0),
            "use_sample_vmax": ctk.BooleanVar(value=True),
            "model":        ctk.StringVar(
                value=next(iter(DINO_MODELS))),
            "source":       ctk.StringVar(value="auto"),
            "pth_path":     ctk.StringVar(value=""),
            "use_pca":      ctk.BooleanVar(value=False),
            "pca_dim":      ctk.IntVar(value=32),
            "use_kmeans":   ctk.BooleanVar(value=True),
            "use_aglo":     ctk.BooleanVar(value=False),
            "use_hdbscan":  ctk.BooleanVar(value=False),
            "use_fcm":      ctk.BooleanVar(value=False),
            "fcm_m":        ctk.DoubleVar(value=2.0),
            "aglo_dist":    ctk.StringVar(value="cosine"),
            "K":            ctk.IntVar(value=6),
            "auto_K":       ctk.BooleanVar(value=True),
            "min_cluster_size": ctk.IntVar(value=30),
            "batch":        ctk.IntVar(value=64),
        }

        # ---- top bar ----
        top = ctk.CTkFrame(self)
        top.pack(side="top", fill="x", padx=6, pady=6)
        try:
            from data import SAMPLES
            sample_values = sorted(SAMPLES.keys())
        except Exception:
            sample_values = [""]
        # Dataset follows the Data tab (top of the app) — no per-panel sample
        # dropdown / loader.
        ctk.CTkLabel(top, text="dataset:").pack(side="left", padx=(8, 4))
        self._ds_lbl = ctk.CTkLabel(top, text="(load a cube in the Data tab)",
                                     font=("Consolas", 10, "bold"))
        self._ds_lbl.pack(side="left", padx=4)
        self._info_lbl = ctk.CTkLabel(top, text="(no sample yet)",
                                        font=("Consolas", 10))
        self._info_lbl.pack(side="left", padx=8)
        self._run_btn = ctk.CTkButton(top, text="Run",
                                         width=100,
                                         fg_color=("#2D7A2D", "#1F7A1F"),
                                         command=self._kickoff_run)
        self._run_btn.pack(side="right", padx=4)
        self._stop_btn = ctk.CTkButton(top, text="Stop",
                                          width=80,
                                          fg_color=("#a23030", "#7a1f1f"),
                                          state="disabled",
                                          command=self._on_stop_clicked)
        self._stop_btn.pack(side="right", padx=4)

        # ---- body ----
        body = ctk.CTkFrame(self)
        body.pack(side="top", fill="both", expand=True, padx=6, pady=4)
        sb = ctk.CTkScrollableFrame(body, width=320)
        sb.pack(side="left", fill="y")

        _section(sb, "Pre-process scaling (DINO-style)")
        v_row = ctk.CTkFrame(sb, fg_color="transparent")
        v_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(v_row, text="vmax:", width=60).pack(side="left")
        ctk.CTkEntry(v_row, textvariable=self._vars["vmax"],
                       width=70).pack(side="left", padx=2)
        ctk.CTkCheckBox(v_row, text="use sample default",
                          variable=self._vars["use_sample_vmax"],
                          command=self._on_use_sample_vmax
                          ).pack(side="left", padx=8)

        _section(sb, "Pretrained DINO")
        ctk.CTkOptionMenu(sb, variable=self._vars["model"],
                            values=list(DINO_MODELS.keys()),
                            width=300
                            ).pack(anchor="w", padx=8, pady=2)
        ctk.CTkLabel(sb,
            text="(weights download lazily on first run)",
            font=("Segoe UI", 9), text_color=("#666", "#aaa")
            ).pack(anchor="w", padx=8, pady=(0, 2))
        s_row = ctk.CTkFrame(sb, fg_color="transparent")
        s_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(s_row, text="source:", width=60).pack(side="left")
        ctk.CTkOptionMenu(s_row, variable=self._vars["source"],
                            values=["auto", "hub", "hf"], width=120
                            ).pack(side="left", padx=4)
        ctk.CTkLabel(s_row,
            text="(auto: hub → HuggingFace fallback)",
            font=("Segoe UI", 9), text_color=("#666", "#aaa")
            ).pack(side="left", padx=4)
        # Optional: load custom .pth weights (overrides hub/HF download).
        pth_row = ctk.CTkFrame(sb, fg_color="transparent")
        pth_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(pth_row, text=".pth:", width=60).pack(side="left")
        ctk.CTkEntry(pth_row, textvariable=self._vars["pth_path"],
                       width=170).pack(side="left", padx=2)
        ctk.CTkButton(pth_row, text="Browse",
                       width=70,
                       command=self._browse_pth
                       ).pack(side="left", padx=2)
        ctk.CTkLabel(sb,
            text="(blank = use hub/HF download. Set a path to load "
                  "custom DINO weights — architecture still comes from "
                  "the model dropdown.)",
            font=("Segoe UI", 9), text_color=("#666", "#aaa"),
            wraplength=300, justify="left"
            ).pack(anchor="w", padx=8, pady=(0, 2))

        bb_row = ctk.CTkFrame(sb, fg_color="transparent")
        bb_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(bb_row, text="batch:", width=60).pack(side="left")
        ctk.CTkEntry(bb_row, textvariable=self._vars["batch"],
                       width=60).pack(side="left", padx=4)

        _section(sb, "Embedding pre-cluster reduction")
        p_row = ctk.CTkFrame(sb, fg_color="transparent")
        p_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkCheckBox(p_row, text="PCA before clustering",
                          variable=self._vars["use_pca"]
                          ).pack(side="left", padx=8)
        ctk.CTkLabel(p_row, text="dim:").pack(side="left", padx=(4, 2))
        ctk.CTkEntry(p_row, textvariable=self._vars["pca_dim"],
                       width=60).pack(side="left", padx=2)

        _section(sb, "Clustering methods")
        ctk.CTkCheckBox(sb, text="K-means  (default)",
                          variable=self._vars["use_kmeans"]
                          ).pack(anchor="w", padx=8, pady=1)
        a_row = ctk.CTkFrame(sb, fg_color="transparent")
        a_row.pack(fill="x", padx=4, pady=1)
        ctk.CTkCheckBox(a_row, text="Aglo",
                          variable=self._vars["use_aglo"]
                          ).pack(side="left", padx=8)
        ctk.CTkLabel(a_row, text="distance:").pack(side="left",
                                                       padx=(4, 2))
        ctk.CTkOptionMenu(a_row, variable=self._vars["aglo_dist"],
                            values=AGLO_DISTANCES, width=110
                            ).pack(side="left", padx=2)
        h_row = ctk.CTkFrame(sb, fg_color="transparent")
        h_row.pack(fill="x", padx=4, pady=1)
        ctk.CTkCheckBox(h_row, text="HDBSCAN",
                          variable=self._vars["use_hdbscan"]
                          ).pack(side="left", padx=8)
        ctk.CTkLabel(h_row, text="min size:").pack(side="left",
                                                       padx=(4, 2))
        ctk.CTkEntry(h_row, textvariable=self._vars["min_cluster_size"],
                       width=60).pack(side="left", padx=2)
        f_row = ctk.CTkFrame(sb, fg_color="transparent")
        f_row.pack(fill="x", padx=4, pady=1)
        ctk.CTkCheckBox(f_row, text="FCM  (Bezdek 1981)",
                          variable=self._vars["use_fcm"]
                          ).pack(side="left", padx=8)
        ctk.CTkLabel(f_row, text="m:").pack(side="left", padx=(4, 2))
        ctk.CTkEntry(f_row, textvariable=self._vars["fcm_m"],
                       width=60).pack(side="left", padx=2)

        _section(sb, "K  (ignored by HDBSCAN)")
        k_row = ctk.CTkFrame(sb, fg_color="transparent")
        k_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(k_row, text="K:", width=80).pack(side="left")
        ctk.CTkEntry(k_row, textvariable=self._vars["K"], width=60
                       ).pack(side="left", padx=4)
        ctk.CTkCheckBox(k_row, text="auto (silhouette)",
                          variable=self._vars["auto_K"]
                          ).pack(side="left", padx=8)

        ctk.CTkButton(sb, text="Show UMAP  (popup)",
                       command=self._show_umap_popup
                       ).pack(fill="x", padx=8, pady=(8, 2))
        ctk.CTkButton(sb, text="Class attentions  (grid)",
                       command=self._show_class_attentions
                       ).pack(fill="x", padx=8, pady=(2, 2))
        ctk.CTkLabel(sb,
            text="inline class map: left-click → single-pattern attention,"
                  " right-click → grain attention.\n"
                  "Big viewer: left → pattern, right → cluster-grain avg,"
                  " shift+right → stack.",
            font=("Segoe UI", 9), justify="left",
            text_color=("#666", "#aaa"), wraplength=300
            ).pack(anchor="w", padx=8, pady=(2, 4))
        ctk.CTkButton(sb, text="Open interactive map…",
                       fg_color=("#4D6FB0", "#3A5380"),
                       command=self._open_interactive_map
                       ).pack(fill="x", padx=8, pady=(2, 2))
        ctk.CTkButton(sb, text="Save snapshot",
                       command=self._save_snapshot
                       ).pack(fill="x", padx=8, pady=(2, 4))

        self._status_lbl = ctk.CTkLabel(sb,
            text="", font=("Consolas", 9), justify="left",
            text_color=("#444", "#aaa"), wraplength=300)
        self._status_lbl.pack(anchor="w", padx=8, pady=(8, 4))

        # ---- canvas ----
        canv = ctk.CTkFrame(body)
        canv.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self._fig = Figure(figsize=(13, 8))
        self._canvas = FigureCanvasTkAgg(self._fig, master=canv)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self._canvas, canv)

        # Bind to whatever is loaded in the Data tab (no default selection).
        self.after(200, self._sync_from_pre)
        # Auto-link to whatever Post-hoc / NMF have already loaded.
        try:
            self.after(150, self._try_auto_link)
        except Exception:
            pass
        self._render_idle()

    def _on_sample_change(self):
        s = self._vars["sample"].get()
        self.sample = s if s else None
        try:
            self._ds_lbl.configure(
                text=self.sample or "(load a cube in the Data tab)")
        except Exception:
            pass
        self._refresh_scan_shape()
        if self._vars["use_sample_vmax"].get():
            self._snap_vmax_to_sample()
        self._info_lbl.configure(
            text=f"sample: {self.sample}   scan = {self._scan_shape}  "
                  f"vmax = {self._vars['vmax'].get():g}")

    def _on_use_sample_vmax(self):
        if self._vars["use_sample_vmax"].get():
            self._snap_vmax_to_sample()

    def _try_auto_link(self):
        ph = getattr(self.app, "posthoc", None) if self.app else None
        if ph is None:
            return
        s = getattr(ph, "sample", None)
        out = getattr(ph, "outdir", None)
        if not s:
            return
        try:
            from data import SAMPLES
            if s not in SAMPLES:
                return
        except Exception:
            return
        self.outdir = out; self.sample = s
        self._vars["sample"].set(s); self._refresh_scan_shape()
        if self._vars["use_sample_vmax"].get():
            self._snap_vmax_to_sample()
        self._info_lbl.configure(
            text=f"auto-linked: {s}   scan = {self._scan_shape}")

    def _browse_pth(self):
        p = filedialog.askopenfilename(
            title="Pick a DINO .pth checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pth *.pt"),
                          ("All files", "*.*")])
        if p:
            self._vars["pth_path"].set(p)

    def _load_cube_from_disk(self):
        """Pick a .prz / .npy / .h5 cube from disk (master files
        supported via Dectris external-link stitching) and register
        it as a runtime sample. Mirrors the NMF tab's loader."""
        import re as _re
        p = filedialog.askopenfilename(
            title="Pick a cube  (.prz / .npz / .npy / .h5)",
            filetypes=[("Cube files",
                          "*.prz *.npz *.npy *.h5 *.hdf5"),
                          ("PRZ / NPZ", "*.prz *.npz"),
                          ("NPY", "*.npy"),
                          ("HDF5", "*.h5 *.hdf5"),
                          ("All files", "*.*")])
        if not p:
            return
        # Dectris fragment swap.
        bn = os.path.basename(p)
        m = _re.match(r"^(.*)_data_(\d+)\.h5$", bn)
        if m:
            master = os.path.join(os.path.dirname(p),
                                     m.group(1) + "_master.h5")
            if os.path.exists(master):
                if messagebox.askyesno("Dectris fragment",
                    f"You picked {bn}\n\nLoad the master\n  "
                    f"{os.path.basename(master)}\ninstead? (recommended)"):
                    p = master
        try:
            v = float(self._vars["vmax"].get())
        except Exception:
            v = 2.0
        scan_shape = None
        if p.lower().endswith((".h5", ".hdf5")):
            try:
                import h5py
                from data import (_h5_find_data_path,
                                      _h5_infer_scan_shape,
                                      _h5_dectris_external_data)
                with h5py.File(p, "r") as fh:
                    try:
                        dpath, ndim = _h5_find_data_path(fh)
                        s = tuple(fh[dpath].shape)
                    except ValueError:
                        pairs = _h5_dectris_external_data(fh, p)
                        if not pairs:
                            raise
                        total = 0; H_ = W_ = None
                        for fp, dp in pairs:
                            with h5py.File(fp, "r") as gf:
                                sh = tuple(gf[dp].shape)
                                if len(sh) != 3: continue
                                total += sh[0]
                                H_, W_ = sh[1], sh[2]
                        s = (total, H_, W_); ndim = 3
                    if ndim == 3:
                        scan_shape = _h5_infer_scan_shape(fh, s[0])
            except Exception as e:
                messagebox.showerror("h5 peek failed", repr(e)); return
            if ndim == 3 and scan_shape is None:
                from gui_app._dialogs import ask_scan_shape
                N, H, W = s
                scan_shape = ask_scan_shape(self, N, H, W)
                if scan_shape is None:
                    return
        try:
            from data import register_runtime_sample, SAMPLES
            kwargs = dict(vmax=v)
            if scan_shape is not None:
                kwargs["scan_shape"] = scan_shape
            key = register_runtime_sample(p, **kwargs)
        except Exception as e:
            messagebox.showerror("Load cube", f"failed:\n{e!r}"); return
        try:
            self._sample_menu.configure(values=sorted(SAMPLES.keys()))
        except Exception:
            pass
        self._vars["sample"].set(key)
        self._on_sample_change()

    # ----- run + stop --------------------------------------------------
    def _on_stop_clicked(self):
        if not self._compute_running:
            return
        self._stop_requested = True
        with self._lock:
            self._compute_progress = "stopping (waiting for current step to finish)…"
        try: self._stop_btn.configure(state="disabled")
        except Exception: pass

    def _check_stop(self):
        if self._stop_requested:
            raise RuntimeError("stop requested")

    def _kickoff_run(self):
        if self._compute_running:
            messagebox.showinfo("DINO+cluster",
                "compute already running"); return
        self._sync_from_pre()
        if not self.sample:
            messagebox.showinfo("DINO+cluster",
                "Load a cube in the Data tab (top) first."); return
        self._compute_running = True
        self._stop_requested = False
        self._run_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._compute_progress = "loading model + inference…"
        self._thread = threading.Thread(target=self._compute_worker,
                                            daemon=True)
        self._thread.start()
        self._poll()

    def _compute_worker(self):
        try:
            t0 = time.perf_counter()
            model_name = self._vars["model"].get()
            model_cfg = DINO_MODELS[model_name]
            vmax = float(self._vars["vmax"].get())
            batch = max(1, int(self._vars["batch"].get()))

            def cb(done, total, label):
                with self._lock:
                    self._compute_progress = (
                        f"{label}: {done}/{total} "
                        f"({100*done/max(total,1):.0f}%)")
            pth = str(self._vars["pth_path"].get()).strip() or None
            if pth and not os.path.exists(pth):
                raise RuntimeError(f".pth not found: {pth}")
            embeds, model_obj, src_used = compute_embeddings(
                self.sample, model_cfg, vmax,
                batch=batch, progress_cb=cb,
                stop_check=lambda: self._stop_requested,
                source=str(self._vars["source"].get()),
                pth_path=pth)
            self._check_stop()
            with self._lock:
                self._compute_progress = (
                    f"embeddings: shape={embeds.shape}  "
                    f"reducing / clustering…")
            # Optional PCA reduction.
            W = embeds
            if self._vars["use_pca"].get():
                from sklearn.decomposition import PCA
                d = max(2, int(self._vars["pca_dim"].get()))
                d = min(d, embeds.shape[1])
                pca = PCA(n_components=d, random_state=42)
                W = pca.fit_transform(embeds).astype(np.float32)
            # Auto K.
            if self._vars["auto_K"].get():
                with self._lock:
                    self._compute_progress = "auto K (silhouette)…"
                from gui_app.nmf_panel import auto_K
                K, sil = auto_K(W,
                                 progress_cb=lambda d, t, l: cb(d, t, l))
                self._vars["K"].set(K)
            else:
                K = int(self._vars["K"].get()); sil = None
            self._check_stop()
            # Cluster with chosen methods.
            from gui_app.nmf_panel import cluster_W
            labels = {}
            min_cs = int(self._vars["min_cluster_size"].get())
            if self._vars["use_kmeans"].get():
                with self._lock:
                    self._compute_progress = "K-means…"
                labels["K-means"] = cluster_W(W, "K-means", k=K)
                self._check_stop()
            if self._vars["use_aglo"].get():
                d = self._vars["aglo_dist"].get()
                with self._lock:
                    self._compute_progress = f"Aglo ({d})…"
                labels["Aglo"] = cluster_W(W, "Aglo", k=K, distance=d)
                self._check_stop()
            if self._vars["use_hdbscan"].get():
                with self._lock:
                    self._compute_progress = "HDBSCAN…"
                labels["HDBSCAN"] = cluster_W(
                    W, "HDBSCAN", min_cluster_size=min_cs)
                self._check_stop()
            if self._vars["use_fcm"].get():
                with self._lock:
                    self._compute_progress = "FCM…"
                labels["FCM"] = cluster_W(
                    W, "FCM", k=K,
                    fcm_m=float(self._vars["fcm_m"].get()))
                self._check_stop()
            self._last = dict(
                sample=self.sample,
                scan_shape=self._scan_shape,
                model=model_name, model_cfg=model_cfg,
                model_obj=model_obj, src_used=src_used,
                vmax=vmax,
                embeds=embeds, W=W, K=K,
                labels=labels, sil=sil,
            )
            with self._lock:
                self._compute_progress = (
                    f"done ({time.perf_counter() - t0:.1f}s)  "
                    f"D={embeds.shape[1]}  K={K}")
        except Exception as e:
            err = repr(e)
            print(f"[dino+cluster] worker failed: {err}", flush=True)
            with self._lock:
                self._compute_progress = (
                    "stopped by user." if "stop requested" in err
                    else f"failed: {err}")
        finally:
            self._compute_running = False

    def _poll(self):
        with self._lock:
            prog = self._compute_progress
        self._status_lbl.configure(text=prog or "(running…)")
        if self._compute_running:
            self.after(500, self._poll)
        else:
            self._run_btn.configure(state="normal")
            try: self._stop_btn.configure(state="disabled")
            except Exception: pass
            self._stop_requested = False
            if self._last is not None:
                self._render_last()

    # ----- rendering ---------------------------------------------------
    def _render_idle(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.text(0.5, 0.5,
                "1. Pick a sample.\n"
                "2. Pick a pretrained DINO model.\n"
                "3. Pick clustering methods.\n"
                "4. Click Run.\n\n"
                "First run downloads model weights via torch.hub "
                "(~100 MB – 4 GB depending on size).",
                ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        self._canvas.draw_idle()

    def _render_last(self):
        if self._last is None:
            self._render_idle(); return
        d = self._last
        embeds = d["embeds"]
        labels = d["labels"]
        K = d["K"]
        # Use the scan_shape pinned at Run time (defensive against the
        # user changing the sample dropdown after Run).
        Ny, Nx = (d.get("scan_shape") or self._scan_shape
                    or (1, embeds.shape[0]))

        self._fig.clear()
        n_methods = max(1, len(labels))
        # Class maps occupy the full canvas, large.  UMAP / embedding
        # scatter is opt-in via the sidebar 'Show UMAP' button.
        cols = n_methods
        for col, (method, lbl) in enumerate(labels.items()):
            ax = self._fig.add_subplot(1, cols, col + 1)
            n_classes = int(lbl.max()) + 1
            base = "tab20" if n_classes > 10 else "tab10"
            cmap = plt.get_cmap(base)
            palette = ListedColormap(
                [cmap(i % cmap.N) for i in range(n_classes)])
            grid = lbl.reshape(Ny, Nx) if lbl.size == Ny * Nx \
                else lbl.reshape(1, -1)
            im = ax.imshow(grid, cmap=palette,
                             vmin=-0.5, vmax=n_classes - 0.5,
                             interpolation="nearest", aspect="equal")
            ax.set_title(f"{method}   K={n_classes}", fontsize=11)
            ax.set_axis_off()
            cb = self._fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                                       ticks=list(range(n_classes)))
            cb.set_label("class id", fontsize=8)
            try:
                nm_per_px = float(self.app.real_res.get()) if self.app \
                              else 0.0
            except Exception:
                nm_per_px = 0.0
            if nm_per_px > 0:
                from gui_app._calib_utils import add_real_scalebar
                add_real_scalebar(ax, nm_per_px, length_nm=100,
                                    color="white")

        self._fig.suptitle(
            f"{self.sample}   model: {d['model']}   "
            f"D={embeds.shape[1]}   K={K}",
            fontsize=11)
        self._fig.tight_layout()
        # Click handlers on every class-map axis:
        #   left-click  → single-pattern attention popup
        #   right-click → grain-average attention popup
        # Tracked via a dict mapping axis → labels-grid so the handler
        # knows which method's map was clicked.
        if not hasattr(self, "_axis_to_grid"):
            self._axis_to_grid = {}
        self._axis_to_grid.clear()
        # Iterate Axes in the bottom row only.
        for ax in self._fig.axes:
            t = ax.get_title()
            if "K=" in t and ax.images:
                # bottom-row class maps are the ones with K= in the title
                # and an imshow — match labels by title prefix.
                for method, lbl in labels.items():
                    if t.startswith(method + " "):
                        grid = (lbl.reshape(Ny, Nx)
                                  if lbl.size == Ny * Nx
                                  else lbl.reshape(1, -1))
                        self._axis_to_grid[ax] = grid
                        break

        def _on_click(event):
            ax = event.inaxes
            if ax is None or ax not in self._axis_to_grid:
                return
            if event.xdata is None or event.ydata is None:
                return
            grid = self._axis_to_grid[ax]
            Hg, Wg = grid.shape
            xi = int(round(event.xdata))
            yi = int(round(event.ydata))
            xi = max(0, min(Wg - 1, xi))
            yi = max(0, min(Hg - 1, yi))
            scan_idx = yi * Wg + xi
            if event.button == 1:
                self._show_pattern_attention_popup(scan_idx)
            elif event.button == 3:
                self._show_grain_attention_popup(yi, xi, grid)
        # Disconnect prior cid first (otherwise re-render stacks handlers).
        cid = getattr(self, "_click_cid", None)
        if cid is not None:
            try: self._canvas.mpl_disconnect(cid)
            except Exception: pass
        self._click_cid = self._canvas.mpl_connect(
            "button_press_event", _on_click)
        self._canvas.draw_idle()

    # ----- interactive map --------------------------------------------
    def _recip_per_px(self):
        try:
            return float(self.app.recip_res.get()) if self.app else 0.0
        except Exception:
            return 0.0

    def _open_interactive_map(self):
        from data import SAMPLES
        if self._last is None or not self._last.get("labels"):
            messagebox.showinfo("interactive map",
                "Run DINO + clustering first."); return
        if not self.sample or self.sample not in SAMPLES:
            messagebox.showinfo("interactive map",
                "No dataset loaded for this sample."); return
        if not self._scan_shape:
            messagebox.showinfo("interactive map",
                "Scan shape unknown — cannot map clusters to positions.")
            return
        try:
            from gui_app.cluster_interactive import (
                open_interactive_clustermap)
            open_interactive_clustermap(
                self, sample=self.sample, scan_shape=self._scan_shape,
                labels=self._last["labels"],
                recip_per_px=self._recip_per_px(),
                title=f"DINO + cluster — {self.sample}")
        except Exception as e:
            messagebox.showerror("interactive map", repr(e))

    # ----- save --------------------------------------------------------
    def _save_snapshot(self):
        if self._last is None or not self.sample:
            messagebox.showinfo("save", "nothing to save"); return
        out_dir = os.path.join(
            self.outdir or ".", "dino_cluster",
            f"{self.sample}_{_safe_name(self._last['model'])}")
        os.makedirs(out_dir, exist_ok=True)
        try:
            self._fig.savefig(os.path.join(out_dir, "summary.png"),
                                dpi=140, bbox_inches="tight")
            np.save(os.path.join(out_dir, "embeds.npy"),
                    self._last["embeds"])
            np.save(os.path.join(out_dir, "W.npy"),
                    self._last["W"])
            for m, lbl in self._last["labels"].items():
                np.save(os.path.join(out_dir, f"labels_{m}.npy"), lbl)
            with open(os.path.join(out_dir, "summary.json"), "w") as fh:
                json.dump({
                    "sample": self.sample,
                    "model":  self._last["model"],
                    "D":      int(self._last["embeds"].shape[1]),
                    "K":      int(self._last["K"]),
                    "methods": list(self._last["labels"].keys()),
                }, fh, indent=2)
        except Exception as e:
            messagebox.showerror("save", repr(e)); return
        self._status_lbl.configure(text=f"saved → {out_dir}")

    # ====================================================================
    # Attention extraction (per pattern / per class / per grain)
    # ====================================================================
    def _show_umap_popup(self):
        """Open a Toplevel with a UMAP (or PCA-fallback) scatter of the
        embeddings, coloured by the user-picked clustering method."""
        if self._last is None:
            messagebox.showinfo("UMAP", "Run first."); return
        labels_dict = self._last.get("labels") or {}
        W = self._last["W"]

        win = tk.Toplevel(self)
        win.title(f"{self.sample} — embedding projection")
        win.geometry("900x780")

        ctrl = ctk.CTkFrame(win, fg_color="transparent")
        ctrl.pack(side="top", fill="x", padx=6, pady=4)
        method_var = ctk.StringVar(
            value=next(iter(labels_dict)) if labels_dict else "")
        ctk.CTkLabel(ctrl, text="colour by:").pack(side="left", padx=(8, 4))
        ctk.CTkOptionMenu(ctrl, variable=method_var,
                            values=list(labels_dict.keys()) or [""],
                            width=160,
                            command=lambda _v: _redraw()
                            ).pack(side="left", padx=4)
        status = ctk.CTkLabel(win, text="", font=("Consolas", 9),
                                 anchor="w")
        status.pack(side="top", fill="x", padx=8)

        fig = Figure(figsize=(7.5, 7.0), dpi=110, facecolor="white")
        canv = FigureCanvasTkAgg(fig, master=win)
        canv.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canv, win)

        # Cache the 2-D projection (UMAP is slow on big N — fit once).
        proj_cache = {}

        def _project():
            if "xy" in proj_cache:
                return proj_cache["xy"], proj_cache["tag"]
            status.configure(text="computing 2-D projection …")
            win.update_idletasks()
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                xy = reducer.fit_transform(W)
                tag = "UMAP"
            except Exception:
                from sklearn.decomposition import PCA
                xy = PCA(n_components=2,
                            random_state=42).fit_transform(W)
                tag = "PCA"
            proj_cache["xy"] = xy
            proj_cache["tag"] = tag
            return xy, tag

        def _redraw():
            xy, tag = _project()
            method = method_var.get()
            fig.clear()
            ax = fig.add_subplot(111)
            if method and method in labels_dict:
                lbl = labels_dict[method]
                n_classes = int(lbl.max()) + 1
                cmap = plt.get_cmap(
                    "tab20" if n_classes > 10 else "tab10")
                for c in range(n_classes):
                    m = lbl == c
                    ax.scatter(xy[m, 0], xy[m, 1], s=4, alpha=0.6,
                                  color=cmap(c % cmap.N),
                                  label=f"p{c}", edgecolors="none")
                ax.legend(fontsize=7, ncol=min(n_classes, 8),
                            loc="upper right")
            else:
                ax.scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.6,
                              color="0.4", edgecolors="none")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{tag} of embeddings  "
                          f"({W.shape[0]} pts × {W.shape[1]} dim)",
                          fontsize=11)
            fig.tight_layout()
            canv.draw_idle()
            status.configure(
                text=f"{tag} ready  (colour: {method or 'none'})")

        _redraw()

    def _have_model(self) -> bool:
        return (self._last is not None
                and self._last.get("model_obj") is not None)

    def _attention_ds(self):
        """Build a LoadPRZ for whichever sample was used at Run time
        (pinned in self._last). Avoids the bug where the user changes
        the sample dropdown after Run and the attention path tries to
        index the wrong cube → IndexError."""
        from data import SAMPLES, LoadPRZ
        s = (self._last or {}).get("sample") or self.sample
        v = float((self._last or {}).get(
            "vmax", float(self._vars["vmax"].get())))
        cfg = SAMPLES[s]
        return LoadPRZ(cfg["path"], resize=192, vmax=v)

    def _attention_for_raw(self, raw_2d: np.ndarray) -> np.ndarray:
        """Run the cached DINO on one raw 2D pattern and return a
        (H_p, W_p) attention map normalised to [0, 1]. Caller is
        responsible for upsampling / overlay."""
        import torch
        import torch.nn.functional as F
        if not self._have_model():
            raise RuntimeError("no cached DINO model — Run first")
        model = self._last["model_obj"]
        src = self._last["src_used"]
        cfg = self._last["model_cfg"]
        vmax = float(self._last["vmax"])
        device = next(model.parameters()).device
        sz = int(cfg["input_size"])
        x = (torch.from_numpy(raw_2d.astype(np.float32))
                  .unsqueeze(0).unsqueeze(0).to(device))     # (1, 1, h, w)
        # vmax-normalise to match training pipeline
        x = torch.clamp(x / max(vmax, 1e-6), 0.0, 1.0)
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(sz, sz),
                            mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device
                              ).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device
                              ).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.no_grad():
            heat = _extract_attention(model, x, src)         # (1, p, p)
        return heat[0].detach().cpu().numpy()

    def _attention_for_indices(self, idx_list: np.ndarray,
                                  top_n: int = 200,
                                  progress_cb=None) -> np.ndarray:
        """Average DINO attention over a list of scan indices. Subsamples
        to `top_n` patterns for speed. Returns the averaged (H_p, W_p)
        map normalised to [0, 1]."""
        ds = self._attention_ds()
        idx = np.asarray(idx_list).astype(np.int64).ravel()
        if idx.size == 0:
            raise RuntimeError("empty index list")
        if idx.size > top_n:
            rng = np.random.default_rng(42)
            idx = rng.choice(idx, top_n, replace=False)
        accum = None
        for i, scan_idx in enumerate(idx):
            try:
                raw = ds.get_raw(int(scan_idx)).astype(np.float32)
                heat = self._attention_for_raw(raw)
            except Exception as e:
                print(f"[attn] idx={scan_idx} skipped: {e!r}",
                      flush=True)
                continue
            if accum is None:
                accum = heat.astype(np.float64)
            else:
                accum += heat
            if progress_cb is not None and i % 16 == 0:
                progress_cb(i + 1, idx.size)
        if accum is None:
            raise RuntimeError("no patterns produced an attention map")
        avg = accum / max(1, idx.size)
        mn, mx = float(avg.min()), float(avg.max())
        return ((avg - mn) / max(mx - mn, 1e-12)).astype(np.float32)

    def _show_class_attentions(self):
        """Grid of K class-averaged attention maps overlaid on each
        class-averaged diffraction pattern."""
        if not self._have_model():
            messagebox.showinfo("attention", "Run first."); return
        labels_dict = self._last["labels"]
        if not labels_dict:
            messagebox.showinfo("attention",
                "No clustering results."); return
        # Use the first enabled method's labels.
        method = next(iter(labels_dict))
        lbl = labels_dict[method]
        K = int(lbl.max()) + 1
        win = tk.Toplevel(self)
        win.title(f"{self.sample} — class attentions  "
                    f"({self._last['model']}, {method})")
        win.geometry("1200x780")
        status = ctk.CTkLabel(win, text="computing per-class attention …",
                                 font=("Consolas", 9), anchor="w")
        status.pack(side="top", fill="x", padx=8, pady=4)
        fig = Figure(figsize=(12, 7.0), dpi=110, facecolor="white")
        canv = FigureCanvasTkAgg(fig, master=win)
        canv.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canv, win)

        def _worker():
            try:
                ds = self._attention_ds()
                # Class average pattern (in raw resolution) + attention.
                results = []
                for c in range(K):
                    idx = np.where(lbl == c)[0]
                    if idx.size == 0:
                        results.append(None); continue
                    self.after(0, lambda c=c, n=int(idx.size):
                                  status.configure(text=
                                  f"class p{c}: averaging {min(200, n)} "
                                  f"patterns + attention…"))
                    rng = np.random.default_rng(42)
                    pick = (idx if idx.size <= 200
                              else rng.choice(idx, 200, replace=False))
                    pats = np.stack(
                        [ds.get_raw(int(i)) for i in pick],
                        axis=0).astype(np.float32)
                    avg_pat = pats.mean(axis=0)
                    heat = self._attention_for_raw(avg_pat)
                    results.append((c, avg_pat, heat, int(idx.size)))
                # Render.
                def _draw():
                    fig.clear()
                    cols = min(K, 4)
                    rows = (K + cols - 1) // cols
                    for slot, r in enumerate(results):
                        if r is None:
                            continue
                        c, avg_pat, heat, n_pts = r
                        ax = fig.add_subplot(rows, cols, slot + 1)
                        # Display-stretch the underlying pattern.
                        ref = avg_pat.flatten()
                        if ref.size and ref.max() > 0:
                            lo = float(np.percentile(ref, 2))
                            hi = float(np.percentile(ref, 99.5))
                            disp = np.log1p(np.clip(avg_pat, lo, hi)
                                              - lo)
                        else:
                            disp = avg_pat
                        ax.imshow(disp, cmap="gray",
                                    aspect="equal",
                                    interpolation="nearest")
                        # Up-sample heat to pattern resolution.
                        h_up = _upsample_heatmap(heat, disp.shape)
                        ax.imshow(h_up, cmap="jet", alpha=0.55,
                                    aspect="equal",
                                    interpolation="bilinear")
                        ax.set_title(f"p{c}   N={n_pts}", fontsize=10)
                        ax.set_xticks([]); ax.set_yticks([])
                    fig.suptitle(
                        f"{self.sample} — class attentions  "
                        f"({self._last['model']})", fontsize=11)
                    fig.tight_layout()
                    canv.draw_idle()
                    status.configure(text=f"done. (method={method})")
                self.after(0, _draw)
            except Exception as e:
                err = repr(e)
                print(f"[attn] worker failed: {err}", flush=True)
                self.after(0, lambda:
                              messagebox.showerror("attention", err))
        threading.Thread(target=_worker, daemon=True).start()

    def _show_pattern_attention_popup(self, scan_idx: int):
        """Left-click handler: single pattern + attention overlay."""
        if not self._have_model():
            messagebox.showinfo("attention", "Run first."); return
        try:
            ds = self._attention_ds()
            if int(scan_idx) >= len(ds):
                raise IndexError(
                    f"scan_idx={scan_idx} out of range (cube has "
                    f"{len(ds)} frames). Did you change the sample "
                    f"after Run?")
            raw = ds.get_raw(int(scan_idx)).astype(np.float32)
            heat = self._attention_for_raw(raw)
        except Exception as e:
            messagebox.showerror("attention", repr(e)); return
        self._render_attention_popup(
            title=f"pattern (idx={scan_idx})",
            subtitle=f"{self.sample}  idx={scan_idx}",
            pattern=raw, heat=heat)

    def _show_grain_attention_popup(self, y: int, x: int,
                                       labels_grid: np.ndarray):
        """Right-click handler: grain-average attention (4-conn CC)."""
        if not self._have_model():
            messagebox.showinfo("attention", "Run first."); return
        try:
            from scipy.ndimage import label as cclabel
            Ny, Nx = labels_grid.shape
            cls = int(labels_grid[y, x])
            mask = (labels_grid == cls)
            lab_grid, _n = cclabel(mask)
            grain_id = int(lab_grid[y, x])
            if grain_id == 0:
                self._status_lbl.configure(
                    text="grain lookup failed"); return
            grain_mask = (lab_grid == grain_id)
            grain_pix = np.where(grain_mask.flatten())[0]
            n_pix = int(grain_pix.size)
            ds = self._attention_ds()
            # Guard: drop any flat indices past the cube length (defensive
            # against sample-swap-after-Run).
            grain_pix = grain_pix[grain_pix < len(ds)]
            if grain_pix.size == 0:
                raise RuntimeError(
                    "no in-range patterns in this grain — "
                    "did you change the sample after Run?")
            rng = np.random.default_rng(42)
            pick = (grain_pix if grain_pix.size <= 200
                      else rng.choice(grain_pix, 200, replace=False))
            pats = np.stack([ds.get_raw(int(i)) for i in pick],
                              axis=0).astype(np.float32)
            avg_pat = pats.mean(axis=0)
            heat = self._attention_for_raw(avg_pat)
        except Exception as e:
            messagebox.showerror("attention", repr(e)); return
        self._render_attention_popup(
            title=f"grain @ (y={y}, x={x}) — class p{cls}  "
                   f"({n_pix} pixels)",
            subtitle=f"{self.sample}  grain p{cls} ({n_pix} px)",
            pattern=avg_pat, heat=heat,
            grain_mask=grain_mask, labels_grid=labels_grid)

    def _render_attention_popup(self, title, subtitle, pattern, heat,
                                   grain_mask=None, labels_grid=None):
        """Shared 2/3-panel popup viewer for pattern + attention
        (and optional class map with grain in black)."""
        win = tk.Toplevel(self)
        win.title(title)
        n_panels = 3 if grain_mask is not None else 2
        win.geometry(f"{420 * n_panels}x540")
        fig = Figure(figsize=(4.5 * n_panels, 4.5), dpi=110,
                        facecolor="white")
        canv = FigureCanvasTkAgg(fig, master=win)
        canv.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canv, win)
        # Panel 1: class map with grain in black (only when grain mode).
        if grain_mask is not None:
            ax_map = fig.add_subplot(1, n_panels, 1)
            K = int(labels_grid.max()) + 1
            cmap = plt.get_cmap("tab20" if K > 10 else "tab10")
            palette = ListedColormap(
                [cmap(i % cmap.N) for i in range(K)])
            ax_map.imshow(labels_grid, cmap=palette,
                            vmin=-0.5, vmax=K - 0.5,
                            interpolation="nearest")
            overlay = np.zeros((*labels_grid.shape, 4), dtype=np.float32)
            overlay[grain_mask, 3] = 1.0
            ax_map.imshow(overlay, interpolation="nearest")
            ax_map.set_title("class map  (grain in black)",
                                fontsize=10)
            ax_map.set_axis_off()
            try:
                nm_per_px = float(self.app.real_res.get()) if self.app \
                              else 0.0
            except Exception:
                nm_per_px = 0.0
            if nm_per_px > 0:
                from gui_app._calib_utils import add_real_scalebar
                add_real_scalebar(ax_map, nm_per_px,
                                    length_nm=100, color="white")
            pat_pos = 2; atn_pos = 3
        else:
            pat_pos = 1; atn_pos = 2

        # Panel: pattern (log-stretched).
        ax_pat = fig.add_subplot(1, n_panels, pat_pos)
        ref = pattern.flatten()
        if ref.size and ref.max() > 0:
            lo = float(np.percentile(ref, 2))
            hi = float(np.percentile(ref, 99.5))
            disp = np.log1p(np.clip(pattern, lo, hi) - lo)
        else:
            disp = pattern
        ax_pat.imshow(disp, cmap="inferno",
                        aspect="equal", interpolation="nearest")
        ax_pat.set_title(subtitle, fontsize=9)
        ax_pat.set_xticks([]); ax_pat.set_yticks([])

        # Panel: attention overlay on the same pattern.
        ax_atn = fig.add_subplot(1, n_panels, atn_pos)
        ax_atn.imshow(disp, cmap="gray",
                        aspect="equal", interpolation="nearest")
        h_up = _upsample_heatmap(heat, disp.shape)
        ax_atn.imshow(h_up, cmap="jet", alpha=0.55,
                        aspect="equal", interpolation="bilinear")
        ax_atn.set_title("attention overlay", fontsize=10)
        ax_atn.set_xticks([]); ax_atn.set_yticks([])

        # Reciprocal scale bars on the two pattern panels (raw resolution).
        try:
            rp = float(self.app.recip_res.get()) if self.app else 0.0
        except Exception:
            rp = 0.0
        if rp > 0:
            from gui_app._calib_utils import add_recip_scalebar
            for a in (ax_pat, ax_atn):
                add_recip_scalebar(a, q_per_disp_px=rp, length_q=0.2)
        fig.tight_layout()
        canv.draw_idle()
