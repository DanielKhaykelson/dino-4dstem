# dino_sr_contrastive/

Fresh-session workspace for the DINO-SR + Contrastive Head experiment.

## What's in here

| file | purpose |
|------|---------|
| `SPEC.md` | Verbatim spec extracted from `dino sr contrastive prompt.pdf`. The source of truth for requirements. |
| `NEW_SESSION_PROMPT.md` | Hand-off prompt for the fresh session. Contains the spec + Q&A clarifications + environment info + concrete task list. **Start here.** |
| `dino sr contrastive prompt.pdf` | Original PDF spec (reference). |
| `dino_sr_ablation.py` | Copy of the current DINO-SR L2/L1 trainer. Contains `AblationDINOModelSR`, `train_ablation()`, `PolarTransform`, `PolarMaskLeft`, `get_ablation_transforms()`. Edit in place. |
| `dino_sr_fixed.py` | Copy of shared model components (`CenterMask`, `Prototypes`, `ProjectionHead`, aug transforms, helper metrics). Imported by `dino_sr_ablation.py`. Edit if needed; prefer not to. |
| `eval_all.py` | Copy. Contains `LoadPRZ` dataset + `SAMPLES` registry with local paths. |
| `elliptical_correction.py` | Copy. `EllipticalCorrection` + PACBED + `fit_ellipse_affine`. Contrastive spec doesn't require this — keep for optional reuse. |
| `scorecard.py` | Copy. Existing scorecard metrics (`conf_score`, `effk_score`, etc.). Contrastive spec's eval metrics are different (NMI, KNN purity, intra/inter cosine, etc.) — build those fresh. Keep scorecard for optional reuse during debugging. |
| `run_with_scorecard.py` | Copy. Existing CLI entry point. The new session should fork a new entry point (e.g. `run_contrastive.py`) rather than editing this. |

## Rules (from spec)

- **Edit only the copies in this subfolder.** Do not touch the originals in `../`.
- All edits stay here, keep `# %%` Jupytext cell structure if producing notebook-style files, `bbox_inches='tight'` on figure saves, tqdm-only progress.
- `contrastive_lambda=0` + `theta_roll_aug=False` must bit-exactly reproduce a pure-DINO L1 run.

## Hand-off

Read `NEW_SESSION_PROMPT.md` first. Everything needed to execute is there or referenced from there.
