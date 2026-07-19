#!/usr/bin/env bash
# Auto-relaunch watchdog for run_model_comparisons.py: survives NVIDIA driver
# crashes (nvlddmkm TDR) that silently kill the Python process. The runner's
# skip-logic (a cell counts done only with eval/metrics.json) makes every
# relaunch resume where it left off, losing at most the in-progress cell.
cd /d/DINOSR/Claude/PaperRun_claude/dino_sr_contrastive || exit 1
PY=/c/Users/danielkh/AppData/Local/anaconda3/envs/py4DSTEM_SAM/python.exe
TARGET=13          # total cells (2 ViT + 8 depth + 3 cluster1d)
MAX_ATTEMPTS=40
for ((i=1; i<=MAX_ATTEMPTS; i++)); do
  DONE=$(find runs/ModelComparisons -name metrics.json 2>/dev/null | wc -l)
  echo "[watchdog $(date '+%F %T')] attempt $i  ($DONE/$TARGET cells done)"
  if [ "$DONE" -ge "$TARGET" ]; then echo "[watchdog] ALL $TARGET CELLS DONE"; break; fi
  "$PY" src/run_model_comparisons.py
  echo "[watchdog $(date '+%F %T')] runner exited code $? (likely driver crash if mid-cell)"
  sleep 20
done
echo "[watchdog] final: $(find runs/ModelComparisons -name metrics.json 2>/dev/null | wc -l)/$TARGET cells done"
