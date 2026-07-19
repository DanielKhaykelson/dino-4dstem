#!/bin/bash
# Train the five 50nm IMC datasets sequentially with the 150nm recipe.
# Restart-proof: a dataset is "done" only when BOTH _done.flag (training finished)
# and eval/metrics.json (eval finished) exist. An interrupted run (best.pth but no
# _done.flag, e.g. a GPU TDR crash) is cleaned and retrained from scratch. Loops
# until all five are complete, so it survives crashes.
cd /d/DINOSR/Claude/PaperRun_claude/dino_sr_contrastive
P=/c/Users/danielkh/AppData/Local/anaconda3/envs/py4DSTEM_SAM/python.exe
DATASETS="50nm_SI1 50nm_SI2 50nm_SI3 50nm_SI4 50nm_SI5"

# Single-instance lock: refuse to start if another watchdog is already alive.
LOCK="runs/_gui/.watchdog50.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[watchdog] another instance (pid $(cat "$LOCK")) is already running -> exiting"; exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

while true; do
  alldone=1
  for d in $DATASETS; do
    od="runs/_gui/IMC50_${d#50nm_}"
    if [ -f "$od/_done.flag" ] && [ -f "$od/eval/metrics.json" ]; then
      continue
    fi
    alldone=0
    # Interrupted training (best.pth or a stale eval but no _done.flag) -> wipe and retrain
    if [ ! -f "$od/_done.flag" ] && { [ -f "$od/best.pth" ] || [ -f "$od/eval/metrics.json" ]; }; then
      echo "[watchdog $(date +%H:%M:%S)] $d interrupted (no _done.flag) -> cleaning .pth + stale eval"
      rm -f "$od"/*.pth; rm -rf "$od/eval"
    fi
    echo "[watchdog $(date +%H:%M:%S)] === running $d -> $od ==="
    PYTHONIOENCODING=utf-8 $P src/run_si_extra.py "$d" 2>&1 | grep -viE "deprecat|warn\(|UserWarning"
    break   # re-scan from the top after each dataset so a crash restarts cleanly
  done
  if [ "$alldone" = "1" ]; then
    echo "[watchdog $(date +%H:%M:%S)] ALL 5 DONE"
    break
  fi
done
