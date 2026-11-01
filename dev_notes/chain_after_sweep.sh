#!/bin/bash
# Chain after the current SupCon sweep finishes:
#   1. Wait for current sweep done
#   2. Run Tfin ablation (C2 only, both samples)
#   3. Run analysis + select winner
#   4. Run winner on Na007a + Na006a
#   5. Re-run analysis with all 4 samples

PY="/c/Users/danielkh/AppData/Local/anaconda3/envs/py4DSTEM_SAM/python.exe"
ROOT="/d/DINOSR/Claude/PaperRun_claude/dino_sr_contrastive"
SWEEP_LOG="/d/DINOSR/Claude/PaperRun_claude/logs/supcon_sweep_20260426_193353.log"

cd "$ROOT" || exit 1

echo "[chain $(date +%H:%M:%S)] step 1: waiting for current sweep to finish"
until grep -q 'supcon sweep] done in' "$SWEEP_LOG" 2>/dev/null; do
  sleep 60
done
echo "[chain $(date +%H:%M:%S)] current sweep finished"

echo "[chain $(date +%H:%M:%S)] step 2: launching Tfin ablation"
"$PY" -u run_supcon_tfin_sweep.py 2>&1
echo "[chain $(date +%H:%M:%S)] Tfin sweep finished"

echo "[chain $(date +%H:%M:%S)] step 3: running analysis + winner selection"
"$PY" -u analyze_supcon_sweep.py 2>&1
echo "[chain $(date +%H:%M:%S)] analysis done"

echo "[chain $(date +%H:%M:%S)] step 4: winner config on Na007a + Na006a"
"$PY" -u run_winner_other_samples.py 2>&1
echo "[chain $(date +%H:%M:%S)] winner runs finished"

echo "[chain $(date +%H:%M:%S)] step 5: final analysis with all 4 samples"
"$PY" -u analyze_supcon_sweep.py 2>&1
echo "[chain $(date +%H:%M:%S)] FINAL analysis done"
echo "[chain $(date +%H:%M:%S)] report: runs/_supcon_sweep/REPORT.md"
