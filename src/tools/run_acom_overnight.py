"""Overnight launcher: run per-pixel ACOM + crystallinity for SI3/SI4/SI5
sequentially, each as a child subprocess, with timestamped logging and a final
ALL-DONE marker. Designed to be started DETACHED (Start-Process) so it survives
the Claude session going idle / user logout.
Usage: python run_acom_overnight.py [stride]   (default stride=1)"""
import subprocess, sys, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
STRIDE = sys.argv[1] if len(sys.argv) > 1 else "1"
SAMPLES = ["SI3", "SI4", "SI5"]

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

log(f"=== OVERNIGHT ACOM run start; stride={STRIDE}; python={PY} ===")
for s in SAMPLES:
    log(f">>> launching {s}")
    t0 = time.time()
    r = subprocess.run([PY, os.path.join(HERE, "imc_acom_fullpx.py"), s, STRIDE],
                       cwd=os.path.dirname(HERE))
    log(f"<<< {s} finished rc={r.returncode} in {(time.time()-t0)/60:.1f} min")
log("=== ALL DONE ===")
