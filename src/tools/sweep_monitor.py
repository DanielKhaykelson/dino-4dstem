"""sweep_monitor.py — long-running babysitter for the m-K sweep.

Polls SWEEP_PROGRESS.csv + the launcher stdout every POLL_SECS.
Touches STOP_SWEEP if any of the failure conditions are met.
Writes a short status line to MONITOR_STATUS.txt after every poll so
the user (and Claude on next interaction) can read the latest state
without parsing the full CSV.

Run in background alongside the sweep:
    python tools/sweep_monitor.py --root <sweep_root> [--stdout <file>]

Stops automatically when the sweep root contains a `report.html`
for every expected sample, OR when SWEEP_PROGRESS.csv has not
grown for STALL_HOURS, OR when STOP_SWEEP exists.
"""
from __future__ import annotations
import argparse, csv, os, sys, time
from datetime import datetime

POLL_SECS    = 1200    # 20 min between polls
STALL_HOURS  = 2.5     # no new CSV row + stdout idle this long => hang
EXPECTED_SAMPLES = ["Na007b", "EuInAs_B100", "IMC_SI5"]


def _read_csv(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _status_write(root, msg):
    out = os.path.join(root, "MONITOR_STATUS.txt")
    stamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    try:
        # Keep last ~200 lines.
        prev = []
        if os.path.exists(out):
            with open(out, encoding="utf-8") as f:
                prev = f.read().splitlines()[-200:]
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(prev + [line]) + "\n")
    except Exception:
        pass


def _touch_stop(root, reason: str):
    p = os.path.join(root, "STOP_SWEEP")
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"stopped by monitor: {reason}\n"
                f"timestamp: {datetime.now().isoformat()}\n")
    _status_write(root, f"STOPPED sweep -- reason: {reason}")


def _last_run_rows(rows, n=5):
    return rows[-n:] if len(rows) > n else rows


def _check_failures(rows):
    """Return (should_stop, reason) tuple."""
    # 1. 3+ consecutive failed (ok=False) runs.
    if len(rows) >= 3:
        last3 = rows[-3:]
        if all(r.get("ok", "").lower() in ("false", "0", "") and
               r.get("ok", "").lower() != "true" for r in last3):
            # Re-check: must be EXPLICITLY false (not just missing)
            if all(str(r.get("ok", "")).lower() == "false"
                   for r in last3):
                return True, ("3 consecutive ok=False runs: "
                              + "; ".join(r.get("error", "")[:80]
                                          for r in last3))
    # 2. NaN on 2+ consecutive runs.
    if len(rows) >= 2:
        last2 = rows[-2:]
        if all(str(r.get("has_nan", "")).lower() == "true"
               for r in last2):
            return True, "2 consecutive runs with NaN loss"
    # 3. Stage 1 all-collapse for ONE complete sample.
    by_sample = {}
    for r in rows:
        by_sample.setdefault(r.get("sample", "?"), []).append(r)
    for s, sample_rows in by_sample.items():
        stage1 = [r for r in sample_rows if r.get("stage") == "stage1"]
        if len(stage1) >= 16:                 # all 16 stage-1 cells done
            try:
                bad = 0
                for r in stage1:
                    ac = float(r.get("avg_conf_e5") or "nan")
                    keff = float(r.get("K_eff_end_smooth") or "nan")
                    if ac > 0.95 and keff < 1.5:
                        bad += 1
                if bad == len(stage1):
                    return True, (f"sample {s}: every stage-1 run "
                                  f"shows total-collapse signature "
                                  f"(avg_conf>0.95, K_eff<1.5)")
            except Exception:
                pass
    return False, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Sweep root containing SWEEP_PROGRESS.csv")
    ap.add_argument("--stdout", default=None,
                    help="Optional launcher stdout file to monitor "
                         "for stall detection.")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    csv_path = os.path.join(root, "SWEEP_PROGRESS.csv")
    stop_path = os.path.join(root, "STOP_SWEEP")
    if not os.path.isdir(root):
        print(f"[monitor] root not found: {root}", flush=True)
        sys.exit(2)

    _status_write(root, f"monitor started; polling every "
                            f"{POLL_SECS}s; root={root}")
    last_row_count = -1
    last_change_time = time.time()
    last_stdout_size = 0
    last_stdout_change = time.time()
    poll_n = 0

    while True:
        # Bail if the user pre-stopped the sweep.
        if os.path.exists(stop_path):
            _status_write(root, "STOP_SWEEP already exists; "
                                  "monitor exiting.")
            break

        rows = _read_csv(csv_path)
        n = len(rows)
        if n != last_row_count:
            last_row_count = n
            last_change_time = time.time()
        # Stdout growth check.
        idle_secs = None
        if args.stdout and os.path.exists(args.stdout):
            try:
                sz = os.path.getsize(args.stdout)
                if sz != last_stdout_size:
                    last_stdout_size = sz
                    last_stdout_change = time.time()
                idle_secs = time.time() - last_stdout_change
            except Exception:
                pass
        csv_idle = time.time() - last_change_time

        # Summary line.
        ok_n = sum(1 for r in rows
                       if str(r.get("ok", "")).lower() == "true")
        bad_n = sum(1 for r in rows
                        if str(r.get("ok", "")).lower() == "false")
        last_sample = rows[-1].get("sample", "?") if rows else "?"
        last_stage = rows[-1].get("stage", "?") if rows else "?"
        _status_write(root,
            f"poll #{poll_n}  rows={n}  ok={ok_n}  failed={bad_n}  "
            f"last=({last_sample}/{last_stage})  "
            f"csv_idle={csv_idle/60:.1f}m  "
            + (f"stdout_idle={idle_secs/60:.1f}m" if idle_secs
               else "stdout_idle=?"))

        # Stop conditions.
        should_stop, reason = _check_failures(rows)
        if should_stop:
            _touch_stop(root, reason)
            break

        # Stall: BOTH csv and stdout idle.
        stall = csv_idle / 3600.0 > STALL_HOURS
        if args.stdout and idle_secs is not None:
            stall = stall and (idle_secs / 3600.0 > STALL_HOURS)
        if stall:
            _touch_stop(root,
                f"stall detected (CSV idle {csv_idle/60:.0f}m, "
                f"stdout idle "
                f"{(idle_secs or 0)/60:.0f}m)")
            break

        # Natural completion: all expected samples have a final report.
        all_done = all(
            os.path.exists(os.path.join(root, s, "report.html"))
            for s in EXPECTED_SAMPLES)
        if all_done:
            _status_write(root, "all expected reports present; "
                                  "sweep finished naturally; monitor "
                                  "exiting.")
            break

        poll_n += 1
        time.sleep(POLL_SECS)


if __name__ == "__main__":
    main()
