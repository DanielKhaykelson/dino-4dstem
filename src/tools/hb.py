"""Heartbeat: append epoch + wall-clock every 5s for ~40 min. Used to test
whether a process keeps running while the Claude session is idle / user logged out.
Usage: python hb.py <tag>"""
import sys, time, os
tag = sys.argv[1] if len(sys.argv) > 1 else "X"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"hb_{tag}.log")
for k in range(480):
    with open(out, "a") as f:
        f.write(f"{tag} tick {k:4d} epoch={time.time():.1f} {time.strftime('%H:%M:%S')}\n")
    time.sleep(5)
