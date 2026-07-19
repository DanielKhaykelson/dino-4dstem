"""Figure 6 (real screenshots): the DINO4DSTEM software and its multi-scale
inspection, built from genuine GUI captures (from ForLothar deck). Top: the full
GUI window (class-averages / per-class view). Bottom: two real inspection levels —
single frame + GradCAM, and single grain (grain average + GradCAM). Saves to
BorisEdits + latest_review.
  python src/tools/fig6_gui_real.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import image as mpimg

FIGS = "docs/paper/draft_v2/figs"; OUT = os.path.join(FIGS, "BorisEdits"); REVIEW = os.path.join(FIGS, "latest_review")
D = os.path.join(OUT, "gui_real")
win = mpimg.imread(os.path.join(D, "gui_window_classes.png"))   # full GUI, class averages
frm = mpimg.imread(os.path.join(D, "gui_single_frame.png"))     # single frame + GradCAM
grn = mpimg.imread(os.path.join(D, "gui_single_grain.png"))     # single grain avg + GradCAM

fig = Figure(figsize=(12.5, 11.2), facecolor="white")
gs = fig.add_gridspec(2, 2, height_ratios=[1.62, 1.0], hspace=0.10, wspace=0.06,
                      left=0.015, right=0.985, top=0.93, bottom=0.02)

axA = fig.add_subplot(gs[0, :]); axA.imshow(win); axA.set_xticks([]); axA.set_yticks([])
axA.set_title("A   The graphical interface — one fixed model; the user sets only pre-processing",
              fontsize=12, fontweight="bold", loc="left", pad=6)

axB = fig.add_subplot(gs[1, 0]); axB.imshow(frm); axB.set_xticks([]); axB.set_yticks([])
axB.set_title("B   Single-frame view + GradCAM", fontsize=12, fontweight="bold", loc="left", pad=6)

axC = fig.add_subplot(gs[1, 1]); axC.imshow(grn); axC.set_xticks([]); axC.set_yticks([])
axC.set_title("C   Single-grain view (class map · grain average · GradCAM)", fontsize=12, fontweight="bold", loc="left", pad=6)

fig.suptitle("Multi-scale, human-in-the-loop inspection in the DINO4DSTEM interface",
             fontsize=13.5, fontweight="bold", y=0.985)
FigureCanvasAgg(fig)
p = os.path.join(OUT, "fig6_gui_real.png"); fig.savefig(p, dpi=150, facecolor="white")
import shutil; shutil.copy(p, os.path.join(REVIEW, "fig6_gui_real.png"))
print("wrote fig6_gui_real.png", flush=True)
