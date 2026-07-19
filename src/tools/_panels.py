"""Shared panel-label helper for paper figures. Labels a, b, c, ... placed at the
top-left of each axis, applied in ROW-MAJOR order by the caller."""
import string


def lbl(ax, i, dark=True, fs=12):
    """Put panel letter i (0->a, 1->b, ...) at the top-left of ax."""
    ax.text(0.035, 0.965, string.ascii_lowercase[i], transform=ax.transAxes,
            fontsize=fs, fontweight="bold", va="top", ha="left",
            color=("white" if dark else "black"),
            bbox=dict(boxstyle="round,pad=0.15", fc=("black" if dark else "white"),
                      alpha=0.55, ec="none"), zorder=30)
