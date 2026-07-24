"""Shared publication figure style (APL two-column) and validated palette."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette (dataviz reference instance, light mode)
C = {
    "blue":    "#2a78d6",
    "aqua":    "#1baf7a",
    "yellow":  "#eda100",
    "green":   "#008300",
    "violet":  "#4a3aa7",
    "red":     "#e34948",
    "magenta": "#e87ba4",
    "orange":  "#eb6834",
    "ink":     "#0b0b0b",
    "ink2":    "#52514e",
    "muted":   "#8a8984",
    "grid":    "#e5e4e0",
    "surface": "#fcfcfb",
}
SERIES = [C["blue"], C["aqua"], C["yellow"], C["green"], C["violet"], C["red"]]

SINGLE_COL = 3.37   # inches
DOUBLE_COL = 6.90

def use_style():
    plt.rcParams.update({
        "font.size": 7.5, "font.family": "DejaVu Sans",
        "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.8, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "axes.edgecolor": C["ink2"],
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.color": C["ink2"], "ytick.color": C["ink2"],
        "axes.labelcolor": C["ink"], "text.color": C["ink"],
        "grid.color": C["grid"], "grid.linewidth": 0.5,
        "lines.linewidth": 1.4, "lines.markersize": 4,
        "figure.dpi": 200, "savefig.dpi": 400,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })

def panel_label(ax, s, dx=-0.12, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left", color=C["ink"])
