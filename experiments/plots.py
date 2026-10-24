"""Shared matplotlib styling for the report figures, so every chart looks the same."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # I only save files, so no window is needed
import matplotlib.pyplot as plt  # noqa: E402

# categorical colours in a fixed order (blue, orange, aqua, yellow), checked for colour-blind separation
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
REFERENCE = "#8a8984"  # grey for diagonals and baselines
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    # quiet axes and grid so the data stands out
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=TEXT, pad=10)
    ax.set_xlabel(xlabel, color=TEXT_MUTED)
    ax.set_ylabel(ylabel, color=TEXT_MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED)


def new_figure(width: float = 6.4, height: float = 4.8):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
