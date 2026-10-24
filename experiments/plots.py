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


def plot_group_dots(rows: list[tuple], metric: str, title: str, xlabel: str, labels: tuple[str, str], out,
                    reference: str | None = None) -> None:
    """Dot plot of one rate for every group under two models, with Wilson intervals where I have them.

    rows holds (group label, main model record, other model record). With reference set, the plot shows
    metric minus reference (e.g. mean PD minus actual default rate) around a zero line instead.
    """
    import numpy as np

    fig, ax = new_figure(7.2, 0.42 * len(rows) + 1.4)
    ys = np.arange(len(rows))[::-1]
    for offset, which, color, label in [(0.14, 1, SERIES[0], labels[0]), (-0.14, 2, SERIES[1], labels[1])]:
        for yy, row in zip(ys, rows):
            r = row[which]
            value = r[metric] if reference is None else r[metric] - r[reference]
            # the interval line goes under the dot
            if reference is None and f"{metric}_low" in r:
                ax.plot([r[f"{metric}_low"] * 100, r[f"{metric}_high"] * 100], [yy + offset] * 2, color=color,
                        linewidth=2, solid_capstyle="round")
            ax.plot(value * 100, yy + offset, "o", color=color, markersize=6,
                    markeredgecolor="white", markeredgewidth=1.2, label=label if yy == ys[0] else None)
    if reference is not None:
        ax.axvline(0, color=REFERENCE, linewidth=1)
    ax.set_yticks(ys, [r[0] for r in rows], color=TEXT_MUTED)
    style_axes(ax, title, xlabel, "")
    ax.grid(axis="y", visible=False)
    # the legend sits under the axis label so it never runs into the title
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    save(fig, out)
