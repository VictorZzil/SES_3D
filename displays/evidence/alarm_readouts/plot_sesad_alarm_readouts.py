"""Plot held-out operating rates and sensitivity to object-group splits."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def draw(source, output):
    data = json.loads(source.read_text(encoding="utf-8"))
    arms = ("COSSAD_NATIVE", "POINT_MAX", "SPATIAL_EVIDENCE")
    labels = ("COSSAD", "Point max.", "Spatial")
    colors = ("#6c7f72", "#56758a", "#a85b50")
    with plt.rc_context({"font.family": "Times New Roman", "font.size": 8,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": .55,
                         "axes.edgecolor": "#52606c", "text.color": "#22313c",
                         "axes.labelcolor": "#22313c", "xtick.color": "#3d4c55",
                         "ytick.color": "#22313c", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False}):
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.53), sharey=True)
        for ax, endpoint, title, limit, ticks in zip(
            axes, ("scored_fpr", "scored_miss_rate"),
            ("False alarms (%)", "Misses (%)"), ((0, 16), (0, 34)),
            ((0, 5, 10, 15), (0, 10, 20, 30))
        ):
            ax.set_title(title, fontsize=9, pad=4, fontweight="bold")
            ax.set_xlim(*limit)
            ax.set_xticks(ticks)
            ax.set_ylim(-.55, 2.58)
            ax.set_yticks(range(3))
            ax.set_yticklabels(labels[::-1])
            ax.tick_params(axis="both", length=0, pad=3)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#e1e6e7", linewidth=.55)
            if endpoint == "scored_fpr":
                ax.axvline(5, linewidth=.8, color="#6f747a", linestyle=(0, (2.5, 2.5)))
            for j, (name, color) in enumerate(zip(arms, colors)):
                y = 2-j
                agg = data["repeat_summary"][name][endpoint]
                low, middle, high = (100*agg[k] for k in ("q10", "mean", "q90"))
                fixed = 100*data["fixed_counts"][name][endpoint]
                ax.plot([low, high], [y+.10, y+.10], color=color, linewidth=1.2, solid_capstyle="round")
                ax.scatter(middle, y+.10, s=19, color=color, zorder=3)
                ax.scatter(fixed, y-.17, s=16, facecolor="white", edgecolor=color,
                           marker="D", linewidth=.85, zorder=3)
        axes[1].tick_params(labelleft=False)
        handles = [Line2D([0], [0], color="#515b62", marker="o", markersize=4,
                          label="Repeat mean and 10th–90th pct."),
                   Line2D([0], [0], color="#515b62", marker="D", markerfacecolor="white",
                          linestyle="none", markersize=3.5, label="Fixed split")]
        fig.legend(handles=handles, loc="lower center", fontsize=7.2, frameon=False, ncol=2,
                   handlelength=1.4, handletextpad=.4, columnspacing=1.15,
                   borderaxespad=.45)
        fig.subplots_adjust(left=.19, right=.98, top=.80, bottom=.25, wspace=.22)
        for extension in ("pdf", "svg"):
            fig.savefig(output.with_suffix("."+extension), metadata={"Creator": "Matplotlib"})
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    draw(args.summary, args.output)
