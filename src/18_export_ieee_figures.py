
#!/usr/bin/env python3
"""Export IEEE single-column versions of Stage 17 manuscript Figs. 2 and 3.

READS (existing, validated Stage 16 tables only):
  outputs/stage16_statistical_audit/stage16_by_condition.csv
  outputs/stage16_statistical_audit/stage16_roi_level.csv

WRITES (new folder only, no model training or modifications to Stage 16/17):
  outputs/ieee_figures/condition_effects.png / .svg
  outputs/ieee_figures/paired_roi_scatter.png / .svg

Place this file in src/ and run from the project root:
  python src/18_export_ieee_figures.py
"""

from pathlib import Path
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGE16 = ROOT / "outputs" / "stage16_statistical_audit"
OUTPUT = ROOT / "outputs" / "ieee_figures"

CONDITIONS = (
    "H_low_severe",
    "H_high_severe",
    "E_low_severe",
    "E_high_severe",
    "HE_low_severe",
    "HE_high_severe",
)

LABELS = (
    "H lower",
    "H higher",
    "E lower",
    "E higher",
    "H+E lower",
    "H+E higher",
)

COLORS = {
    "adaptive": "#006E68",
    "primary": "#3D628C",
    "metastatic": "#9A6A53",
}


def configure_ieee_fonts():
    families = {f.name for f in font_manager.fontManager.ttflist}

    if "Times New Roman" in families:
        family = "Times New Roman"

    elif os.getenv("IEEE_FONT_FALLBACK") == "1":
        family = (
            "Liberation Serif"
            if "Liberation Serif" in families
            else "DejaVu Serif"
        )
        print(
            f"PREVIEW ONLY: Times New Roman unavailable; "
            f"using {family}."
        )

    else:
        raise RuntimeError(
            "Times New Roman was not found by Matplotlib. "
            "On Windows, ensure it is installed, restart "
            "VS Code/Python, then run again. "
            "For a non-submission preview only, set "
            "IEEE_FONT_FALLBACK=1."
        )

    plt.rcParams.update({
        "font.family": family,
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#505B66",
        "axes.labelcolor": "#263441",
        "xtick.color": "#344454",
        "ytick.color": "#344454",
        "text.color": "#263441",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    })

    print(
        f"Figure font: {family}, "
        f"8 pt at native 3.45-inch figure width"
    )


def load_data():
    condition_file = (
        STAGE16 / "stage16_by_condition.csv"
    )

    roi_file = (
        STAGE16 / "stage16_roi_level.csv"
    )

    for path in (condition_file, roi_file):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. "
                "Check that Stage 16 outputs exist "
                "under the project folder and that "
                "this script is saved in src/."
            )

    cond = (
        pd.read_csv(condition_file)
        .set_index("condition")
        .loc[list(CONDITIONS)]
        .reset_index()
    )

    roi = pd.read_csv(roi_file)

    assert (
        len(cond) == 6
        and (cond.n_rois == 205).all()
    ), "Condition summary failed audit"

    assert (
        len(roi) == 205
        and roi.roi.nunique() == 205
    ), "ROI summary failed audit"

    assert (
        roi.fold.nunique() == 5
    ), "Expected five held-out folds"

    assert (
        int((roi.mean_adaptive_advantage > 0).sum()) == 200
    ), "ROI wins differ from audit"

    assert np.isclose(
        roi.mean_adaptive_advantage.mean(),
        0.0729701049713311,
        atol=1e-8,
    ), "Paired Dice differs from audit"

    for _, row in cond.iterrows():
        assert (
            row.roi_bootstrap_95ci_low
            <= row.mean_adaptive_advantage
            <= row.roi_bootstrap_95ci_high
        )

    print(
        "DATA VALIDATION PASS: "
        "205 ROIs, 5 folds, "
        "6 severe conditions, 200 ROI wins"
    )

    return cond, roi


def save_both(fig, name):
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    png = OUTPUT / f"{name}.png"
    svg = OUTPUT / f"{name}.svg"

    fig.savefig(
        png,
        dpi=600,
        facecolor="white",
    )

    fig.savefig(
        svg,
        facecolor="white",
    )

    plt.close(fig)

    print(f"CREATED {png}")
    print(f"CREATED {svg}")


def export_condition_effects(cond):
    """Export six conditions and their bootstrap intervals."""

    d = (
        cond.set_index("condition")
        .loc[list(CONDITIONS)]
    )

    means = (
        d.mean_adaptive_advantage
        .to_numpy(dtype=float)
    )

    lower = (
        d.roi_bootstrap_95ci_low
        .to_numpy(dtype=float)
    )

    upper = (
        d.roi_bootstrap_95ci_high
        .to_numpy(dtype=float)
    )

    wins = (
        d.roi_wins
        .to_numpy(dtype=int)
    )

    positions = np.arange(5, -1, -1)

    # IEEE single-column dimensions
    fig = plt.figure(
        figsize=(3.45, 2.65),
        facecolor="white",
    )

    # Reserve space at right for ROI win counts
    ax = fig.add_axes([
        0.235,
        0.19,
        0.525,
        0.74,
    ])

    ax.axvline(
        0,
        color="#79838B",
        linewidth=0.8,
    )

    # Plot validated Stage 16 confidence intervals
    ax.errorbar(
        means,
        positions,
        xerr=np.vstack((
            means - lower,
            upper - means,
        )),
        fmt="o",
        color=COLORS["adaptive"],
        ecolor=COLORS["adaptive"],
        markersize=4.5,
        capsize=2.7,
        elinewidth=1.25,
        capthick=1.05,
    )

    ax.set_yticks(
        positions,
        LABELS,
    )

    ax.set_ylim(-0.55, 5.55)
    ax.set_xlim(-0.008, 0.178)

    ax.set_xticks([
        0.00,
        0.05,
        0.10,
        0.15,
    ])

    ax.xaxis.set_major_formatter(
        FormatStrFormatter("%.2f")
    )

    ax.set_xlabel(
        "Mean Dice: adaptive - frozen",
        labelpad=4,
    )

    ax.tick_params(
        axis="both",
        pad=2,
        length=2.7,
    )

    ax.grid(
        axis="x",
        color="#CCD5D9",
        alpha=0.33,
        linewidth=0.55,
    )

    ax.set_axisbelow(True)

    # Right-side counts, not part of the data x-axis
    fig.text(
        0.785,
        0.95,
        "ROI wins",
        fontsize=8,
        ha="left",
        va="center",
    )

    for y, win in zip(positions, wins):
        ax.text(
            1.045,
            y,
            f"{win}/205",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8,
            color="#354350",
            clip_on=False,
        )

    save_both(
        fig,
        "condition_effects",
    )


def export_paired_roi_scatter(roi):
    """Export the 205 ROI-averaged Dice pairs."""

    fig = plt.figure(
        figsize=(3.45, 3.45),
        facecolor="white",
    )

    ax = fig.add_axes([
        0.195,
        0.17,
        0.755,
        0.77,
    ])

    for kind in ("primary", "metastatic"):
        g = roi.loc[
            roi.slide_type.str.lower() == kind
        ]

        if len(g) == 0:
            raise AssertionError(
                f"Missing slide type: {kind}"
            )

        ax.scatter(
            g.mean_frozen_dice,
            g.mean_adaptive_dice,
            s=17,
            alpha=0.78,
            c=COLORS[kind],
            label=kind.capitalize(),
            edgecolors="white",
            linewidths=0.25,
        )

    # Identity line: adaptive Dice = frozen Dice
    ax.plot(
        [0.32, 1.0],
        [0.32, 1.0],
        "--",
        lw=1.0,
        color="#586875",
        zorder=0,
    )

    ax.set_xlim(0.32, 1.0)
    ax.set_ylim(0.32, 1.0)

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xticks(
        np.arange(0.4, 1.01, 0.1)
    )

    ax.set_yticks(
        np.arange(0.4, 1.01, 0.1)
    )

    ax.xaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

    ax.set_xlabel(
        "Mean Dice, frozen scaling",
        labelpad=3,
    )

    ax.set_ylabel(
        "Mean Dice, adaptive scaling",
        labelpad=3,
    )

    ax.tick_params(
        axis="both",
        pad=2,
        length=2.7,
    )

    ax.grid(
        color="#CCD5D9",
        alpha=0.32,
        linewidth=0.55,
    )

    ax.set_axisbelow(True)

    ax.legend(
        loc="lower right",
        frameon=False,
        markerscale=1.1,
        labelspacing=0.3,
        handlelength=1.1,
        borderaxespad=0.2,
    )

    save_both(
        fig,
        "paired_roi_scatter",
    )


def main():
    configure_ieee_fonts()

    cond, roi = load_data()

    export_condition_effects(cond)
    export_paired_roi_scatter(roi)

    print(
        "SUCCESS: Only two IEEE figure pairs regenerated. "
        "Stage 16/17 unchanged."
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        raise
