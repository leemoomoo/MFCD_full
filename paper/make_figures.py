"""Regenerate the two figures that were missing from the paper folder.

Fig. 1  figures/cd.png             -- toy example of standard CD vs. MFCD
Fig. 3  figures/ours_flowchart.pdf -- overall architecture of MFCD

Run:  python make_figures.py
"""
import os

import matplotlib

matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 9,
    }
)

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

EDGE = "#2f4b63"
TEXT = "#142433"
FILL_NEUTRAL = "#eef2f6"
FILL_COG = "#e6eef7"
FILL_BEH = "#eaf3ea"
FILL_FUSE = "#fdf1e0"
FILL_OUT = "#f7e9ee"


def box(ax, x, y, w, h, text, fill=FILL_NEUTRAL, fontsize=8.5, bold=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=0.9,
            edgecolor=EDGE,
            facecolor=fill,
        )
    )
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT,
        fontweight="bold" if bold else "normal",
        linespacing=1.4,
    )


def arrow(ax, start, end, style="-|>", color=EDGE, lw=0.95, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=9,
            linewidth=lw,
            color=color,
            linestyle=ls,
            shrinkA=1.5,
            shrinkB=1.5,
        )
    )


def blank_axes(fig, rect, xlim, ylim):
    ax = fig.add_axes(rect)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return ax


def figure1():
    fig = plt.figure(figsize=(7.2, 2.5))
    ax_a = blank_axes(fig, [0.02, 0.06, 0.46, 0.86], (0, 10), (0, 5))
    ax_b = blank_axes(fig, [0.52, 0.06, 0.46, 0.86], (0, 10), (0, 5))

    ax_a.text(0.1, 4.85, "(a) Standard CD", fontsize=9, fontweight="bold", color=TEXT)
    box(ax_a, 0.1, 3.2, 2.5, 1.0, "Interaction log\n(binary correctness)", FILL_NEUTRAL, 7.6)
    box(ax_a, 3.4, 3.2, 2.6, 1.0, "CDM\n(IRT / DINA / NCDM)", FILL_COG, 7.6)
    box(ax_a, 6.8, 3.2, 2.7, 1.0, "Mastery only\n(one dimension)", FILL_OUT, 7.6)
    arrow(ax_a, (2.6, 3.7), (3.4, 3.7))
    arrow(ax_a, (6.0, 3.7), (6.8, 3.7))

    ax_a.plot([1.15, 1.15, 1.75, 1.75, 2.35, 2.35], [1.9, 0.95, 0.95, 1.25, 1.25, 1.9],
              color=FILL_OUT, linewidth=7, solid_capstyle="butt")
    ax_a.text(0.1, 2.35, "Estimated profile", fontsize=7.6, color=TEXT)
    ax_a.text(0.1, 0.35,
              "A single careless slip is indistinguishable\nfrom a lack of cognitive mastery.",
              fontsize=7.4, color="#5b6b7a", style="italic", linespacing=1.5)

    ax_b.text(0.1, 4.85, "(b) Mastery-fluency CD (MFCD)", fontsize=9, fontweight="bold", color=TEXT)
    box(ax_b, 0.1, 3.2, 1.9, 1.0, "Interaction log\n+ history", FILL_NEUTRAL, 7.6)
    box(ax_b, 2.7, 3.7, 2.9, 0.85, "Mastery stream\n(CDM backbone)", FILL_COG, 7.6)
    box(ax_b, 2.7, 2.2, 2.9, 0.95, "Behavioral stream\nEI, EA, RC proxies", FILL_BEH, 7.6)
    box(ax_b, 6.2, 3.0, 1.7, 1.4, "Dynamic\nfusion\ngate", FILL_FUSE, 7.6)
    box(ax_b, 8.2, 3.0, 1.7, 1.4, "Mastery\n+\nFluency", FILL_OUT, 7.6)
    arrow(ax_b, (2.0, 3.85), (2.7, 4.05))
    arrow(ax_b, (2.0, 3.55), (2.7, 2.75))
    arrow(ax_b, (5.6, 4.1), (6.2, 3.9))
    arrow(ax_b, (5.6, 2.7), (6.2, 3.5))
    arrow(ax_b, (7.9, 3.7), (8.2, 3.7))

    ax_b.text(0.1, 1.75, "Disentangled", fontsize=7.6, color=TEXT)
    ax_b.plot([2.6, 2.6, 3.15, 3.15], [1.5, 0.75, 0.75, 1.5],
              color=FILL_COG, linewidth=7, solid_capstyle="butt")
    ax_b.plot([3.55, 3.55, 4.1, 4.1], [1.5, 1.05, 1.05, 1.5],
              color=FILL_BEH, linewidth=7, solid_capstyle="butt")
    ax_b.text(2.6, 0.35, "mastery\n$0.78$", fontsize=7.0, color=TEXT, ha="center", linespacing=1.4)
    ax_b.text(3.55, 0.35, "fluency\n$0.41$", fontsize=7.0, color=TEXT, ha="center", linespacing=1.4)
    ax_b.text(0.1, 1.05,
              "Mastery and fluency are estimated separately,\nso slips no longer erase mastery evidence.",
              fontsize=7.4, color="#5b6b7a", style="italic", linespacing=1.5)

    path = os.path.join(OUT, "cd.png")
    fig.savefig(path, dpi=400)
    plt.close(fig)
    print("wrote", path)


def figure3():
    fig = plt.figure(figsize=(7.2, 4.0))
    ax = blank_axes(fig, [0.0, 0.0, 1.0, 1.0], (0, 12), (0, 7))

    box(ax, 0.15, 5.15, 2.5, 1.05,
        "Interaction records $\\mathcal{R}$,\nQ-matrix $\\boldsymbol{Q}$,\nstudent $u_i$, exercise $e_j$",
        FILL_NEUTRAL, 7.3)

    box(ax, 3.35, 5.35, 3.5, 0.85,
        "Cognitive (mastery) stream\nbackbone CDM: IRT / DINA / NCDM / ...", FILL_COG, 7.3)
    box(ax, 3.35, 3.95, 3.5, 0.8,
        "cognitive logit $o^{\\mathrm{logit}}_{i,j}$\nand exercise features $f^{\\mathrm{exe}}_j$",
        FILL_COG, 7.3)

    box(ax, 3.35, 2.55, 1.05, 0.72, "$f^{\\mathrm{EI}}$\n$f^{\\mathrm{EA}}$\n$f^{\\mathrm{RC}}$", FILL_BEH, 7.0)
    box(ax, 4.85, 2.55, 2.0, 0.72, "concept encoders\nthen aggregation", FILL_BEH, 7.0)
    box(ax, 3.35, 1.35, 3.5, 0.72,
        "behavioral latent representation $z_{i,j}$", FILL_BEH, 7.3)

    box(ax, 7.4, 5.35, 2.4, 0.85,
        "fluency head\n$s^{\\mathrm{flu}}_{i,j}=\\sigma(\\mathrm{MLP}_{\\mathrm{flu}}(z_{i,j}))$", FILL_BEH, 7.0)
    box(ax, 7.4, 3.95, 2.4, 0.85,
        "dynamic fusion gate\n$g_{i,j}=\\sigma(\\mathrm{MLP}_{\\mathrm{gate}}([z_{i,j}\\,\\|\\,f^{\\mathrm{exe}}_j]))$",
        FILL_FUSE, 7.0)
    box(ax, 7.4, 2.45, 2.4, 0.95,
        "residual correction\n$\\hat{y}_{i,j}=\\sigma(o^{\\mathrm{logit}}_{i,j}"
        "+\\gamma\\,g_{i,j}\\Delta_{i,j})$\n$\\Delta_{i,j}=(s^{\\mathrm{flu}}_{i,j}-0.5)\\times 4$",
        FILL_OUT, 7.0)
    box(ax, 10.15, 3.95, 1.7, 0.85,
        "prediction\n$\\hat{y}_{i,j}$", FILL_OUT, 7.3)

    box(ax, 7.4, 1.15, 2.4, 0.9,
        "auxiliary heads\n$\\mathrm{MLP}_{\\mathrm{EI}},\\mathrm{MLP}_{\\mathrm{EA}},"
        "\\mathrm{MLP}_{\\mathrm{RC}}$\nreconstruct $f^{\\mathrm{EI}},f^{\\mathrm{EA}},f^{\\mathrm{RC}}$",
        FILL_BEH, 7.0)
    box(ax, 10.15, 1.15, 1.7, 0.9, "multi-task\nlosses", FILL_FUSE, 7.3)

    arrow(ax, (2.65, 5.85), (3.35, 5.85))
    arrow(ax, (5.1, 5.35), (5.1, 4.75))
    arrow(ax, (2.65, 5.45), (3.35, 3.05), ls=(0, (4, 2)))
    arrow(ax, (4.4, 3.0), (4.4, 2.6), style="<|-|>")
    arrow(ax, (5.85, 2.91), (5.85, 2.07))
    arrow(ax, (6.85, 1.71), (7.4, 1.71))
    arrow(ax, (6.85, 1.9), (7.4, 5.4), ls=(0, (2, 2)))
    arrow(ax, (6.85, 1.9), (7.4, 4.1), ls=(0, (2, 2)))
    arrow(ax, (6.85, 5.75), (7.4, 5.75))
    arrow(ax, (6.85, 4.35), (7.4, 4.35))
    arrow(ax, (10.0, 5.75), (10.75, 5.75))
    arrow(ax, (10.0, 4.37), (7.4, 3.1))
    arrow(ax, (10.0, 2.92), (10.75, 2.92))
    arrow(ax, (2.65, 1.71), (7.4, 1.6), ls=(0, (2, 2)))

    box(ax, 4.85, 0.15, 7.0, 0.7,
        "$\\mathcal{L}_{\\mathrm{Total}}=(1-\\lambda)\\,\\mathcal{L}_y"
        "+\\lambda\\,(\\mathcal{L}_{\\mathrm{EI}}+\\mathcal{L}_{\\mathrm{EA}}"
        "+\\mathcal{L}_{\\mathrm{RC}})$", FILL_NEUTRAL, 7.3)

    path = os.path.join(OUT, "ours_flowchart.pdf")
    fig.savefig(path)
    preview = os.path.join(os.path.dirname(OUT), ".build", "fig3_preview.png")
    os.makedirs(os.path.dirname(preview), exist_ok=True)
    fig.savefig(preview, dpi=160)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    figure1()
    figure3()
