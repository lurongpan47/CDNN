#!/usr/bin/env python3
"""Generate a publication-style architecture figure for the CD-Transformer.

Panel (a): the full stack — embedding, N x [MLA attention + MoE FFN] with
residuals/RMSNorm, final norm, tied LM head + MTP heads. CD (block-circulant)
sublayers are colour-coded.
Panel (b): the CDLinear mechanism — a dense weight replaced by block-circulant
blocks, only the first column c stored (B x fewer params), each block
diagonalized by FFT with eigenvalues |FFT(c)|^2 and condition number kappa.

Writes cd_architecture.png (and .pdf).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

# palette
CD_F, CD_E, CD_T = "#d7f0e8", "#1d9e75", "#0f6e56"     # CD layers (teal)
ST_F, ST_E, ST_T = "#f1efe8", "#b4b2a9", "#5f5e5a"     # standard ops (gray)
HD_F, HD_E, HD_T = "#e6f1fb", "#378add", "#0c447c"     # heads (blue)
AC = "#444441"                                          # arrows


def box(ax, x, y, w, h, text, fc, ec, tc, fs=9, weight="normal", pad=0.02):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size=0.03",
                       linewidth=1.2, facecolor=fc, edgecolor=ec, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight=weight, zorder=3, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, ls="-", color=AC, lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle="-|>", mutation_scale=11, linewidth=lw,
                 color=color, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}", zorder=1))


fig = plt.figure(figsize=(12.4, 7.4), dpi=150)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.32], wspace=0.04)
axL = fig.add_subplot(gs[0]); axR = fig.add_subplot(gs[1])
for ax in (axL, axR):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

# ============================ (a) the stack ================================
axL.text(0.2, 9.7, "(a) CD-Transformer stack", fontsize=12, weight="bold", color="#111827")
cx, w = 2.6, 4.8
def cbox(y, h, t, kind, fs=9, weight="normal"):
    f, e, tc = {"cd": (CD_F, CD_E, CD_T), "st": (ST_F, ST_E, ST_T),
                "hd": (HD_F, HD_E, HD_T)}[kind]
    box(axL, cx, y, w, h, t, f, e, tc, fs=fs, weight=weight)

cbox(0.5, 0.6, "input tokens", "st")
cbox(1.4, 0.6, "token embedding  (tied)", "st")
# repeated block container
axL.add_patch(Rectangle((1.7, 2.35), 6.6, 5.2, fill=False, linewidth=1.1,
              edgecolor="#9aa0a6", linestyle=(0, (5, 3)), zorder=0))
axL.text(8.45, 7.45, "x 12\nlayers", fontsize=9, color="#5f5e5a", ha="center", va="top", weight="bold")
cbox(2.55, 0.5, "RMSNorm", "st", fs=8)
cbox(3.2, 0.78, "MLA self-attention\n(CD q / kv / out, low-rank KV)", "cd", fs=8.3)
cbox(4.25, 0.45, "+  residual", "st", fs=8)
cbox(5.0, 0.5, "RMSNorm", "st", fs=8)
cbox(5.65, 0.78, "MoE FFN\n(router -> top-4 of 16 CD experts)", "cd", fs=8.3)
cbox(6.7, 0.45, "+  residual", "st", fs=8)
cbox(7.75, 0.55, "final RMSNorm", "st", fs=8.5)
# heads
box(axL, 1.5, 8.6, 3.0, 0.62, "LM head\n(tied embed)", HD_F, HD_E, HD_T, fs=8.3)
box(axL, 5.0, 8.6, 3.4, 0.62, "MTP heads\n(multi-token predict)", HD_F, HD_E, HD_T, fs=8.3)
# arrows up the stack
xs = cx + w / 2
for y1, y2 in [(1.1, 1.4), (2.0, 2.55), (3.05, 3.2), (3.98, 4.25),
               (4.7, 5.0), (6.15, 6.7)]:
    arrow(axL, xs, y1, xs, y2)
arrow(axL, xs, 4.7, xs, 5.0)
arrow(axL, xs, 7.15, xs, 7.75)
arrow(axL, xs, 8.3, 3.0, 8.6); arrow(axL, xs, 8.3, 6.7, 8.6)
# residual skip hints
axL.add_patch(FancyArrowPatch((cx, 2.8), (cx, 4.475), arrowstyle="-|>",
              mutation_scale=9, lw=1.0, color="#1d9e75",
              connectionstyle="arc3,rad=-0.45", linestyle=(0, (3, 2)), zorder=1))
axL.add_patch(FancyArrowPatch((cx, 5.25), (cx, 6.925), arrowstyle="-|>",
              mutation_scale=9, lw=1.0, color="#1d9e75",
              connectionstyle="arc3,rad=-0.45", linestyle=(0, (3, 2)), zorder=1))

# legend
leg = [Line2D([0], [0], marker="s", color="w", markerfacecolor=CD_F,
              markeredgecolor=CD_E, markersize=12, label="CD (block-circulant) layer"),
       Line2D([0], [0], marker="s", color="w", markerfacecolor=ST_F,
              markeredgecolor=ST_E, markersize=12, label="standard op"),
       Line2D([0], [0], marker="s", color="w", markerfacecolor=HD_F,
              markeredgecolor=HD_E, markersize=12, label="output head")]
axL.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.0, -0.02),
           fontsize=7.6, frameon=False, ncol=1)

# ===================== (b) CDLinear mechanism ==============================
axR.text(0.1, 9.7, "(b) CDLinear: block-circulant weight", fontsize=12, weight="bold", color="#111827")

# dense weight grid
def grid(ax, x, y, s, n, fc, ec, lw=0.6):
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x + j * s, y + (n - 1 - i) * s), s, s,
                         facecolor=fc, edgecolor=ec, linewidth=lw))
gs_ = 0.34
grid(axR, 0.5, 7.0, gs_, 6, "#e9e7df", "#c9c7bd")
axR.text(0.5 + 3 * gs_, 6.75, "dense W\n(n_out x n_in)", ha="center", va="top", fontsize=8.2, color="#5f5e5a")
arrow(axR, 2.75, 8.05, 3.45, 8.05, lw=1.4)
axR.text(3.05, 8.5, "block-\ncirculant", ha="center", fontsize=7.6, color="#0f6e56", weight="bold")

# block-circulant grid (B x B blocks)
bx, by, B, bs = 4.2, 7.0, 5, 0.30
for bi in range(2):
    for bj in range(2):
        ox, oy = bx + bj * (B * bs + 0.12), by + (1 - bi) * (B * bs + 0.12) - 0.6
        for i in range(B):
            for j in range(B):
                shade = "#d7f0e8" if (i - j) % B == 0 else "#eef6f2"
                axR.add_patch(Rectangle((ox + j * bs, oy + (B - 1 - i) * bs), bs, bs,
                              facecolor=shade, edgecolor="#1d9e75", linewidth=0.5))
axR.text(6.0, 6.18, "each B x B block is circulant", ha="center", fontsize=8, color="#0f6e56")

# first-column c callout
axR.annotate("", xy=(4.2, 7.0 + 4 * bs * 0 + 0.0), xytext=(4.2, 7.0))
cy = 7.0
for i in range(B):
    axR.add_patch(Rectangle((3.62, cy + (B - 1 - i) * bs), bs, bs,
                  facecolor="#1d9e75", edgecolor="#0f6e56", linewidth=0.6))
axR.text(3.77, cy + B * bs + 0.12, "c", fontsize=10, color="#0f6e56", weight="bold", ha="center")
axR.text(3.77, cy - 0.18, "store only\nfirst column", ha="center", va="top", fontsize=7.4, color="#0f6e56")

box(axR, 0.5, 5.05, 9.0, 0.62,
    "store only c  ->  B = 5x fewer parameters   (4.22e8 params  ~=  1.84e9-param dense model)",
    CD_F, CD_E, CD_T, fs=8.6, weight="bold")

# FFT diagonalization + spectrum
arrow(axR, 5.0, 4.95, 5.0, 4.35, lw=1.4)
axR.text(5.25, 4.62, "FFT diagonalizes each circulant block", fontsize=8.2, color="#111827", va="center")

# spectrum bar plot inset
sx, sy, sw, sh = 0.7, 1.5, 4.0, 2.4
axR.add_patch(Rectangle((sx, sy), sw, sh, facecolor="white", edgecolor="#c9c7bd", linewidth=0.8))
np.random.seed(3)
spec = np.array([1.0, 0.62, 0.30, 0.11, 0.015])
bw = sw / (len(spec) + 1)
for k, v in enumerate(spec):
    axR.add_patch(Rectangle((sx + (k + 0.5) * bw, sy + 0.15), bw * 0.7, v * (sh - 0.5),
                  facecolor="#7c3aed", edgecolor="none"))
axR.text(sx + sw / 2, sy + sh - 0.18, "eigenvalues  s_k = |FFT(c)|^2",
         ha="center", va="top", fontsize=8, color="#3c3489", weight="bold")
axR.annotate("max", xy=(sx + 0.5 * bw + bw * 0.35, sy + 0.15 + spec[0] * (sh - 0.5)),
             xytext=(sx + 0.2, sy + sh - 0.55), fontsize=7, color="#5f5e5a",
             arrowprops=dict(arrowstyle="->", color="#888780", lw=0.7))
axR.annotate("min ~ 0", xy=(sx + 4.5 * bw + bw * 0.35, sy + 0.15 + spec[4] * (sh - 0.5)),
             xytext=(sx + sw - 1.3, sy + 0.9), fontsize=7, color="#5f5e5a",
             arrowprops=dict(arrowstyle="->", color="#888780", lw=0.7))

box(axR, 5.1, 2.5, 4.4, 1.35,
    "kappa = max(s) / min(s)\n\nTheorem 2: flatten the spectrum\n-> kappa -> 1  (well-conditioned)",
    "#f3f0fb", "#7c3aed", "#3c3489", fs=8.6)
box(axR, 5.1, 1.5, 4.4, 0.78,
    "forward: build circulant from c\n-> BF16 matmul (tensor cores)",
    CD_F, CD_E, CD_T, fs=8.2)

fig.suptitle("Figure 1.  CD-Transformer architecture: a DeepSeek-V3-style MLA + MoE + MTP backbone "
             "whose linear projections are block-circulant CDLinear layers.",
             fontsize=10.5, y=0.025, color="#374151")
fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.07)
fig.savefig("cd_architecture.png", dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig("cd_architecture.pdf", bbox_inches="tight", facecolor="white")
print("wrote cd_architecture.png / .pdf")
