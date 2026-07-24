#!/usr/bin/env python3
"""
Stress-test divergence figure for IEEE 33-bus TP6.

Panel (a): feasibility rate versus load scaling.
Panel (b): operating-cost reduction relative to MLP.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
    }
)

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = OUTPUT_DIR / "stress_divergence_ieee33_tp6.pdf"
PNG_PATH = OUTPUT_DIR / "stress_divergence_ieee33_tp6.png"


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

lam = np.array([1.0, 1.1, 1.2, 1.3, 1.4])

feas = {
    "MLP": (
        [0.9844, 0.9037, 0.8015, 0.6686, 0.5674],
        [0.0060, 0.0044, 0.0222, 0.0348, 0.0178],
    ),
    "GCN": (
        [0.9875, 0.9191, 0.8278, 0.6999, 0.6085],
        [0.0060, 0.0303, 0.0287, 0.0243, 0.0225],
    ),
    "GAT": (
        [0.9842, 0.9112, 0.8085, 0.6860, 0.6081],
        [0.0060, 0.0145, 0.0153, 0.0093, 0.0168],
    ),
    "TAGConv": (
        [0.9875, 0.9157, 0.8255, 0.7047, 0.6160],
        [0.0060, 0.0124, 0.0170, 0.0155, 0.0079],
    ),
}

cost = {
    "MLP": (
        [606.08, 686.85, 772.62, 862.96, 952.06],
        [12.01, 11.92, 14.54, 18.26, 17.52],
    ),
    "GCN": (
        [602.67, 682.41, 763.26, 849.54, 939.80],
        [11.74, 11.52, 11.89, 11.48, 8.64],
    ),
    "GAT": (
        [604.84, 682.15, 762.63, 848.36, 937.03],
        [6.80, 7.04, 6.29, 5.75, 4.90],
    ),
    "TAGConv": (
        [597.65, 676.90, 758.43, 845.24, 935.07],
        [7.45, 7.95, 8.54, 9.06, 9.73],
    ),
}

colors = {
    "MLP": "#7F7F7F",
    "GCN": "#2E75B6",
    "GAT": "#8064A2",
    "TAGConv": "#C0504D",
}

markers = {
    "MLP": "s",
    "GCN": "o",
    "GAT": "^",
    "TAGConv": "D",
}

model_order = ["MLP", "GCN", "GAT", "TAGConv"]


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(7.0, 2.7),
)


# Panel (a): feasibility
for model in model_order:
    mean = np.asarray(feas[model][0], dtype=float)
    std = np.asarray(feas[model][1], dtype=float)

    ax1.plot(
        lam,
        mean,
        color=colors[model],
        marker=markers[model],
        markersize=4,
        linewidth=1.4,
        label=model,
    )

    ax1.fill_between(
        lam,
        mean - std,
        mean + std,
        color=colors[model],
        alpha=0.15,
        linewidth=0,
    )

ax1.set_xlabel(r"Load scaling factor $\lambda$")
ax1.set_ylabel("Feasible rate")
ax1.set_xticks(lam)
ax1.set_xlim(0.98, 1.42)
ax1.set_ylim(0.50, 1.02)
ax1.grid(alpha=0.3, linewidth=0.5)
ax1.legend(
    fontsize=7.5,
    frameon=False,
    loc="lower left",
)
ax1.set_title(
    "(a) Operational feasibility",
    fontsize=9,
)


# Panel (b): cost reduction relative to MLP
mlp_cost = np.asarray(cost["MLP"][0], dtype=float)

for model in ["GCN", "GAT", "TAGConv"]:
    model_cost = np.asarray(cost[model][0], dtype=float)

    reduction = 100.0 * (
        mlp_cost - model_cost
    ) / mlp_cost

    ax2.plot(
        lam,
        reduction,
        color=colors[model],
        marker=markers[model],
        markersize=4,
        linewidth=1.4,
        label=model,
    )

ax2.axhline(
    0.0,
    color="#999999",
    linewidth=0.8,
    linestyle=":",
)

ax2.set_xlabel(r"Load scaling factor $\lambda$")
ax2.set_ylabel("Cost reduction vs. MLP (%)")
ax2.set_xticks(lam)
ax2.set_xlim(0.98, 1.42)
ax2.grid(alpha=0.3, linewidth=0.5)
ax2.legend(
    fontsize=7.5,
    frameon=False,
    loc="upper left",
)
ax2.set_title(
    "(b) Economic advantage over MLP",
    fontsize=9,
)


fig.tight_layout(pad=0.5)

fig.savefig(
    PDF_PATH,
    bbox_inches="tight",
    pad_inches=0.03,
)

fig.savefig(
    PNG_PATH,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.03,
)

plt.close(fig)

print("=" * 78)
print("FIGURE GENERATED")
print("=" * 78)
print(f"PDF: {PDF_PATH.resolve()}")
print(f"PNG: {PNG_PATH.resolve()}")
print("=" * 78)