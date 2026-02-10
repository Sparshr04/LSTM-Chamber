"""
Performance Gap Comparative Analysis
=====================================
Creates a clean bar chart comparing Model-2, Model-6, and Model-7
using ACTUAL data from experiment_results.csv
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Data from actual experiment results
models = ["Model-2\n(Proposed)", "Model-7\n(Paper)", "Model-6\n(6 Layers Deep)"]
r2_scores = [0.501, 0.237, 0.191]  # Actual values from experiment_results.csv
colors = ["#2ecc71", "#95a5a6", "#95a5a6"]  # Green for winner, grey for others

# Create bar chart
bars = ax.bar(
    models, r2_scores, color=colors, edgecolor="black", linewidth=2, alpha=0.85
)

# Add value labels on top of bars
for i, (bar, score) in enumerate(zip(bars, r2_scores)):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.02,
        f"R² = {score:.3f}",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

# Add performance improvement annotation
improvement = (r2_scores[0] / r2_scores[1]) if r2_scores[1] > 0 else 0
ax.annotate(
    f"{improvement:.1f}x Better\nAccuracy",
    xy=(0, r2_scores[0]),
    xytext=(0.5, r2_scores[0] + 0.08),
    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2.5),
    fontsize=13,
    fontweight="bold",
    color="darkgreen",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
)

# Add horizontal line for Model-2 benchmark
ax.axhline(y=r2_scores[0], color="#2ecc71", linestyle="--", linewidth=2, alpha=0.5)

# Styling
ax.set_ylabel("R² Score (Higher is Better)", fontsize=14, fontweight="bold")
ax.set_xlabel("Model Configuration", fontsize=14, fontweight="bold")
ax.set_title(
    "Performance Gap: Proposed Model vs. Research Paper Model\n"
    + "Pune Weather Dataset (2008-2022)",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

# Set y-axis limits
ax.set_ylim(0, 0.65)

# Add grid
ax.yaxis.grid(True, linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

# Add summary text box
summary_text = (
    "Key Findings:\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    f"✓ Model-2: R² = {r2_scores[0]:.3f} (64 epochs)\n"
    f"✗ Model-7: R² = {r2_scores[1]:.3f} (100 epochs)\n"
    f"✗ Model-6: R² = {r2_scores[2]:.3f} (43 epochs)\n\n"
    "Model-2 achieved 2.1x better\n"
    "performance than the paper's\n"
    "recommended model."
)

ax.text(
    0.98,
    0.65,
    summary_text,
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9, pad=1),
    family="monospace",
)

plt.tight_layout()
plt.savefig("plots/performance_gap_comparison.png", dpi=300, bbox_inches="tight")
print("✓ Created: plots/performance_gap_comparison.png")
plt.show()
