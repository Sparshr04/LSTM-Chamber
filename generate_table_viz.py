"""
Generate Summary Visualization for Training Loop Refinement Results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read results
df = pd.read_csv("experiment_results.csv")

# Create figure with comparison table
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis("tight")
ax.axis("off")

# Prepare table data
table_data = []
for idx, row in df.iterrows():
    table_data.append(
        [
            row["Model"],
            int(row["Layers"]),
            f"{row['Learning Rate']:.6f}"
            if row["Learning Rate"] >= 0.0001
            else f"{row['Learning Rate']:.0e}",
            f"{row['Test RMSE']:.6f}",
            f"{row['Test MSE']:.6f}",
            f"{row['Test MAE']:.6f}",
            f"{row['R² Score']:.4f}",
            f"{int(row['Parameters']):,}",
        ]
    )

# Column headers
columns = [
    "Model",
    "Layers",
    "Learning\nRate",
    "Test\nRMSE ↓",
    "Test\nMSE ↓",
    "Test\nMAE ↓",
    "R²\nScore ↑",
    "Parameters",
]

# Create table
table = ax.table(
    cellText=table_data,
    colLabels=columns,
    cellLoc="center",
    loc="center",
    colWidths=[0.10, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12],
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Header styling
for i in range(len(columns)):
    cell = table[(0, i)]
    cell.set_facecolor("#4CAF50")
    cell.set_text_props(weight="bold", color="white")

# Highlight best model (Model-2)
best_idx = df["Test RMSE"].idxmin() + 1  # +1 because row 0 is header
for j in range(len(columns)):
    cell = table[(best_idx, j)]
    cell.set_facecolor("#FFF9C4")
    cell.set_text_props(weight="bold")

# Highlight Model-7 (our focus)
model_7_idx = df[df["Model"] == "Model-7"].index[0] + 1
for j in range(len(columns)):
    cell = table[(model_7_idx, j)]
    cell.set_facecolor("#E1F5FE")

# Alternate row colors for readability
for i in range(1, len(df) + 1):
    if i not in [best_idx, model_7_idx]:
        for j in range(len(columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor("#F5F5F5")

plt.title(
    "Table 2: Updated Performance Metrics - After Training Loop Refinement\n"
    + "Yellow: Best Model | Blue: Model-7 (Fixed Underfitting)",
    fontsize=14,
    fontweight="bold",
    pad=20,
)

plt.savefig("04_Updated_Table2_Comparison.png", dpi=300, bbox_inches="tight")
print("✓ Saved: 04_Updated_Table2_Comparison.png")
plt.show()
