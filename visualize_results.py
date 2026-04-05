"""
WebPurifier – Report Visualizations
====================================
Generates publication-ready figures from the training results.

Outputs (saved to  figures/ ):
  1. model_comparison.png       – Grouped bar chart: F1, Precision, Recall per model
  2. smote_impact.png           – Side-by-side SMOTE vs no-SMOTE for each base model
  3. roc_auc_comparison.png     – Horizontal bar chart of ROC-AUC scores
  4. training_time.png          – Bar chart of training durations
  5. precision_recall_tradeoff.png – Scatter: Precision vs Recall per model
  6. dataset_distribution.png   – Class distribution pie + bar
  7. confusion_matrices.png     – Confusion matrices for all models
  8. cv_vs_test_f1.png          – Cross-validation F1 vs Test F1 comparison
  9. radar_chart.png            – Radar/spider chart of top 3 models

Usage:  python visualize_results.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import json

# ── Style (Light Theme) ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#f8f9fa",
    "axes.edgecolor":    "#cccccc",
    "axes.labelcolor":   "#333333",
    "text.color":        "#333333",
    "xtick.color":       "#555555",
    "ytick.color":       "#555555",
    "grid.color":        "#dddddd",
    "grid.alpha":        0.7,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "figure.titlesize":  16,
    "figure.titleweight":"bold",
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "#ffffff",
})

# colour palette (adjusted for light backgrounds)
COLORS = {
    "primary":   "#5b4fcf",
    "secondary": "#00a8a8",
    "accent":    "#e84393",
    "warning":   "#e67e22",
    "success":   "#27ae60",
    "danger":    "#c0392b",
    "info":      "#2980b9",
    "muted":     "#7f8c8d",
}

MODEL_COLORS = [
    "#5b4fcf",  # purple
    "#00a8a8",  # teal
    "#e84393",  # pink
    "#e67e22",  # orange
    "#27ae60",  # green
    "#2980b9",  # blue
    "#d35400",  # dark orange
    "#8e44ad",  # plum
    "#16a085",  # sea green
    "#c0392b",  # crimson
]

OUTPUT_DIR    = "figures"
RESULTS_CSV   = "comparison_results.csv"
DATASET_CSV   = "webpurifier_dataset.csv"


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV, engine="python", on_bad_lines="skip", quotechar='"', skipinitialspace=True)
    # Strip whitespace from column names and string columns
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    # Convert SMOTE column
    df["SMOTE"] = df["SMOTE"].map({"True": True, "False": False, True: True, False: False})
    return df


def add_value_labels(ax, bars, fmt=".4f", fontsize=8):
    """Add value labels on top of bars."""
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + 0.005,
            f"{height:{fmt}}", ha="center", va="bottom",
            fontsize=fontsize, color="#333333", fontweight="bold",
        )


# =====================================================================
# 1. MODEL COMPARISON – Grouped Bar Chart
# =====================================================================
def plot_model_comparison(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 6))

    models = df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    bars1 = ax.bar(x - width, df["Test_F1"],        width, label="F1-Score",  color=COLORS["primary"],   alpha=0.85, edgecolor="#999", linewidth=0.3)
    bars2 = ax.bar(x,         df["Test_Precision"],  width, label="Precision", color=COLORS["secondary"], alpha=0.85, edgecolor="#999", linewidth=0.3)
    bars3 = ax.bar(x + width, df["Test_Recall"],     width, label="Recall",    color=COLORS["accent"],    alpha=0.85, edgecolor="#999", linewidth=0.3)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: F1, Precision & Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    for bars in [bars1, bars2, bars3]:
        add_value_labels(ax, bars, fmt=".3f", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"))
    plt.close(fig)
    print("  ✓ model_comparison.png")


# =====================================================================
# 2. SMOTE IMPACT – Paired comparison
# =====================================================================
def plot_smote_impact(df: pd.DataFrame):
    # Find models that have both SMOTE and non-SMOTE versions
    base_models = []
    for name in df["Model"]:
        base = name.replace("_SMOTE", "")
        if base not in base_models:
            base_models.append(base)

    paired = []
    for base in base_models:
        no_smote = df[df["Model"] == base]
        with_smote = df[df["Model"] == f"{base}_SMOTE"]
        if len(no_smote) > 0 and len(with_smote) > 0:
            paired.append(base)

    if not paired:
        print("  ⚠ No SMOTE pairs found, skipping smote_impact.png")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["Test_F1", "Test_Precision", "Test_Recall"]
    metric_names = ["F1-Score", "Precision", "Recall"]
    bar_colors = [COLORS["info"], COLORS["accent"]]

    for ax, metric, mname in zip(axes, metrics, metric_names):
        x = np.arange(len(paired))
        width = 0.35

        no_smote_vals = [df[df["Model"] == m][metric].values[0] for m in paired]
        smote_vals    = [df[df["Model"] == f"{m}_SMOTE"][metric].values[0] for m in paired]

        b1 = ax.bar(x - width / 2, no_smote_vals, width, label="Without SMOTE", color=bar_colors[0], alpha=0.85, edgecolor="#999", linewidth=0.3)
        b2 = ax.bar(x + width / 2, smote_vals,    width, label="With SMOTE",    color=bar_colors[1], alpha=0.85, edgecolor="#999", linewidth=0.3)

        add_value_labels(ax, b1, fmt=".3f", fontsize=7)
        add_value_labels(ax, b2, fmt=".3f", fontsize=7)

        ax.set_title(mname)
        ax.set_xticks(x)
        ax.set_xticklabels(paired, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, framealpha=0.7)

    fig.suptitle("Impact of SMOTE on Model Performance", y=1.02, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "smote_impact.png"))
    plt.close(fig)
    print("  ✓ smote_impact.png")


# =====================================================================
# 3. ROC-AUC COMPARISON – Horizontal bars
# =====================================================================
def plot_roc_auc(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_df = df.sort_values("Test_ROC_AUC", ascending=True)
    y_pos = np.arange(len(sorted_df))

    # colour gradient based on AUC value
    norm = plt.Normalize(sorted_df["Test_ROC_AUC"].min(), sorted_df["Test_ROC_AUC"].max())
    colors = plt.cm.cool(norm(sorted_df["Test_ROC_AUC"]))

    bars = ax.barh(y_pos, sorted_df["Test_ROC_AUC"], color=colors, edgecolor="#999", linewidth=0.3, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_df["Model"], fontsize=10)
    ax.set_xlabel("ROC-AUC Score")
    ax.set_title("ROC-AUC Score Comparison")
    ax.set_xlim(0.9, 1.005)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, sorted_df["Test_ROC_AUC"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color="#333333", fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "roc_auc_comparison.png"))
    plt.close(fig)
    print("  ✓ roc_auc_comparison.png")


# =====================================================================
# 4. TRAINING TIME
# =====================================================================
def plot_training_time(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_df = df.sort_values("Train_Time_s", ascending=True)
    y_pos = np.arange(len(sorted_df))
    colors = [COLORS["success"] if t < 60 else COLORS["warning"] if t < 200 else COLORS["danger"]
              for t in sorted_df["Train_Time_s"]]

    bars = ax.barh(y_pos, sorted_df["Train_Time_s"], color=colors, edgecolor="#999", linewidth=0.3, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_df["Model"], fontsize=10)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Training Time Comparison")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, sorted_df["Train_Time_s"]):
        label = f"{val:.0f}s" if val < 60 else f"{val / 60:.1f}min"
        ax.text(val + 2, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9, color="#333333", fontweight="bold")

    # Add legend for time categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["success"], label="< 1 min"),
        Patch(facecolor=COLORS["warning"], label="1-3 min"),
        Patch(facecolor=COLORS["danger"],  label="> 3 min"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.7, fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "training_time.png"))
    plt.close(fig)
    print("  ✓ training_time.png")


# =====================================================================
# 5. PRECISION vs RECALL SCATTER
# =====================================================================
def plot_precision_recall_tradeoff(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (_, row) in enumerate(df.iterrows()):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        marker = "D" if row["SMOTE"] else "o"
        size = 120 + row["Test_F1"] * 200

        ax.scatter(row["Test_Recall"], row["Test_Precision"],
                   s=size, c=color, marker=marker, edgecolors="#666",
                   linewidth=1.2, alpha=0.9, zorder=5)

        # Label
        offset_x = 0.005 if row["Test_Recall"] < 0.92 else -0.005
        ha = "left" if row["Test_Recall"] < 0.92 else "right"
        ax.annotate(
            row["Model"], (row["Test_Recall"], row["Test_Precision"]),
            textcoords="offset points", xytext=(8, 5),
            fontsize=8, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=color, alpha=0.3),
        )

    # F1 iso-curves
    for f1_val in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
        recall_range = np.linspace(0.01, 1.0, 300)
        precision_curve = (f1_val * recall_range) / (2 * recall_range - f1_val)
        mask = (precision_curve > 0) & (precision_curve <= 1)
        ax.plot(recall_range[mask], precision_curve[mask],
                "--", color="#aaa", alpha=0.5, linewidth=0.8)
        # Label the curve
        idx = np.argmin(np.abs(recall_range - 0.98))
        if mask[idx] and precision_curve[idx] <= 1:
            ax.text(0.98, precision_curve[idx], f"F1={f1_val}",
                    fontsize=7, color="#999", ha="left", va="bottom")

    # Legend: marker shapes
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666",
               markersize=10, label="Without SMOTE"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#666",
               markersize=9, label="With SMOTE"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", framealpha=0.7, fontsize=10)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision–Recall Trade-off")
    ax.set_xlim(0.28, 1.02)
    ax.set_ylim(0.28, 1.02)
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "precision_recall_tradeoff.png"))
    plt.close(fig)
    print("  ✓ precision_recall_tradeoff.png")


# =====================================================================
# 6. DATASET CLASS DISTRIBUTION
# =====================================================================
def plot_dataset_distribution():
    if not os.path.exists(DATASET_CSV):
        print("  ⚠ Dataset CSV not found, skipping dataset_distribution.png")
        return

    data = pd.read_csv(DATASET_CSV)
    counts = data["label"].value_counts().sort_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        counts,
        labels=["Noise (0)", "Content (1)"],
        autopct="%1.1f%%",
        colors=[COLORS["danger"], COLORS["success"]],
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=12),
        pctdistance=0.6,
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")
        autotext.set_fontsize(13)
    ax1.set_title("Class Distribution", fontsize=14, fontweight="bold")

    # Bar chart
    bars = ax2.bar(
        ["Noise (0)", "Content (1)"], counts.values,
        color=[COLORS["danger"], COLORS["success"]],
        edgecolor="#999", linewidth=0.5, width=0.5,
    )
    for bar, val in zip(bars, counts.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                 f"{val:,}", ha="center", fontsize=12, fontweight="bold", color="#333333")
    ax2.set_ylabel("Number of Samples")
    ax2.set_title("Sample Count per Class", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Imbalance ratio annotation
    ratio = counts[0] / counts[1]
    fig.text(0.5, -0.02, f"Imbalance Ratio:  {ratio:.1f} : 1  (Noise : Content)",
             ha="center", fontsize=12, color=COLORS["danger"], fontweight="bold")

    fig.suptitle("WebPurifier Dataset — Class Imbalance", y=1.02, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "dataset_distribution.png"))
    plt.close(fig)
    print("  ✓ dataset_distribution.png")


# =====================================================================
# 7. CONFUSION MATRICES (from training.py output – reconstructed)
# =====================================================================
def plot_confusion_matrices(df: pd.DataFrame):
    """
    Reconstruct confusion matrices from precision/recall/support.
    Test set: 10973 total, 10355 noise, 618 content (from training output).
    """
    support_0 = 10355
    support_1 = 618

    n_models = len(df)
    cols = 4
    rows_grid = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows_grid, cols, figsize=(4 * cols, 4 * rows_grid))
    axes = axes.flatten() if n_models > 1 else [axes]

    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i]

        # Reconstruct: TP = recall * support_1, FP from precision
        tp = round(row["Test_Recall"] * support_1)
        fn = support_1 - tp
        # precision = tp / (tp + fp) → fp = tp/precision - tp
        fp = round(tp / row["Test_Precision"] - tp) if row["Test_Precision"] > 0 else 0
        tn = support_0 - fp

        cm = np.array([[tn, fp], [fn, tp]])

        # Plot
        im = ax.imshow(cm, cmap="YlGnBu", aspect="auto")

        for (r, c), val in np.ndenumerate(cm):
            text_color = "white" if val > cm.max() * 0.6 else "#222"
            ax.text(c, r, f"{val:,}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Noise", "Content"], fontsize=8)
        ax.set_yticklabels(["Noise", "Content"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)

        smote_tag = " ★" if row["SMOTE"] else ""
        ax.set_title(f"{row['Model']}{smote_tag}", fontsize=10, fontweight="bold", pad=6)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices  (★ = SMOTE applied)", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"))
    plt.close(fig)
    print("  ✓ confusion_matrices.png")


# =====================================================================
# 8. CV F1 vs TEST F1
# =====================================================================
def plot_cv_vs_test(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (_, row) in enumerate(df.iterrows()):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        marker = "D" if row["SMOTE"] else "o"
        ax.scatter(row["Best_CV_F1"], row["Test_F1"],
                   s=160, c=color, marker=marker, edgecolors="#666",
                   linewidth=1.2, alpha=0.9, zorder=5)
        ax.annotate(
            row["Model"], (row["Best_CV_F1"], row["Test_F1"]),
            textcoords="offset points", xytext=(8, 5),
            fontsize=8, color=color, fontweight="bold",
        )

    # Perfect correlation line
    lim_min = min(df["Best_CV_F1"].min(), df["Test_F1"].min()) - 0.02
    lim_max = max(df["Best_CV_F1"].max(), df["Test_F1"].max()) + 0.02
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="#888", alpha=0.5, label="Perfect agreement")

    ax.set_xlabel("Cross-Validation F1 (Best)", fontsize=12)
    ax.set_ylabel("Test F1", fontsize=12)
    ax.set_title("Cross-Validation F1 vs. Test F1")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.7)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cv_vs_test_f1.png"))
    plt.close(fig)
    print("  ✓ cv_vs_test_f1.png")


# =====================================================================
# 9. RADAR CHART – Top 3 models
# =====================================================================
def plot_radar_chart(df: pd.DataFrame):
    top3 = df.nlargest(3, "Test_F1")
    metrics = ["Test_Accuracy", "Test_F1", "Test_Precision", "Test_Recall", "Test_ROC_AUC"]
    metric_labels = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC"]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#f8f9fa")

    for i, (_, row) in enumerate(top3.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]
        color = MODEL_COLORS[i]

        ax.fill(angles, values, alpha=0.15, color=color)
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=row["Model"], markersize=6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0.8, 1.0)
    ax.set_yticks([0.85, 0.90, 0.95, 1.0])
    ax.set_yticklabels(["0.85", "0.90", "0.95", "1.00"], fontsize=8, color="#666")
    ax.set_rlabel_position(30)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), framealpha=0.7, fontsize=10)
    ax.set_title("Top 3 Models — Performance Radar", pad=25, fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "radar_chart.png"))
    plt.close(fig)
    print("  ✓ radar_chart.png")


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 55)
    print("  WebPurifier – Generating Report Figures")
    print("=" * 55 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_results()

    print(f"  Loaded {len(df)} model results from {RESULTS_CSV}\n")

    plot_model_comparison(df)
    plot_smote_impact(df)
    plot_roc_auc(df)
    plot_training_time(df)
    plot_precision_recall_tradeoff(df)
    plot_dataset_distribution()
    plot_confusion_matrices(df)
    plot_cv_vs_test(df)
    plot_radar_chart(df)

    print(f"\n  All figures saved to  {OUTPUT_DIR}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
