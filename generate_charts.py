"""
================================================================================
 generate_charts.py — Yellow MAIze Chart Generator
================================================================================
 PURPOSE:
   Reads all training CSV logs produced by train_bouncer.py, train_teacher.py,
   and train_student.py and generates publication-quality PNG comparison charts.
   Run this any time after training without re-running the training scripts.

 USAGE:
   python generate_charts.py              # generate all charts
   python generate_charts.py --bouncer    # bouncer charts only
   python generate_charts.py --teacher    # teacher charts only
   python generate_charts.py --student    # student charts only

 INPUT CSVs (all under logs/):
   Bouncer:
     logs/bouncer_comparison.csv
     logs/bouncer_{variant}_metrics.csv   (per-epoch training curves)

   Teacher:
     logs/teacher_comparison.csv
     logs/teacher_{variant}_metrics.csv   (per-epoch training curves)
     logs/teacher_test_metrics.csv        (held-out test evaluation)

   Student:
     logs/student_comparison_stage1.csv   (encoder ablation)
     logs/student_comparison_stage2.csv   (mode ablation)
     logs/student_{enc}_{mode}_metrics.csv (per-epoch training curves)
     logs/student_test_metrics_{enc}_{mode}.csv (per-variant test results)
     logs/student_confusion_{enc}_{mode}.csv    (3×3 confusion matrices)

 OUTPUT PNGs (all under reports/charts/):
   Bouncer:
     bouncer_comparison_bar.png           — F1 / Specificity / Recall / ROC-AUC
     bouncer_training_curves_{variant}.png — per-epoch loss, F1, specificity
     bouncer_confusion_matrix_{variant}.png — TP/FP/TN/FN heatmap

   Teacher:
     teacher_comparison_bar.png           — Best Dice / CPU latency bar chart
     teacher_training_curves_{variant}.png — per-epoch loss, Dice, IoU, Recall

   Student:
     student_stage1_comparison_bar.png    — encoder ablation multi-metric bar
     student_stage2_comparison_bar.png    — mode ablation multi-metric bar
     student_training_curves_{enc}_{mode}.png — per-epoch curves (both phases)
     student_confusion_{enc}_{mode}.png   — 3×3 confusion matrix heatmap
     student_radar_{enc}_{mode}.png       — radar/spider chart of test metrics
================================================================================
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════════════════════
# PATHS  (mirror config.py — no import dependency on project modules)
# ══════════════════════════════════════════════════════════════════════════════

ROOT      = Path(__file__).parent
LOGS_DIR  = ROOT / "logs"
CHART_DIR = ROOT / "reports" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# Known Bouncer variants for per-epoch curve lookup
BOUNCER_VARIANTS = ["mobilenet_v2", "mobilenet_v3_large", "edgevit_xxs"]

# Known Teacher variants
TEACHER_VARIANTS = ["resnet50", "efficientnet-b2", "mit_b2", "deeplabv3plus-eb2"]

# ══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE  (consistent across all charts)
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {
    # Variants
    "gabor_lbp":          "#6B7280",   # neutral grey  — traditional baseline
    "mobilenet_v2":       "#3B82F6",   # blue          — thesis primary arch
    "mobilenet_v3_large": "#10B981",   # emerald green — deployed model
    "edgevit_xxs":        "#F59E0B",   # amber         — ViT comparison
    "patchcore":          "#8B5CF6",   # violet        — anomaly detector
    "resnet50":           "#6B7280",   # grey          — CNN baseline
    "efficientnet-b2":    "#10B981",   # emerald       — recommended
    "mit_b2":             "#F59E0B",   # amber         — ViT encoder
    "deeplabv3plus-eb2":  "#EF4444",   # red           — decoder comparison
    # Student encoders
    "mobilenet_v2_cbam":  "#8B5CF6",   # violet        — expected winner
    "mobilenet_v3_small": "#EC4899",   # pink          — ultra-compact
    "efficientnet_b0":    "#F97316",   # orange        — compound scaling
    "efficientnet_b0_cbam":"#14B8A6",  # teal          — CBAM generalization
    # Factory modes
    "mode_a":             "#6B7280",
    "mode_b":             "#3B82F6",
    "mode_c":             "#10B981",
    "mode_d":             "#F59E0B",
}

FALLBACK_COLORS = [
    "#3B82F6","#10B981","#F59E0B","#EF4444",
    "#8B5CF6","#EC4899","#F97316","#14B8A6","#6B7280",
]

BG_COLOR   = "#0F172A"   # dark navy background
GRID_COLOR = "#1E293B"   # subtle grid lines
TEXT_COLOR = "#F1F5F9"   # near-white text
ACCENT     = "#34D399"   # teal accent for highlights


def get_color(label: str, idx: int = 0) -> str:
    return PALETTE.get(str(label).lower(), FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


def apply_dark_style(fig, axes=None):
    """Apply consistent dark theme to a figure and its axes."""
    fig.patch.set_facecolor(BG_COLOR)
    if axes is None:
        return
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.6)


def save_chart(fig, name: str):
    path = CHART_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    print(f"  ✓  {path.relative_to(ROOT)}")


def safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ══════════════════════════════════════════════════════════════════════════════
# BOUNCER CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def chart_bouncer_comparison():
    """
    Grouped bar chart: F1 / Specificity / Maize Recall / ROC-AUC
    across all Bouncer variants from bouncer_comparison.csv.
    """
    csv_path = LOGS_DIR / "bouncer_comparison.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path.name} not found.")
        return

    df = pd.read_csv(csv_path)
    # Numeric columns only — gabor_lbp may have 'N/A' strings for some fields
    metrics = ["best_f1", "specificity", "maize_recall", "roc_auc"]
    labels  = ["Maize F1", "Specificity", "Maize Recall", "ROC-AUC"]
    variants = df["variant"].tolist()
    n_variants = len(variants)
    n_metrics  = len(metrics)

    x      = np.arange(n_variants)
    width  = 0.18
    offsets= np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

    fig, ax = plt.subplots(figsize=(max(9, n_variants * 2.2), 5.5))
    apply_dark_style(fig, ax)

    metric_colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444"]

    for mi, (col, label, color) in enumerate(zip(metrics, labels, metric_colors)):
        vals = [safe_float(v) for v in df[col]]
        bars = ax.bar(x + offsets[mi], vals, width=width,
                      label=label, color=color, alpha=0.88,
                      zorder=3, edgecolor=BG_COLOR, linewidth=0.6)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=7.5, color=TEXT_COLOR, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", "\n") for v in variants],
                       color=TEXT_COLOR, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", color=TEXT_COLOR)
    ax.set_title("Bouncer — Model Comparison\nF1 · Specificity · Maize Recall · ROC-AUC",
                 color=TEXT_COLOR, fontsize=12, pad=12)

    # Mark deployed model
    if "mobilenet_v3_large" in variants:
        idx = variants.index("mobilenet_v3_large")
        ax.axvspan(idx - 0.42, idx + 0.42, alpha=0.07, color=ACCENT, zorder=1)
        ax.text(idx, 1.06, "▲ deployed", ha="center", va="bottom",
                fontsize=8, color=ACCENT)

    legend = ax.legend(loc="upper right", framealpha=0.15,
                       labelcolor=TEXT_COLOR, fontsize=9,
                       facecolor=GRID_COLOR, edgecolor="#334155")
    fig.tight_layout()
    save_chart(fig, "bouncer_comparison_bar.png")


def chart_bouncer_training_curves():
    """Per-epoch training curves (loss, F1, specificity) for each neural variant."""
    for variant in BOUNCER_VARIANTS:
        csv_path = LOGS_DIR / f"bouncer_{variant}_metrics.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        apply_dark_style(fig, axes)

        color = get_color(variant)

        # Loss
        ax = axes[0]
        ax.plot(df["epoch"], df["train_loss"], color="#94A3B8",
                linewidth=1.4, linestyle="--", label="train loss", alpha=0.8)
        ax.plot(df["epoch"], df["val_loss"],   color=color,
                linewidth=2.0, label="val loss")
        ax.set_title("Loss", color=TEXT_COLOR)
        ax.set_xlabel("Epoch", color=TEXT_COLOR)
        ax.legend(labelcolor=TEXT_COLOR, fontsize=8,
                  facecolor=GRID_COLOR, edgecolor="#334155", framealpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Maize F1
        ax = axes[1]
        ax.plot(df["epoch"], df["maize_f1"], color="#3B82F6",
                linewidth=2.0, label="maize F1")
        if "accuracy" in df.columns:
            ax.plot(df["epoch"], df["accuracy"], color="#94A3B8",
                    linewidth=1.4, linestyle="--", label="accuracy", alpha=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_title("Maize F1 & Accuracy", color=TEXT_COLOR)
        ax.set_xlabel("Epoch", color=TEXT_COLOR)
        ax.legend(labelcolor=TEXT_COLOR, fontsize=8,
                  facecolor=GRID_COLOR, edgecolor="#334155", framealpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Specificity + Precision + Recall
        ax = axes[2]
        ax.plot(df["epoch"], df["specificity"],  color="#10B981",
                linewidth=2.0, label="specificity")
        ax.plot(df["epoch"], df["maize_rec"],    color="#F59E0B",
                linewidth=1.6, linestyle="-.", label="maize recall")
        ax.plot(df["epoch"], df["maize_prec"],   color="#EF4444",
                linewidth=1.6, linestyle=":",   label="maize prec")
        ax.set_ylim(0, 1.05)
        ax.set_title("Specificity / Recall / Precision", color=TEXT_COLOR)
        ax.set_xlabel("Epoch", color=TEXT_COLOR)
        ax.legend(labelcolor=TEXT_COLOR, fontsize=8,
                  facecolor=GRID_COLOR, edgecolor="#334155", framealpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.suptitle(f"Bouncer Training Curves — {variant}",
                     color=TEXT_COLOR, fontsize=13, y=1.01)
        fig.tight_layout()
        save_chart(fig, f"bouncer_training_curves_{variant}.png")


def chart_bouncer_confusion():
    """
    TP/FP/TN/FN visualised as a 2×2 confusion matrix heatmap per variant,
    sourced from the comparison CSV summary row.
    """
    csv_path = LOGS_DIR / "bouncer_comparison.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    needed = {"TP", "FP", "TN", "FN"}
    if not needed.issubset(df.columns):
        print("  [SKIP] bouncer_confusion — TP/FP/TN/FN columns missing.")
        return

    neural = df[df["variant"] != "gabor_lbp"]
    if neural.empty:
        neural = df

    n = len(neural)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2))
    if n == 1:
        axes = [axes]
    apply_dark_style(fig, axes)

    class_names = ["not_maize\n(neg)", "maize\n(pos)"]

    for ax, (_, row) in zip(axes, neural.iterrows()):
        tp = safe_float(row.get("TP", 0))
        fp = safe_float(row.get("FP", 0))
        fn = safe_float(row.get("FN", 0))
        tn = safe_float(row.get("TN", 0))

        total = tp + fp + fn + tn or 1
        mat   = np.array([[tn / total, fp / total],
                           [fn / total, tp / total]])
        raw   = np.array([[int(tn), int(fp)],
                           [int(fn), int(tp)]])

        im = ax.imshow(mat, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(2):
            for j in range(2):
                label = ["TN", "FP", "FN", "TP"][i * 2 + j]
                ax.text(j, i, f"{label}\n{raw[i,j]:,}\n({mat[i,j]:.2%})",
                        ha="center", va="center", fontsize=9,
                        color="black" if mat[i, j] > 0.45 else TEXT_COLOR,
                        fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: not_maize", "Pred: maize"],
                           color=TEXT_COLOR, fontsize=8)
        ax.set_yticklabels(["True: not_maize", "True: maize"],
                           color=TEXT_COLOR, fontsize=8)
        ax.set_title(row["variant"].replace("_", " "), color=TEXT_COLOR)

    fig.suptitle("Bouncer — Confusion Matrix (val split)",
                 color=TEXT_COLOR, fontsize=13, y=1.02)
    fig.tight_layout()
    save_chart(fig, "bouncer_confusion_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
# TEACHER CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def chart_teacher_comparison():
    """
    Side-by-side bar chart: Best Dice (primary) + CPU latency for all Teacher variants.
    """
    csv_path = LOGS_DIR / "teacher_comparison.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path.name} not found.")
        return

    df       = pd.read_csv(csv_path)
    variants = df["variant"].tolist()
    n        = len(variants)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    apply_dark_style(fig, [ax1, ax2])

    colors = [get_color(v, i) for i, v in enumerate(variants)]
    x      = np.arange(n)

    # Best Dice
    dice_vals = [safe_float(v) for v in df["best_dice"]]
    bars = ax1.bar(x, dice_vals, color=colors, alpha=0.88,
                   edgecolor=BG_COLOR, linewidth=0.8, zorder=3)
    for bar, val in zip(bars, dice_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom",
                 fontsize=9, color=TEXT_COLOR)
    ax1.set_xticks(x)
    ax1.set_xticklabels([v.replace("_", "\n").replace("-", "\n") for v in variants],
                        color=TEXT_COLOR, fontsize=8.5)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Dice Score", color=TEXT_COLOR)
    ax1.set_title("Best Validation Dice", color=TEXT_COLOR)

    # Highlight winner
    best_idx = int(np.argmax(dice_vals))
    ax1.get_children()[best_idx].set_edgecolor(ACCENT)
    ax1.get_children()[best_idx].set_linewidth(2.2)
    ax1.text(best_idx, dice_vals[best_idx] + 0.025,
             "★ best", ha="center", fontsize=8, color=ACCENT)

    # CPU latency
    if "lat_cpu_ms" in df.columns:
        lat_vals = [safe_float(v) for v in df["lat_cpu_ms"]]
        bars2 = ax2.bar(x, lat_vals, color=colors, alpha=0.88,
                        edgecolor=BG_COLOR, linewidth=0.8, zorder=3)
        for bar, val in zip(bars2, lat_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f"{val:.1f} ms", ha="center", va="bottom",
                     fontsize=9, color=TEXT_COLOR)
        ax2.set_xticks(x)
        ax2.set_xticklabels([v.replace("_", "\n").replace("-", "\n") for v in variants],
                            color=TEXT_COLOR, fontsize=8.5)
        ax2.set_ylabel("CPU Latency (ms/image @ 512px)", color=TEXT_COLOR)
        ax2.set_title("CPU Inference Latency", color=TEXT_COLOR)
    else:
        ax2.text(0.5, 0.5, "lat_cpu_ms\nnot available",
                 ha="center", va="center", color=TEXT_COLOR,
                 transform=ax2.transAxes, fontsize=11)

    fig.suptitle("Teacher Model Comparison", color=TEXT_COLOR, fontsize=13, y=1.02)
    fig.tight_layout()
    save_chart(fig, "teacher_comparison_bar.png")


def chart_teacher_training_curves():
    """Per-epoch training curves for each Teacher variant."""
    for variant in TEACHER_VARIANTS:
        csv_path = LOGS_DIR / f"teacher_{variant}_metrics.csv"
        if not csv_path.exists():
            continue

        df    = pd.read_csv(csv_path)
        color = get_color(variant)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        apply_dark_style(fig, axes)

        # Loss
        ax = axes[0]
        ax.plot(df["epoch"], df["train_loss"], color="#94A3B8",
                linewidth=1.4, linestyle="--", label="train loss", alpha=0.8)
        ax.plot(df["epoch"], df["val_loss"],   color=color,
                linewidth=2.0, label="val loss")
        ax.set_title("Loss", color=TEXT_COLOR)
        ax.set_xlabel("Epoch", color=TEXT_COLOR)
        ax.legend(labelcolor=TEXT_COLOR, fontsize=8,
                  facecolor=GRID_COLOR, edgecolor="#334155", framealpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Dice + IoU
        ax = axes[1]
        ax.plot(df["epoch"], df["val_dice"], color="#3B82F6",
                linewidth=2.0, label="val Dice")
        ax.plot(df["epoch"], df["val_iou"],  color="#10B981",
                linewidth=1.6, linestyle="-.", label="val IoU")
        ax.set_ylim(0, 1.05)
        ax.set_title("Dice & IoU", color=TEXT_COLOR)
        ax.set_xlabel("Epoch", color=TEXT_COLOR)
        ax.legend(labelcolor=TEXT_COLOR, fontsize=8,
                  facecolor=GRID_COLOR, edgecolor="#334155", framealpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Recall + Precision + Specificity
        ax = axes[2]
        ax.plot(df["epoch"], df["val_recall"],      color="#F59E0B",
                linewidth=1.8, label="recall")
        ax.plot(df["epoch"], df["val_precision"],   color="#EF4444",
                linewidth=1.6, linestyle=":",   label="precision")
        ax.plot(df["epoch"], df["val_specificity"], color="#8B5CF6",
                linewidth=1.6, linestyle="-.", label="specificity")
        ax.set_ylim(0, 1.05)
        ax.set_title("Recall / Precision / Specificity", color=TEXT_COLOR)
        ax.set_xlabel("Epoch", color=TEXT_COLOR)
        ax.legend(labelcolor=TEXT_COLOR, fontsize=8,
                  facecolor=GRID_COLOR, edgecolor="#334155", framealpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.suptitle(f"Teacher Training Curves — {variant}",
                     color=TEXT_COLOR, fontsize=13, y=1.01)
        fig.tight_layout()
        save_chart(fig, f"teacher_training_curves_{variant.replace('-', '_')}.png")


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def _student_comparison_bar(csv_path: Path, stage: int):
    """
    Multi-metric grouped bar chart for one Student comparison CSV.
    Stage 1 groups by encoder variant; Stage 2 groups by factory mode.
    """
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path.name} not found.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    # Determine group column
    group_col = "encoder" if stage == 1 else "mode"
    if group_col not in df.columns:
        group_col = df.columns[0]

    groups  = df[group_col].tolist()
    n       = len(groups)

    # Metrics to plot — use test-level columns when available, fall back to val
    metric_map = [
        ("sil_mIoU",    "sil_mIoU",    "Silhouette\nmIoU",   "#3B82F6"),
        ("sym_mIoU",    "sym_mIoU",    "Symptom\nmIoU",      "#10B981"),
        ("msv_f1",      "msv_f1",      "MSV F1",             "#F59E0B"),
        ("mln_f1",      "mln_f1",      "MLN F1",             "#EC4899"),
        ("msv_roc_auc", "msv_roc_auc", "MSV ROC-AUC",        "#14B8A6"),
        ("macro_f1",    "macro_f1",    "Macro F1",           "#EF4444"),
        ("composite",   "composite",   "Composite",          "#8B5CF6"),
    ]

    # Find which columns actually exist
    available = [(key, col, lbl, clr) for key, col, lbl, clr in metric_map
                 if col in df.columns]
    if not available:
        print(f"  [SKIP] Stage {stage} comparison — no expected columns found.")
        return

    n_m    = len(available)
    x      = np.arange(n)
    width  = 0.14
    offsets= np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width

    fig, ax = plt.subplots(figsize=(max(9, n * 2.5), 5.5))
    apply_dark_style(fig, ax)

    for mi, (_, col, label, clr) in enumerate(available):
        vals = [safe_float(df[col].iloc[i]) for i in range(n)]
        bars = ax.bar(x + offsets[mi], vals, width=width,
                      label=label, color=clr, alpha=0.88,
                      zorder=3, edgecolor=BG_COLOR, linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.006,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=6.5, color=TEXT_COLOR, rotation=50)

    ax.set_xticks(x)
    ax.set_xticklabels([str(g).replace("_", "\n") for g in groups],
                       color=TEXT_COLOR, fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", color=TEXT_COLOR)

    title_stage = "Encoder Ablation (Stage 1, Mode B)" if stage == 1 \
                  else "Mode Ablation (Stage 2, Best Encoder)"
    ax.set_title(f"Student — {title_stage}", color=TEXT_COLOR, fontsize=12, pad=12)

    # Highlight best by composite if column present
    if "composite" in df.columns:
        comp_vals = [safe_float(df["composite"].iloc[i]) for i in range(n)]
        best_idx  = int(np.argmax(comp_vals))
        ax.axvspan(best_idx - 0.42, best_idx + 0.42, alpha=0.07,
                   color=ACCENT, zorder=1)
        ax.text(best_idx, 1.10, "★ best composite",
                ha="center", fontsize=8, color=ACCENT)

    ax.legend(loc="upper right", framealpha=0.15, labelcolor=TEXT_COLOR,
              fontsize=8.5, facecolor=GRID_COLOR, edgecolor="#334155")
    fig.tight_layout()
    save_chart(fig, f"student_stage{stage}_comparison_bar.png")


def chart_student_comparison_stage1():
    _student_comparison_bar(LOGS_DIR / "student_comparison_stage1.csv", stage=1)


def chart_student_comparison_stage2():
    _student_comparison_bar(LOGS_DIR / "student_comparison_stage2.csv", stage=2)


def chart_student_training_curves():
    """
    Per-epoch training curves (both Phase 1 and Phase 2) for every
    student_{enc}_{mode}_metrics.csv found in LOGS_DIR.
    """
    for csv_path in sorted(LOGS_DIR.glob("student_*_metrics.csv")):
        # Skip test-metric CSVs (those start student_test_)
        if "test" in csv_path.stem:
            continue
        # Extract enc_mode label from filename: student_{enc}_{mode}_metrics.csv
        stem  = csv_path.stem                        # e.g. student_mobilenet_v2_cbam_mode_b_metrics
        label = stem[len("student_"):-len("_metrics")]  # mobilenet_v2_cbam_mode_b

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        # Split by phase if column exists
        phases = sorted(df["phase"].unique()) if "phase" in df.columns else [None]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        apply_dark_style(fig, axes)

        phase_colors = {1: "#3B82F6", 2: "#10B981", None: "#3B82F6"}
        phase_labels = {1: "Phase 1 (frozen enc)", 2: "Phase 2 (fine-tune)", None: ""}

        metrics_plot = [
            ("val_loss",   "train_loss",   "Loss",               None,   None),
            ("sil_mIoU",   "sym_mIoU",     "Seg mIoU",           0, 1.05),
            ("sil_dice",   "sym_dice",     "Seg Dice",           0, 1.05),
            ("msv_f1",     "mln_f1",       "MSV / MLN F1",       0, 1.05),
            ("sev_mae",    None,           "Severity MAE (%)",   None, None),
            ("composite",  None,           "Composite Score",    0, 1.05),
        ]

        for ax, (col1, col2, title, ymin, ymax) in zip(axes, metrics_plot):
            for ph in phases:
                sub = df[df["phase"] == ph] if ph is not None else df
                if col1 in sub.columns:
                    ax.plot(sub["epoch"], sub[col1],
                            color=phase_colors[ph],
                            linewidth=2.0,
                            label=f"{col1} {phase_labels.get(ph,'')}")
                if col2 and col2 in sub.columns:
                    ax.plot(sub["epoch"], sub[col2],
                            color=phase_colors[ph],
                            linewidth=1.5, linestyle="--",
                            alpha=0.75,
                            label=f"{col2} {phase_labels.get(ph,'')}")

            # Phase boundary vertical line
            if "phase" in df.columns and 2 in df["phase"].values:
                p2_start = df[df["phase"] == 2]["epoch"].min()
                ax.axvline(p2_start, color="#94A3B8",
                           linewidth=1.0, linestyle=":", alpha=0.7)
                ax.text(p2_start + 0.3,
                        ax.get_ylim()[1] * 0.92,
                        "P2", color="#94A3B8", fontsize=7.5)

            ax.set_title(title, color=TEXT_COLOR, fontsize=9)
            ax.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=8)
            if ymin is not None:
                ax.set_ylim(ymin, ymax)
            ax.legend(labelcolor=TEXT_COLOR, fontsize=6.5,
                      facecolor=GRID_COLOR, edgecolor="#334155", framealpha=0.2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.suptitle(f"Student Training Curves — {label.replace('_', ' ')}",
                     color=TEXT_COLOR, fontsize=12, y=1.01)
        fig.tight_layout()
        save_chart(fig, f"student_training_curves_{label}.png")


def chart_student_confusion_matrices():
    """
    3×3 confusion matrix heatmap for every student_confusion_{enc}_{mode}.csv.
    """
    class_names = ["HEALTHY", "MSV", "MLN"]

    for csv_path in sorted(LOGS_DIR.glob("student_confusion_*.csv")):
        stem  = csv_path.stem                               # student_confusion_...
        label = stem[len("student_confusion_"):]            # enc_mode label

        df = pd.read_csv(csv_path, index_col=0)
        if df.shape != (3, 3):
            # Try reading without index
            df = pd.read_csv(csv_path)
            if "true/pred" in df.columns:
                df = df.set_index("true/pred")
            if df.shape != (3, 3):
                print(f"  [SKIP] {csv_path.name} — unexpected shape {df.shape}")
                continue

        mat    = df.values.astype(float)
        totals = mat.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1
        mat_norm = mat / totals          # row-normalise → recall per class

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        apply_dark_style(fig, [ax1, ax2])

        for ax, data, title_suffix in [
            (ax1, mat.astype(int), "Counts"),
            (ax2, mat_norm,        "Row-Normalised (Recall)"),
        ]:
            im = ax.imshow(data if title_suffix == "Counts" else mat_norm,
                           cmap="YlGn", aspect="auto",
                           vmin=0,
                           vmax=mat.max() if title_suffix == "Counts" else 1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            for i in range(3):
                for j in range(3):
                    val_str = (f"{int(mat[i,j]):,}" if title_suffix == "Counts"
                               else f"{mat_norm[i,j]:.2f}")
                    ax.text(j, i, val_str,
                            ha="center", va="center", fontsize=10,
                            color="black" if (
                                mat_norm[i, j] > 0.45
                            ) else TEXT_COLOR,
                            fontweight="bold")

            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels([f"Pred:\n{c}" for c in class_names],
                               color=TEXT_COLOR, fontsize=8)
            ax.set_yticklabels([f"True:\n{c}" for c in class_names],
                               color=TEXT_COLOR, fontsize=8)
            ax.set_title(title_suffix, color=TEXT_COLOR)

        fig.suptitle(f"Student Confusion Matrix — {label.replace('_', ' ')}",
                     color=TEXT_COLOR, fontsize=12, y=1.02)
        fig.tight_layout()
        save_chart(fig, f"student_confusion_{label}.png")


def chart_student_radar():
    """
    Radar / spider chart comparing all Student test results across key metrics.
    One radar per student_test_metrics_*.csv found.
    Also produces an overlay radar comparing all variants on the same axes.
    """
    test_csvs = sorted(LOGS_DIR.glob("student_test_metrics_*.csv"))
    if not test_csvs:
        print("  [SKIP] No student_test_metrics_*.csv found.")
        return

    radar_metrics = [
        ("sil_mIoU",     "Sil mIoU"),
        ("sym_mIoU",     "Sym mIoU"),
        ("msv_f1",       "MSV F1"),
        ("mln_f1",       "MLN F1"),
        ("msv_roc_auc",  "MSV AUC"),
        ("macro_roc_auc","Macro AUC"),
        ("cls_accuracy", "Cls Acc"),
        ("composite",    "Composite"),
        ("sev_r2",       "Sev R²"),
    ]

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]    # close the polygon

    # ── Per-variant individual radar ──────────────────────────────────────────
    for csv_path in test_csvs:
        stem  = csv_path.stem
        label = stem[len("student_test_metrics_"):]
        df    = pd.read_csv(csv_path)
        if df.empty:
            continue
        row   = df.iloc[0]

        vals = [safe_float(row.get(col, 0)) for col, _ in radar_metrics]
        vals += vals[:1]

        fig, ax = plt.subplots(figsize=(6, 6),
                               subplot_kw=dict(polar=True))
        apply_dark_style(fig)
        ax.set_facecolor(GRID_COLOR)

        ax.plot(angles, vals, color=ACCENT, linewidth=2.2)
        ax.fill(angles, vals, color=ACCENT, alpha=0.20)

        ax.set_thetagrids(np.degrees(angles[:-1]),
                          [lbl for _, lbl in radar_metrics],
                          color=TEXT_COLOR, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"],
                           color="#94A3B8", fontsize=7)
        ax.tick_params(colors=TEXT_COLOR)
        ax.spines["polar"].set_color("#334155")
        ax.grid(color="#334155", linewidth=0.6)

        ax.set_title(f"Student Radar\n{label.replace('_', ' ')}",
                     color=TEXT_COLOR, fontsize=11, pad=20)
        fig.patch.set_facecolor(BG_COLOR)
        fig.tight_layout()
        save_chart(fig, f"student_radar_{label}.png")

    # ── All-variants overlay radar ────────────────────────────────────────────
    if len(test_csvs) < 2:
        return

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    apply_dark_style(fig)
    ax.set_facecolor(GRID_COLOR)

    for i, csv_path in enumerate(test_csvs):
        stem  = csv_path.stem
        label = stem[len("student_test_metrics_"):]
        df    = pd.read_csv(csv_path)
        if df.empty:
            continue
        row   = df.iloc[0]
        vals  = [safe_float(row.get(col, 0)) for col, _ in radar_metrics]
        vals += vals[:1]
        color = FALLBACK_COLORS[i % len(FALLBACK_COLORS)]

        ax.plot(angles, vals, color=color, linewidth=1.8,
                label=label.replace("_", " "))
        ax.fill(angles, vals, color=color, alpha=0.07)

    ax.set_thetagrids(np.degrees(angles[:-1]),
                      [lbl for _, lbl in radar_metrics],
                      color=TEXT_COLOR, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"],
                       color="#94A3B8", fontsize=7)
    ax.tick_params(colors=TEXT_COLOR)
    ax.spines["polar"].set_color("#334155")
    ax.grid(color="#334155", linewidth=0.6)

    ax.legend(loc="lower right", framealpha=0.2, labelcolor=TEXT_COLOR,
              fontsize=8, facecolor=GRID_COLOR, edgecolor="#334155",
              bbox_to_anchor=(1.30, -0.05))

    ax.set_title("Student — All Variants Radar Overlay",
                 color=TEXT_COLOR, fontsize=11, pad=20)
    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout()
    save_chart(fig, "student_radar_all_overlay.png")


def chart_student_metrics_heatmap():
    """
    Heatmap of all key test metrics across all Student variants × modes.
    Rows = variant_mode combinations, columns = metrics.
    Gives the thesis a single-glance comparison table as a visual.
    """
    test_csvs = sorted(LOGS_DIR.glob("student_test_metrics_*.csv"))
    if len(test_csvs) < 2:
        return

    metrics_cols = [
        ("sil_mIoU",        "Sil\nmIoU"),
        ("sym_mIoU",        "Sym\nmIoU"),
        ("msv_f1",          "MSV\nF1"),
        ("msv_roc_auc",     "MSV\nAUC"),
        ("macro_roc_auc",   "Macro\nAUC"),
        ("mln_f1",          "MLN\nF1"),
        ("healthy_f1",      "HLT\nF1"),
        ("cls_accuracy",    "Cls\nAcc"),
        ("mcc",             "MCC"),
        ("cohen_kappa",     "Kappa"),
        ("composite",       "Composite"),
        ("sev_mae_pct",     "Sev\nMAE%"),
        ("sev_r2",          "Sev\nR²"),
        ("cpu_lat_mean_ms", "CPU\nms"),
    ]

    rows, row_labels = [], []
    for csv_path in test_csvs:
        stem  = csv_path.stem
        label = stem[len("student_test_metrics_"):]
        df    = pd.read_csv(csv_path)
        if df.empty:
            continue
        row   = df.iloc[0]
        row_labels.append(label.replace("_", " "))
        rows.append([safe_float(row.get(col, 0)) for col, _ in metrics_cols])

    if not rows:
        return

    mat = np.array(rows)

    # Normalise each column to [0,1] for the heatmap colour
    mat_norm = mat.copy()
    for j in range(mat.shape[1]):
        col_min, col_max = mat[:, j].min(), mat[:, j].max()
        if col_max > col_min:
            mat_norm[:, j] = (mat[:, j] - col_min) / (col_max - col_min)
        else:
            mat_norm[:, j] = 0.5

    # Invert MAE and latency (lower is better)
    invert_cols = {"Sev\nMAE%", "CPU\nms"}
    for j, (_, lbl) in enumerate(metrics_cols):
        if lbl in invert_cols:
            mat_norm[:, j] = 1.0 - mat_norm[:, j]

    fig_h = max(3.5, len(rows) * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(len(metrics_cols) * 1.1 + 1, fig_h))
    apply_dark_style(fig, ax)

    im = ax.imshow(mat_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02,
                 label="Normalised score (higher = better)")

    ax.set_xticks(range(len(metrics_cols)))
    ax.set_xticklabels([lbl for _, lbl in metrics_cols],
                       color=TEXT_COLOR, fontsize=8.5)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, color=TEXT_COLOR, fontsize=8.5)

    # Annotate with raw values
    for i in range(len(rows)):
        for j in range(len(metrics_cols)):
            raw = mat[i, j]
            fmt = f"{raw:.2f}" if raw < 100 else f"{raw:.0f}"
            ax.text(j, i, fmt,
                    ha="center", va="center", fontsize=7.5,
                    color="black" if mat_norm[i, j] > 0.55 else TEXT_COLOR)

    ax.set_title("Student — All Variants Test Metrics Heatmap",
                 color=TEXT_COLOR, fontsize=12, pad=10)
    fig.tight_layout()
    save_chart(fig, "student_metrics_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_bouncer():
    print("\n── Bouncer charts ──────────────────────────────────────────")
    chart_bouncer_comparison()
    chart_bouncer_training_curves()
    chart_bouncer_confusion()


def run_teacher():
    print("\n── Teacher charts ──────────────────────────────────────────")
    chart_teacher_comparison()
    chart_teacher_training_curves()


def run_student():
    print("\n── Student charts ──────────────────────────────────────────")
    chart_student_comparison_stage1()
    chart_student_comparison_stage2()
    chart_student_training_curves()
    chart_student_confusion_matrices()
    chart_student_radar()
    chart_student_metrics_heatmap()


def main():
    parser = argparse.ArgumentParser(
        description="Yellow MAIze — generate comparison charts from training CSV logs.")
    parser.add_argument("--bouncer",  action="store_true", help="Bouncer charts only")
    parser.add_argument("--teacher",  action="store_true", help="Teacher charts only")
    parser.add_argument("--student",  action="store_true", help="Student charts only")
    args = parser.parse_args()

    all_flags = not any([args.bouncer, args.teacher, args.student])

    print("=" * 64)
    print("  Yellow MAIze — Chart Generator")
    print(f"  Output directory: {CHART_DIR.relative_to(ROOT)}")
    print("=" * 64)

    if all_flags or args.bouncer:
        run_bouncer()
    if all_flags or args.teacher:
        run_teacher()
    if all_flags or args.student:
        run_student()

    print("\n  Done.")
    print("=" * 64)


if __name__ == "__main__":
    main()
