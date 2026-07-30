"""
================================================================================
 generate_report.py — HTML Evaluation Report Generator
================================================================================
 PURPOSE:
   Reads all CSV logs, metrics, and image outputs produced by the pipeline
   and generates a single self-contained HTML report with:

     Section 1  — Pipeline overview and dataset statistics
     Section 2  — Preprocessing summary (rejection reasons, class distribution)
     Section 3  — Bouncer comparison (bar charts, ROC-AUC table, confusion matrix)
     Section 4  — Teacher comparison (Dice/IoU/Recall bar charts, training curves)
     Section 5  — Student encoder ablation (multi-metric comparison table + chart)
     Section 6  — Student mode ablation (pseudo-label strategy comparison)
     Section 7  — Best Student test results (all metrics, confusion matrix heatmap,
                   per-class breakdown, severity regression scatter)
     Section 8  — XAI comparison (pointing game / insertion / deletion charts,
                   embedded overlay images)
     Section 9  — Severity reliability (inter-rater analysis)
     Section 10 — Deployment summary (model sizes, latency, TFLite readiness)
     Section 11 — Training curves (loss + metrics over epochs for best Student)

   The report is 100% self-contained — all charts are inline base64 SVG/PNG,
   all images are embedded. Single file, no external dependencies to view.

 RUN AFTER: python select_best_pipeline.py

 OUTPUT:
   reports/evaluation_report.html   ← send to supervisor, include in appendix
================================================================================
"""

import base64
import csv
import io
import json
import re
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from config import (
    LOGS_DIR, REPORTS_DIR, EXPORTS_DIR,
    CLASSES, FACTORY_MODES,
)

REPORT_PATH       = REPORTS_DIR / "evaluation_report.html"
MOBILE_RANKED_CSV = REPORTS_DIR / "student_mobile_ranking.csv"

# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS (matplotlib → base64 PNG embedded in HTML)
# ══════════════════════════════════════════════════════════════════════════════

def _fig_to_b64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _img_to_b64(path: Path) -> str | None:
    """Convert image file to base64 string."""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def _get_mpl():
    """Import matplotlib with non-interactive backend. Raises ImportError with clear message if missing."""
    try:
        import matplotlib
    except ImportError:
        raise ImportError(
            "matplotlib is required for chart generation.\n"
            "Install: pip install matplotlib>=3.7.0"
        )
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    # Dark theme matching the HTML
    plt.rcParams.update({
        "figure.facecolor":  "#1a1a2e",
        "axes.facecolor":    "#16213e",
        "axes.edgecolor":    "#4a4a7a",
        "axes.labelcolor":   "#e0e0e0",
        "text.color":        "#e0e0e0",
        "xtick.color":       "#b0b0c0",
        "ytick.color":       "#b0b0c0",
        "grid.color":        "#2a2a4a",
        "grid.linestyle":    "--",
        "grid.alpha":        0.4,
        "legend.facecolor":  "#1a1a2e",
        "legend.edgecolor":  "#4a4a7a",
        "font.size":         10,
    })
    return plt


ACCENT   = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373", "#ce93d8", "#80cbc4"]
ACCENT_M = "#4fc3f7"


# ══════════════════════════════════════════════════════════════════════════════
# CHART GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def chart_bar_comparison(df: pd.DataFrame, variants_col: str,
                          metrics: list[str], title: str,
                          ylabel: str = "Score") -> str:
    plt = _get_mpl()
    import matplotlib.pyplot as _plt

    variants = df[variants_col].tolist()
    x = np.arange(len(variants))
    width = 0.8 / max(len(metrics), 1)

    fig, ax = _plt.subplots(figsize=(max(6, len(variants)*1.5), 4))
    for i, metric in enumerate(metrics):
        vals = pd.to_numeric(df[metric], errors="coerce").fillna(0).tolist()
        bars = ax.bar(x + i*width - (len(metrics)-1)*width/2,
                      vals, width*0.9, label=metric, color=ACCENT[i % len(ACCENT)])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                    color="#e0e0e0")

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, pad=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.08)
    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    _plt.close(fig)
    return b64


def chart_training_curve(df: pd.DataFrame, metrics: list[str],
                          title: str) -> str:
    plt = _get_mpl()
    import matplotlib.pyplot as _plt

    fig, ax = _plt.subplots(figsize=(9, 4))
    if "epoch" not in df.columns:
        _plt.close(fig)
        return ""

    x = df["epoch"].tolist()
    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
        vals = pd.to_numeric(df[metric], errors="coerce").tolist()
        ax.plot(x, vals, color=ACCENT[i % len(ACCENT)],
                label=metric, linewidth=1.8, marker="o",
                markersize=3, alpha=0.9)

    # Mark phase boundary if both phases present
    if "phase" in df.columns:
        p2_start = df[df["phase"] == 2]["epoch"].min()
        if pd.notna(p2_start):
            ax.axvline(p2_start, color="#ffb74d", linestyle="--",
                       alpha=0.6, linewidth=1.2, label="Phase 2 start")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title(title, fontsize=11, pad=10)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    _plt.close(fig)
    return b64


def chart_confusion_matrix(cm: np.ndarray,
                            labels: list[str], title: str) -> str:
    plt = _get_mpl()
    import matplotlib.pyplot as _plt
    import matplotlib.colors as mcolors

    fig, ax = _plt.subplots(figsize=(5, 4))
    cmap = _plt.cm.Blues
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=cm.max())
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if cm[i, j] < thresh else "#1a1a2e")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([f"Pred\n{l}" for l in labels], fontsize=9)
    ax.set_yticklabels([f"True\n{l}" for l in labels], fontsize=9)
    ax.set_title(title, fontsize=11, pad=10)
    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    _plt.close(fig)
    return b64


def chart_severity_distribution(df: pd.DataFrame, title: str) -> str:
    """Severity histogram per class per mode."""
    plt = _get_mpl()
    import matplotlib.pyplot as _plt

    modes   = [m for m in FACTORY_MODES if m in df.get("mode", pd.Series()).unique()] \
              if "mode" in df.columns else []
    if not modes:
        return ""

    fig, axes = _plt.subplots(1, len(modes), figsize=(4*len(modes), 3.5),
                               sharey=True)
    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        mdf = df[df["mode"] == mode]
        for i, cls in enumerate(CLASSES):
            cdf = mdf[mdf["class"] == cls]
            if cdf.empty:
                continue
            sev = pd.to_numeric(cdf["mean_severity"], errors="coerce").dropna()
            if sev.empty:
                continue
            ax.bar(i, sev.mean(), color=ACCENT[i], width=0.6,
                   label=cls, alpha=0.85)
            ax.text(i, sev.mean() + 0.3, f"{sev.mean():.1f}%",
                    ha="center", fontsize=8, color="#e0e0e0")
        ax.set_title(mode, fontsize=9)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, fontsize=8)
        ax.set_ylabel("Mean Severity %" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    _plt.close(fig)
    return b64


def chart_xai_comparison(df: pd.DataFrame) -> str:
    plt = _get_mpl()
    import matplotlib.pyplot as _plt

    methods = df["method"].unique().tolist() if "method" in df.columns else []
    if not methods:
        return ""

    metrics_xai = ["pointing_game_acc", "insertion_auc", "deletion_auc"]
    method_means = {}
    for m in methods:
        mdf = df[df["method"] == m]
        method_means[m] = {
            k: pd.to_numeric(mdf[k], errors="coerce").mean()
            for k in metrics_xai if k in mdf.columns
        }

    fig, ax = _plt.subplots(figsize=(7, 4))
    x = np.arange(len(metrics_xai))
    width = 0.25
    for i, (method, vals) in enumerate(method_means.items()):
        yvals = [vals.get(m, 0) for m in metrics_xai]
        bars = ax.bar(x + i*width - width, yvals, width*0.85,
                      label=method, color=ACCENT[i], alpha=0.85)
        for bar, val in zip(bars, yvals):
            if not np.isnan(val):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.005,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=7.5, color="#e0e0e0")

    ax.set_xticks(x)
    ax.set_xticklabels(["Pointing Game", "Insertion AUC", "Deletion AUC"])
    ax.set_title("XAI Method Comparison", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    _plt.close(fig)
    return b64


# ══════════════════════════════════════════════════════════════════════════════
# HTML COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
:root {
  --bg: #0f0f1a; --bg2: #1a1a2e; --bg3: #16213e;
  --accent: #4fc3f7; --accent2: #81c784; --accent3: #ffb74d;
  --warn: #e57373; --text: #e0e0e0; --text2: #b0b0c0;
  --border: #2a2a4a; --radius: 8px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg);
       color: var(--text); line-height: 1.6; }
.container { max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem; }
h1 { font-size: 1.8rem; color: var(--accent); margin-bottom: 0.25rem; }
h2 { font-size: 1.25rem; color: var(--accent); margin: 2rem 0 1rem;
     padding-bottom: 0.4rem; border-bottom: 1px solid var(--border); }
h3 { font-size: 1rem; color: var(--accent2); margin: 1.25rem 0 0.5rem; }
.subtitle { color: var(--text2); font-size: 0.9rem; margin-bottom: 2rem; }
.section { background: var(--bg2); border-radius: var(--radius);
           padding: 1.5rem; margin-bottom: 1.5rem;
           border: 1px solid var(--border); }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }
.card { background: var(--bg3); border-radius: var(--radius);
        padding: 1rem; border: 1px solid var(--border); }
.card.good { border-color: var(--accent2); }
.card.warn { border-color: var(--accent3); }
.card.bad  { border-color: var(--warn); }
.metric-val { font-size: 1.6rem; font-weight: 600; color: var(--accent); }
.metric-lbl { font-size: 0.8rem; color: var(--text2); margin-top: 0.15rem; }
table { width: 100%; border-collapse: collapse; font-size: 0.875rem; margin-top: 0.75rem; }
th { background: var(--bg); color: var(--accent); text-align: left;
     padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border);
     font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }
td { padding: 0.45rem 0.75rem; border-bottom: 1px solid var(--border);
     color: var(--text); }
tr:last-child td { border-bottom: none; }
tr.best td { background: rgba(79,195,247,0.08); }
tr.best td:first-child { border-left: 3px solid var(--accent); }
.badge { font-size: 0.75rem; padding: 2px 8px; border-radius: 12px;
         display: inline-block; font-weight: 500; }
.badge-ok   { background: rgba(129,199,132,0.2); color: var(--accent2); }
.badge-warn { background: rgba(255,183,77,0.2);  color: var(--accent3); }
.badge-bad  { background: rgba(229,115,115,0.2); color: var(--warn); }
img.chart { width: 100%; border-radius: var(--radius); margin-top: 0.75rem; }
.overlay-grid { display: grid; grid-template-columns: repeat(auto-fill,minmax(280px,1fr));
                gap: 0.75rem; margin-top: 0.75rem; }
.overlay-card { background: var(--bg3); border-radius: var(--radius);
                padding: 0.5rem; border: 1px solid var(--border); }
.overlay-card img { width: 100%; border-radius: 4px; }
.overlay-lbl { font-size: 0.75rem; color: var(--text2); margin-top: 0.35rem;
               text-align: center; }
.toc { background: var(--bg3); border-radius: var(--radius); padding: 1rem 1.5rem;
       margin-bottom: 1.5rem; border: 1px solid var(--border); }
.toc a { color: var(--accent); text-decoration: none; font-size: 0.875rem; }
.toc a:hover { text-decoration: underline; }
.toc li { margin: 0.25rem 0; }
.note { background: rgba(79,195,247,0.07); border-left: 3px solid var(--accent);
        padding: 0.75rem 1rem; border-radius: 0 var(--radius) var(--radius) 0;
        font-size: 0.875rem; color: var(--text2); margin: 0.75rem 0; }
.warn-box { background: rgba(255,183,77,0.07); border-left: 3px solid var(--accent3);
            padding: 0.75rem 1rem; border-radius: 0 var(--radius) var(--radius) 0;
            font-size: 0.875rem; margin: 0.75rem 0; }
@media(max-width:700px){ .grid2,.grid3{ grid-template-columns:1fr; } }
"""


def tag(t: str, content: str, **attrs) -> str:
    attr_str = " ".join(f'{k.replace("_","-")}="{v}"' for k, v in attrs.items())
    return f"<{t} {attr_str}>{content}</{t}>"


def section(title: str, anchor: str, content: str) -> str:
    return f'''<div class="section" id="{anchor}">
<h2>{title}</h2>
{content}
</div>'''


def metric_card(value, label: str, quality: str = "") -> str:
    cls = f"card {quality}" if quality else "card"
    return f'<div class="{cls}"><div class="metric-val">{value}</div><div class="metric-lbl">{label}</div></div>'


def chart_img(b64: str, alt: str = "chart") -> str:
    if not b64:
        return "<p style='color:#666;font-size:0.8rem;padding:0.5rem'>Chart not available — run evaluate scripts first.</p>"
    return f'<img class="chart" src="data:image/png;base64,{b64}" alt="{alt}">'


def safe_chart(fn, *args, **kwargs) -> str:
    """Wrap chart generation — one bad chart won't crash the whole report."""
    try:
        return chart_img(fn(*args, **kwargs))
    except ImportError as e:
        return f"<div class='warn-box'>Chart unavailable: {e}<br>pip install matplotlib</div>"
    except Exception as e:
        return f"<div style='color:#888;font-size:0.8rem;padding:0.5rem'>Chart error: {str(e)[:140]}</div>"


def df_to_table(df: pd.DataFrame, highlight_col: str = "",
                highlight_max: bool = True, fmt: dict = None) -> str:
    if df.empty:
        return "<p style='color:#666'>No data</p>"
    fmt = fmt or {}
    html = "<table><thead><tr>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"

    if highlight_col and highlight_col in df.columns:
        best_val = df[highlight_col].max() if highlight_max else df[highlight_col].min()
    else:
        best_val = None

    for _, row in df.iterrows():
        is_best = (best_val is not None and
                   pd.to_numeric(row.get(highlight_col, None),
                                 errors="coerce") == best_val)
        html += f'<tr{"  class=\"best\"" if is_best else ""}>'
        for col in df.columns:
            val = row[col]
            if col in fmt and isinstance(val, (int, float)):
                val = fmt[col].format(val)
            html += f"<td>{val}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html


# ══════════════════════════════════════════════════════════════════════════════
# SECTION BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_overview() -> str:
    # Try to load preprocessing summary
    preproc = REPORTS_DIR / "preprocessing_summary.txt"
    summary_text = preproc.read_text() if preproc.exists() else ""

    # Parse key numbers from summary
    def extract(pattern, text, default="N/A"):
        m = re.search(pattern, text)
        return m.group(1).replace(",","") if m else default

    total    = extract(r"Total files scanned\s+:\s+([\d,]+)", summary_text)
    rejected = extract(r"Rejected \(all reasons\)\s+:\s+([\d,]+)", summary_text)
    flagged  = extract(r"Flagged \(review needed\)\s+:\s+([\d,]+)", summary_text)
    final    = extract(r"Final valid images\s+:\s+([\d,]+)", summary_text)

    cards = f"""<div class="grid3">
{metric_card(total,    "Total files scanned")}
{metric_card(rejected, "Rejected (preprocessing)", "warn")}
{metric_card(final,    "Final valid images", "good")}
</div>"""

    # Rejection breakdown table
    rej_csv = REPORTS_DIR / "preprocessing_report.csv"
    rej_table = ""
    if rej_csv.exists():
        df = pd.read_csv(rej_csv)
        if not df.empty and "reason" in df.columns:
            counts = df["reason"].str.split(":").str[0].value_counts().reset_index()
            counts.columns = ["Rejection Reason", "Count"]
            counts["% of Scanned"] = (counts["Count"] / max(int(total.replace(",","") if total != "N/A" else 1), 1) * 100).round(2)
            rej_table = f"<h3>Rejection breakdown</h3>{df_to_table(counts)}"

    note = '<div class="note">Preprocessing pipeline: zero-byte check → magic bytes → truncation → resolution → colour mode → uniformity → MD5 exact dedup → pHash near-dedup → cross-class conflict detection → low-green flag.</div>'
    return cards + note + rej_table


def chart_admission_breakdown(row: pd.Series) -> str:
    """
    Horizontal stacked bar: what happened to every test-split maize image —
    admitted, rejected by the heuristic pre-filter, or rejected by the neural
    Bouncer. A visual complement to the admission-rate metric cards, so the
    breakdown isn't only ever seen as three separate numbers.
    """
    plt = _get_mpl()
    import matplotlib.pyplot as _plt

    n_total = float(row.get("n_test_maize", 0)) or 1.0
    n_heuristic_rej = float(row.get("n_heuristic_reject", 0))
    n_passed = float(row.get("n_passed", 0))
    n_neural_rej = max(n_total - n_heuristic_rej - n_passed, 0.0)

    segments = [
        ("Admitted", n_passed, "#2ecc71"),
        ("Rejected — neural Bouncer", n_neural_rej, "#e67e22"),
        ("Rejected — heuristic pre-filter", n_heuristic_rej, "#e74c3c"),
    ]

    fig, ax = _plt.subplots(figsize=(8, 2.2))
    left = 0.0
    for label, val, color in segments:
        pct = 100 * val / n_total
        if val > 0:
            ax.barh(0, val, left=left, color=color, height=0.55, label=label)
            if pct >= 4:
                ax.text(left + val / 2, 0, f"{pct:.1f}%", ha="center", va="center",
                        fontsize=9, color="#1a1a2e", fontweight="bold")
        left += val

    ax.set_xlim(0, n_total)
    ax.set_yticks([])
    ax.set_xlabel(f"Test-split maize images (n={int(n_total)})")
    ax.set_title("Bouncer Admission Breakdown — Test-Split Maize Images",
                 fontsize=11, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, fontsize=8,
             frameon=False)
    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    _plt.close(fig)
    return b64


def chart_severity_scatter_inline(df: pd.DataFrame) -> str:
    """
    Same content as generate_charts.py's chart_severity_scatter(), reimplemented
    in this file's own lightweight inline-chart style (plain matplotlib, no
    dark theme helpers) since this report embeds charts directly as base64
    rather than referencing generate_charts.py's saved PNG files — the two
    chart systems are intentionally separate (see the note in
    chart_admission_breakdown() above).
    """
    plt = _get_mpl()
    import matplotlib.pyplot as _plt

    fig, ax = _plt.subplots(figsize=(6, 6))
    classes_present = sorted(df["category"].unique().tolist()) if "category" in df.columns else ["all"]
    colors = {"HEALTHY": "#2ecc71", "MSV": "#3498db", "MLN": "#e74c3c"}
    for cls in classes_present:
        cdf = df[df["category"] == cls] if "category" in df.columns else df
        ax.scatter(cdf["human_norm"], cdf["hsv_norm"], s=30, alpha=0.7,
                  color=colors.get(cls, "#95a5a6"), label=cls,
                  edgecolors="white", linewidths=0.3)
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1,
           label="Perfect agreement")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Human-rated severity (normalized)")
    ax.set_ylabel("HSV-derived severity (normalized)")
    ax.set_title("Severity Reliability — Human vs. HSV-Derived")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    _plt.close(fig)
    return b64


def build_bouncer() -> str:
    comp_csv = LOGS_DIR / "bouncer_comparison.csv"
    if not comp_csv.exists():
        return "<p>Bouncer comparison not found. Run train_bouncer.py first.</p>"

    df = pd.read_csv(comp_csv)

    # Table
    display_cols = [c for c in ["variant","best_f1","threshold","specificity",
                                  "maize_recall","roc_auc","lat_cpu_ms","TP","FP","TN","FN"]
                    if c in df.columns]
    table = df_to_table(df[display_cols], highlight_col="specificity")

    # Bar chart
    num_cols = [c for c in ["specificity","maize_recall","best_f1","roc_auc"]
                if c in df.columns]
    df_neural = df[df["variant"].isin(["mobilenet_v3_large","edgevit_xxs"])].copy()
    chart = ""
    if not df_neural.empty and num_cols:
        for col in num_cols:
            df_neural[col] = pd.to_numeric(df_neural[col], errors="coerce")
        chart = safe_chart(
            chart_bar_comparison, df_neural, "variant", num_cols,
            "Bouncer Neural Variants — Key Metrics")

    # Admission rate
    adm_html = ""
    for f in LOGS_DIR.glob("bouncer_admission_rate_*.csv"):
        adf = pd.read_csv(f)
        if not adf.empty:
            row = adf.iloc[0]
            rate = float(row.get("admission_rate", 0))
            quality = "good" if rate >= 0.95 else "warn" if rate >= 0.90 else "bad"
            adm_html += f"""<h3>End-to-end admission rate ({row.get('variant','')})</h3>
<div class="grid3">
{metric_card(f"{rate*100:.1f}%", "Admission rate (test-split maize)", quality)}
{metric_card(f"{float(row.get('false_rejection_rate',0))*100:.1f}%", "False rejection rate", "warn" if float(row.get('false_rejection_rate',0)) > 0.05 else "good")}
{metric_card(str(row.get('n_test_maize','N/A')), "Test-split maize images evaluated")}
</div>
<div class="note">Admission rate directly adjusts the Student model\'s effective end-to-end recall. A false rejection rate below 5% is acceptable.</div>"""
            adm_html += safe_chart(chart_admission_breakdown, row)

    return f"<h3>Variant comparison</h3>{table}{chart}{adm_html}"


def build_symptom() -> str:
    """
    Symptom Teacher section — previously entirely absent from this report.
    Mirrors build_teacher()'s pattern: training curve, then a LAB-comparison
    table+chart (Table 4.13), then qualitative overlay galleries for
    Figures 4.16-4.18, all produced by validate_symptom.py.
    """
    html = ""

    # Training curve
    curve_csv = LOGS_DIR / "symptom_teacher_metrics.csv"
    if curve_csv.exists():
        df = pd.read_csv(curve_csv)
        if not df.empty:
            html += "<h3>Training curves</h3>"
            html += safe_chart(
                chart_training_curve, df,
                ["train_loss", "val_iou", "msv_iou", "mln_iou", "healthy_act"],
                "Symptom Teacher Training")
    else:
        html += "<p style='color:#666;font-size:0.85rem'>symptom_teacher_metrics.csv not found — run train_symptom_model.py first.</p>"

    # LAB comparison (Table 4.13)
    lab_csv = REPORTS_DIR / "symptom_vs_lab_comparison.csv"
    if lab_csv.exists():
        ldf = pd.read_csv(lab_csv)
        if not ldf.empty:
            html += "<h3>Symptom Teacher vs. legacy LAB/HSV (vs. human ground truth)</h3>"

            overall_lab = ldf["iou_lab"].mean()
            overall_teacher = ldf["iou_symptom_teacher"].mean()
            improved = overall_teacher > overall_lab
            html += f"""<div class="grid3">
{metric_card(f"{overall_lab:.3f}", "Mean IoU — Legacy LAB", "")}
{metric_card(f"{overall_teacher:.3f}", "Mean IoU — Symptom Teacher", "good" if improved else "bad")}
{metric_card(f"{'+' if improved else ''}{(overall_teacher - overall_lab):.3f}", "Improvement over LAB", "good" if improved else "bad")}
</div>"""

            classes = sorted(ldf["category"].unique().tolist())
            chart_rows = []
            for cls in classes:
                cdf = ldf[ldf["category"] == cls]
                chart_rows.append({
                    "class": cls,
                    "Legacy LAB": cdf["iou_lab"].mean(),
                    "Symptom Teacher": cdf["iou_symptom_teacher"].mean(),
                })
            chart_rows.append({
                "class": "Overall",
                "Legacy LAB": overall_lab,
                "Symptom Teacher": overall_teacher,
            })
            chart_df = pd.DataFrame(chart_rows)
            html += safe_chart(
                chart_bar_comparison, chart_df, "class",
                ["Legacy LAB", "Symptom Teacher"],
                "Symptom Teacher vs. Legacy LAB — Mean IoU by Class", ylabel="Mean IoU")
        else:
            html += "<p style='color:#666;font-size:0.85rem'>symptom_vs_lab_comparison.csv is empty.</p>"
    else:
        html += "<p style='color:#666;font-size:0.85rem'>LAB comparison not yet run — run validate_symptom.py.</p>"

    # Qualitative overlays — Figures 4.16, 4.17, 4.18
    overlay_dir = REPORTS_DIR / "symptom_overlays"
    if overlay_dir.exists():
        overlay_specs = [
            ("*_ae_anomaly.jpg", "HealthyAE reconstruction and anomaly map (Figure 4.16)"),
            ("*_human_vs_pred.jpg", "Human symptom mask vs. predicted mask (Figure 4.17)"),
            ("*_lab_vs_teacher.jpg", "Symptom Teacher vs. LAB/HSV comparison (Figure 4.18)"),
        ]
        for pattern, heading in overlay_specs:
            imgs = sorted(overlay_dir.glob(pattern))[:6]
            if not imgs:
                continue
            html += f"<h3>{heading}</h3><div class='overlay-grid'>"
            for p in imgs:
                b64 = _img_to_b64(p)
                if b64:
                    html += f'<div class="overlay-card"><img src="data:image/jpeg;base64,{b64}"><div class="overlay-lbl">{p.stem}</div></div>'
            html += "</div>"
    else:
        html += "<p style='color:#666;font-size:0.85rem'>No overlay figures found — run validate_symptom.py.</p>"

    return html


def build_factory() -> str:
    """
    Factory pseudo-labeling QA section — previously had zero standalone
    representation anywhere in this report. factory_filter_breakdown.csv
    (pass/reject counts) was not referenced at all before this function
    existed, and factory_summary.csv was only ever used inside
    build_student_mode() for a mode-ablation severity chart, never to show
    what the Factory pass rate actually was.
    """
    html = ""

    breakdown_csv = REPORTS_DIR / "factory_filter_breakdown.csv"
    if breakdown_csv.exists():
        bdf = pd.read_csv(breakdown_csv)
        if not bdf.empty:
            total = bdf["count"].sum()
            processed_row = bdf[bdf["status"] == "processed"]
            processed_pct = float(processed_row["pct"].iloc[0]) if not processed_row.empty else 0.0
            quality = "good" if processed_pct >= 90 else "warn" if processed_pct >= 75 else "bad"

            html += f"""<div class="grid3">
{metric_card(f"{processed_pct:.1f}%", "Images successfully pseudo-labeled", quality)}
{metric_card(f"{int(total):,}", "Total Tier 2 images evaluated")}
{metric_card(f"{int(total - (processed_row['count'].iloc[0] if not processed_row.empty else 0)):,}", "Rejected (all reasons combined)")}
</div>"""
            html += "<h3>Rejection breakdown by reason</h3>"
            html += df_to_table(bdf, highlight_col="count")

            # Small horizontal bar chart of the breakdown
            plt = _get_mpl()
            import matplotlib.pyplot as _plt
            fig, ax = _plt.subplots(figsize=(7, max(2.5, 0.4 * len(bdf))))
            bars_df = bdf.sort_values("count", ascending=True)
            colors = ["#2ecc71" if s == "processed" else "#e74c3c"
                     for s in bars_df["status"]]
            ax.barh(bars_df["status"], bars_df["count"], color=colors, alpha=0.85)
            for i, (v, p) in enumerate(zip(bars_df["count"], bars_df["pct"])):
                ax.text(v, i, f"  {v:,} ({p}%)", va="center", fontsize=8)
            ax.set_xlabel("Image count")
            ax.set_title("Factory Pseudo-Labeling — Status Breakdown")
            fig.tight_layout()
            b64 = _fig_to_b64(fig)
            _plt.close(fig)
            html += chart_img(b64, "factory status breakdown")
    else:
        html += ("<p style='color:#666;font-size:0.85rem'>"
                 "factory_filter_breakdown.csv not found — run factory_master.py first.</p>")

    # Per-class snapshot for the deployed reference mode (mode_b), distinct
    # from build_student_mode()'s cross-mode ablation chart, which uses this
    # same CSV for a different comparison (mode impact on Student performance,
    # not Factory's own pass-rate health).
    summary_csv = REPORTS_DIR / "factory_summary.csv"
    if summary_csv.exists():
        sdf = pd.read_csv(summary_csv)
        mode_b = sdf[sdf["mode"] == "mode_b"] if "mode" in sdf.columns else pd.DataFrame()
        if not mode_b.empty:
            display_cols = [c for c in ["class", "n_total", "pct_processed",
                                          "mean_severity", "pct_symptomatic",
                                          "pct_excluded"]
                            if c in mode_b.columns]
            html += "<h3>Per-class snapshot (mode_b, the deployed reference mode)</h3>"
            html += df_to_table(mode_b[display_cols])
            html += ('<div class="note">Full cross-mode comparison (mode_a through '
                    'mode_d) and its effect on downstream Student performance is '
                    'shown in the Student Mode Ablation section below, not repeated '
                    'here.</div>')

    return html


def build_teacher() -> str:
    comp_csv = LOGS_DIR / "teacher_comparison.csv"
    test_csv = LOGS_DIR / "teacher_test_metrics.csv"

    html = ""
    if comp_csv.exists():
        df = pd.read_csv(comp_csv)
        display = [c for c in ["variant","best_dice","lat_cpu_ms"] if c in df.columns]
        html += f"<h3>Variant comparison (val Dice)</h3>{df_to_table(df[display], highlight_col='best_dice')}"
        num_df = df.copy()
        for c in ["best_dice"]:
            if c in num_df.columns:
                num_df[c] = pd.to_numeric(num_df[c], errors="coerce")
        if "variant" in num_df.columns and "best_dice" in num_df.columns:
            html += safe_chart(
                chart_bar_comparison, num_df, "variant", ["best_dice"],
                "Teacher Variants — Validation Dice")

    if test_csv.exists():
        df = pd.read_csv(test_csv)
        if not df.empty:
            row = df.iloc[0]
            html += f"""<h3>Test-split evaluation (Tier 1 held-out)</h3>
<div class="grid3">
{metric_card(row.get('test_dice','N/A'), "Test Dice", "good")}
{metric_card(row.get('test_iou','N/A'), "Test IoU", "good")}
{metric_card(row.get('test_recall','N/A'), "Test Recall (Sensitivity)", "good")}
</div>"""

    # Training curves for best Teacher variant
    best_teacher = "efficientnet-b2"
    curve_csv = LOGS_DIR / f"teacher_{best_teacher}_metrics.csv"
    if curve_csv.exists():
        df = pd.read_csv(curve_csv)
        html += f"<h3>Training curves — {best_teacher}</h3>"
        html += safe_chart(
            chart_training_curve, df, ["val_dice","val_iou","val_recall",
                                        "val_specificity","train_loss"],
            f"Teacher Training — {best_teacher}")

    # ── Gold-standard IoU (human ground truth) ──────────────────────────────
    # Previously missing entirely from this report — validate_gold_standard.py
    # already produces this data, it just wasn't being read here.
    gold_csv = REPORTS_DIR / "gold_standard_iou_summary.csv"
    if gold_csv.exists():
        gdf = pd.read_csv(gold_csv)
        if not gdf.empty:
            html += "<h3>Gold-standard validation (501 human-annotated images)</h3>"

            teacher_row = gdf[gdf["artifact"] == "teacher"]
            if not teacher_row.empty:
                r = teacher_row.iloc[0]
                target_met = str(r.get("target_met", "")).lower() in ("true", "1")
                html += f"""<div class="grid3">
{metric_card(f"{r.get('overall_mean_iou','N/A')}", "Overall Mean IoU (vs. human masks)", "good" if target_met else "bad")}
{metric_card(f"{r.get('n_below_warn','N/A')} / {r.get('n_total','N/A')}", "Images below 0.75 warn threshold", "")}
{metric_card("Target met" if target_met else "Below target", "Target: mean IoU ≥ 0.85", "good" if target_met else "bad")}
</div>"""

            # Grouped bar: mean IoU per class, one series per artifact
            # (sam2 / teacher / student — whichever are present)
            classes = ["HEALTHY", "MSV", "MLN", "overall"]
            artifacts_present = gdf["artifact"].tolist()
            chart_rows = []
            for cls in classes:
                row = {"class": cls}
                for art in artifacts_present:
                    art_row = gdf[gdf["artifact"] == art].iloc[0]
                    key = f"{cls}_mean_iou" if cls != "overall" else "overall_mean_iou"
                    row[art] = art_row.get(key, None)
                chart_rows.append(row)
            chart_df = pd.DataFrame(chart_rows)
            if len(artifacts_present) > 0:
                html += safe_chart(
                    chart_bar_comparison, chart_df, "class", artifacts_present,
                    "Gold-Standard IoU by Class (vs. human ground truth)",
                    ylabel="Mean IoU")
        else:
            html += "<p style='color:#666;font-size:0.85rem'>gold_standard_iou_summary.csv is empty.</p>"
    else:
        html += "<p style='color:#666;font-size:0.85rem'>Gold-standard validation not yet run — run validate_gold_standard.py.</p>"

    # Overlays
    # FIX: validate_gold_standard.py saves overlays to REPORTS_DIR /
    # "gold_standard_overlays" (confirmed against that script), not
    # "teacher_overlays" — this directory never existed under the old name,
    # so this section always silently rendered nothing.
    overlay_dir = REPORTS_DIR / "gold_standard_overlays"
    if overlay_dir.exists():
        imgs = sorted(overlay_dir.glob("*_teacher_overlay.jpg"))[:6]
        if imgs:
            html += "<h3>Qualitative leaf silhouette overlays (vs. human ground truth)</h3><div class='overlay-grid'>"
            for p in imgs:
                b64 = _img_to_b64(p)
                if b64:
                    html += f'<div class="overlay-card"><img src="data:image/jpeg;base64,{b64}"><div class="overlay-lbl">{p.stem}</div></div>'
            html += "</div>"

        # SAM2 raw pseudo-mask overlays, saved by the same validate_gold_standard.py
        # run but previously never displayed anywhere — shown here alongside the
        # Teacher overlays above so the improvement discussed numerically in
        # Table 4.11 can also be seen directly, image by image.
        sam2_imgs = sorted(overlay_dir.glob("*_sam2_overlay.jpg"))[:6]
        if sam2_imgs:
            html += "<h3>SAM2 raw pseudo-mask overlays (for direct comparison with the Teacher overlays above)</h3><div class='overlay-grid'>"
            for p in sam2_imgs:
                b64 = _img_to_b64(p)
                if b64:
                    html += f'<div class="overlay-card"><img src="data:image/jpeg;base64,{b64}"><div class="overlay-lbl">{p.stem}</div></div>'
            html += "</div>"

    return html


def build_student_encoder() -> str:
    csv_path = LOGS_DIR / "student_comparison_stage1.csv"
    if not csv_path.exists():
        return "<p>Stage 1 comparison not found. Run train_student.py --stage 1.</p>"

    df = pd.read_csv(csv_path)
    display = [c for c in ["encoder","mode","best_composite",
                             "test_sil_mIoU","test_msv_f1","test_mln_f1",
                             "test_msv_roc_auc","test_cls_accuracy",
                             "test_cpu_lat_mean_ms","test_cpu_fps"]
               if c in df.columns]
    df_num = df.copy()
    for c in display[2:]:
        if c in df_num.columns:
            df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

    table = df_to_table(df[display], highlight_col="best_composite")

    enc_col = "encoder" if "encoder" in df.columns else "variant"
    chart = ""
    metric_cols = [c for c in ["best_composite","test_sil_mIoU","test_msv_f1"]
                   if c in df_num.columns]
    if metric_cols:
        chart = chart_img(
            chart_bar_comparison(df_num, enc_col, metric_cols,
                                  "Student Encoder Ablation — Key Metrics"),
            "student encoder comparison")

    # Mobile ranking table
    mob_html = ""
    if MOBILE_RANKED_CSV.exists():
        import pandas as _pd
        mdf = _pd.read_csv(MOBILE_RANKED_CSV)
        if not mdf.empty:
            disp_cols = [c for c in [
                "variant","mode","mobile_composite","msv_f1","mln_f1",
                "msv_roc_auc","sil_mIoU",
                "cpu_lat_mean_ms","cpu_fps","tflite_size_mb_est","tflite_compatible"
            ] if c in mdf.columns]
            if disp_cols:
                mdf_disp = mdf[disp_cols].copy()
                for c in ["mobile_composite","msv_f1","mln_f1","msv_roc_auc","sil_mIoU"]:
                    if c in mdf_disp.columns:
                        mdf_disp[c] = _pd.to_numeric(mdf_disp[c], errors="coerce")
                mob_html = (
                    "<h3>Mobile-aware ranking (MSV_F1×0.32 + ROC_AUC×0.18 + mIoU×0.18 + Speed×0.16 + MLN_F1×0.08 + Sev×0.05 + Size×0.03)</h3>"
                    + df_to_table(mdf_disp, highlight_col="mobile_composite")
                )

    note = '<div class="note">Stage 1: all 5 encoder variants trained on Factory Mode B. Best encoder selected by composite score (0.40×mIoU + 0.35×MSV_F1 + 0.15×MLN_F1 + 0.10×(1−NormMAE)). Final deployment model selected by mobile composite.</div>'
    return note + table + chart + mob_html


def build_student_mode() -> str:
    csv_path = LOGS_DIR / "student_comparison_stage2.csv"
    if not csv_path.exists():
        return "<p>Stage 2 comparison not found. Run train_student.py --stage 2.</p>"

    df = pd.read_csv(csv_path)
    display = [c for c in ["mode","encoder","best_composite",
                             "test_sil_mIoU","test_sym_mIoU","test_msv_f1",
                             "test_mln_f1","test_msv_roc_auc",
                             "test_sev_mae_pct","test_sev_r2"]
               if c in df.columns]
    df_num = df.copy()
    for c in display[2:]:
        if c in df_num.columns:
            df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

    table = df_to_table(df[display], highlight_col="best_composite")
    chart = ""
    metric_cols = [c for c in ["best_composite","test_sil_mIoU","test_msv_f1",
                                "test_mln_f1","test_msv_roc_auc"]
                   if c in df_num.columns]
    if "mode" in df_num.columns and metric_cols:
        chart = safe_chart(
            chart_bar_comparison, df_num, "mode", metric_cols,
            "Factory Mode Ablation — Impact on Student Performance")

    # Severity distribution per mode
    fac_csv = REPORTS_DIR / "factory_summary.csv"
    sev_chart = ""
    if fac_csv.exists():
        fdf = pd.read_csv(fac_csv)
        sev_chart = safe_chart(
            chart_severity_distribution, fdf,
            "Factory Pseudo-Label Severity Distribution per Mode × Class")

    note = '<div class="note">Stage 2: best encoder trained on all 4 Factory modes (A=Otsu / B=SAM2 hard / C=SAM2 soft silhouette / D=full soft). Mode D uses soft HSV confidence as symptom mask target.</div>'
    return note + table + chart + ("<h3>Severity distribution per mode</h3>" + sev_chart if sev_chart else "")


def build_student_best() -> str:
    # Find best test metrics file
    best_csv = None
    best_score = -1.0
    for f in LOGS_DIR.glob("student_test_metrics_*.csv"):
        df = pd.read_csv(f)
        if df.empty:
            continue
        score = pd.to_numeric(df.iloc[0].get("composite", 0), errors="coerce")
        if score > best_score:
            best_score = score
            best_csv = f

    if best_csv is None:
        return "<p>No student test metrics found. Run train_student.py first.</p>"

    df   = pd.read_csv(best_csv)
    row  = df.iloc[0]
    enc  = row.get("variant", "")
    mode = row.get("mode", "")

    def r(k): return row.get(k, "N/A")

    html = f"<div class='note'>Best model: <strong>{enc}</strong> | Mode: <strong>{mode}</strong> | Composite: <strong>{r('composite')}</strong></div>"

    # Overview cards
    html += f"""<div class="grid3">
{metric_card(r('sil_mIoU'),     "Silhouette mIoU")}
{metric_card(r('msv_f1'),       "MSV F1 (primary)", "good" if float(str(r('msv_f1')).replace('N/A','0') or 0) >= 0.80 else "warn")}
{metric_card(r('msv_roc_auc'),  "MSV ROC-AUC")}
</div>
<div class="grid3">
{metric_card(r('mln_f1'),       "MLN F1")}
{metric_card(r('cls_accuracy'), "Classification Accuracy")}
{metric_card(r('mcc'),          "Matthews Correlation (MCC)")}
</div>
<div class="grid3">
{metric_card(r('sym_mIoU'),     "Symptom mask mIoU")}
{metric_card(r('sev_mae_pct'),  "Severity MAE (%)")}
{metric_card(r('composite'),    "Quality Composite")}
</div>"""

    # Per-class table
    per_class = []
    for cls in CLASSES:
        lc = cls.lower()
        per_class.append({
            "Class":     cls,
            "Precision": r(f"{lc}_prec"),
            "Recall":    r(f"{lc}_rec"),
            "F1":        r(f"{lc}_f1"),
        })
    pc_df = pd.DataFrame(per_class)
    html += f"<h3>Per-class classification metrics (test split)</h3>{df_to_table(pc_df, highlight_col='F1')}"

    # Full metrics table
    seg_rows = [
        {"Metric":"Silhouette mIoU",      "Value":r("sil_mIoU")},
        {"Metric":"Silhouette Dice",       "Value":r("sil_dice")},
        {"Metric":"Silhouette Recall",     "Value":r("sil_recall")},
        {"Metric":"Silhouette Precision",  "Value":r("sil_precision")},
        {"Metric":"Silhouette Specificity","Value":r("sil_specificity")},
        {"Metric":"Symptom mIoU",          "Value":r("sym_mIoU")},
        {"Metric":"Symptom Dice",          "Value":r("sym_dice")},
        {"Metric":"Symptom Recall",        "Value":r("sym_recall")},
        {"Metric":"MSV ROC-AUC",           "Value":r("msv_roc_auc")},
        {"Metric":"MLN ROC-AUC",           "Value":r("mln_roc_auc")},
        {"Metric":"HEALTHY ROC-AUC",       "Value":r("healthy_roc_auc")},
        {"Metric":"Macro OvR AUC",         "Value":r("macro_roc_auc")},
        {"Metric":"MLN F1",                "Value":r("mln_f1")},
        {"Metric":"Cohen's Kappa",         "Value":r("cohen_kappa")},
        {"Metric":"Severity MAE %",        "Value":r("sev_mae_pct")},
        {"Metric":"Severity MSE %",        "Value":r("sev_mse_pct")},
        {"Metric":"Severity RMSE %",       "Value":r("sev_rmse_pct")},
        {"Metric":"Severity MAPE %",       "Value":r("sev_mape_pct")},
        {"Metric":"Severity R²",           "Value":r("sev_r2")},
        {"Metric":"Severity Pearson r",    "Value":r("sev_pearson")},
        {"Metric":"CPU Latency (ms)",      "Value":r("cpu_lat_mean_ms")},
        {"Metric":"CPU FPS",               "Value":r("cpu_fps")},
        {"Metric":"Eval Duration (s)",     "Value":r("eval_duration_s")},
    ]
    html += f"<h3>Complete test metrics</h3>{df_to_table(pd.DataFrame(seg_rows))}"

    # Confusion matrix
    cm_csv = LOGS_DIR / f"student_confusion_{enc}_{mode}.csv"
    if cm_csv.exists():
        cm_df  = pd.read_csv(cm_csv, index_col=0)
        cm_arr = cm_df.values.astype(int)
        html += f"<h3>Confusion matrix (test split)</h3>"
        html += safe_chart(
            chart_confusion_matrix, cm_arr, CLASSES,
            f"Confusion Matrix — {enc} / {mode}")

    # Training curve for best student
    curve_csv = LOGS_DIR / f"student_{enc}_{mode}_metrics.csv"
    if curve_csv.exists():
        cdf = pd.read_csv(curve_csv)
        html += "<h3>Training curves</h3>"
        html += safe_chart(
            chart_training_curve,
            cdf, ["sil_mIoU","sym_mIoU","msv_f1","mln_f1","composite","train_loss"],
            f"Student Training — {enc} / {mode}")

    # Qualitative mask-output predictions — previously the one component in
    # the pipeline with numeric test metrics but no visual output shown
    # anywhere. Produced by validate_student.py on the deployed checkpoint.
    overlay_dir = REPORTS_DIR / "student_overlays"
    if overlay_dir.exists():
        imgs = sorted(overlay_dir.glob("*_student_pred.jpg"))[:9]
        if imgs:
            html += ("<h3>Predicted mask output on sample images "
                     "(silhouette + symptom, deployed model)</h3>"
                     "<div class='overlay-grid'>")
            for p in imgs:
                b64 = _img_to_b64(p)
                if b64:
                    html += (f'<div class="overlay-card">'
                            f'<img src="data:image/jpeg;base64,{b64}">'
                            f'<div class="overlay-lbl">{p.stem}</div></div>')
            html += "</div>"
    else:
        html += ("<p style='color:#666;font-size:0.85rem'>No qualitative "
                 "prediction panels found — run validate_student.py.</p>")

    # Second, different overlay type: silhouette-only, from the SAM2 vs
    # Teacher vs Student IoU chain in validate_gold_standard.py (same source
    # as the Teacher and SAM2 overlays in build_teacher()). This shows only
    # the leaf silhouette against human ground truth, not the symptom mask
    # or classification — complementary to the fuller prediction panel
    # above, not a duplicate of it. Existed on disk already but was never
    # displayed anywhere in this report before now.
    iou_chain_dir = REPORTS_DIR / "gold_standard_overlays"
    if iou_chain_dir.exists():
        sil_imgs = sorted(iou_chain_dir.glob("*_student_overlay.jpg"))[:6]
        if sil_imgs:
            html += ("<h3>Predicted silhouette only, vs. human ground truth "
                     "(IoU chain comparison — see Table 4.11 / build_teacher "
                     "for the SAM2 and Teacher equivalents)</h3>"
                     "<div class='overlay-grid'>")
            for p in sil_imgs:
                b64 = _img_to_b64(p)
                if b64:
                    html += (f'<div class="overlay-card">'
                            f'<img src="data:image/jpeg;base64,{b64}">'
                            f'<div class="overlay-lbl">{p.stem}</div></div>')
            html += "</div>"

    return html


def build_xai() -> str:
    html = ""
    xai_sel_csv = LOGS_DIR / "xai_method_selection.csv"
    if xai_sel_csv.exists():
        sel_df = pd.read_csv(xai_sel_csv)
        if not sel_df.empty:
            sel = sel_df.iloc[0]
            html += (
                '<div class="metric-card good">'
                '<div class="metric-label">Selected XAI Method (auto)</div>'
                f'<div class="metric-value">{sel.get("selected_method","N/A")}</div>'
                f'<div class="metric-sub">MSV PG: {sel.get("msv_pg","N/A")} | '
                f'Ins AUC: {sel.get("ins_auc","N/A")}</div>'
                f'<div class="metric-sub">GradCAM: {sel.get("gradcam_pg","N/A")} | '
                f'GradCAM++: {sel.get("gradcamplusplus_pg","N/A")} | '
                f'ScoreCAM: {sel.get("scorecam_pg","N/A")}</div>'
                '</div>'
            )

    xai_csv = LOGS_DIR / "xai_comparison.csv"
    if not xai_csv.exists():
        return html + "<p>XAI comparison not found. Run evaluate_xai.py first.</p>"

    df  = pd.read_csv(xai_csv)

    # Summary table (mean per method)
    methods = df["method"].unique().tolist() if "method" in df.columns else []
    summary = []
    for m in methods:
        mdf = df[df["method"] == m]
        summary.append({
            "Method": m,
            "Pointing Game (mean)": round(pd.to_numeric(mdf["pointing_game_acc"],
                                                         errors="coerce").mean(), 4),
            "Insertion AUC (mean)": round(pd.to_numeric(mdf["insertion_auc"],
                                                          errors="coerce").mean(), 4),
            "Deletion AUC (mean)":  round(pd.to_numeric(mdf["deletion_auc"],
                                                          errors="coerce").mean(), 4),
        })
    if summary:
        html += f"<h3>Quantitative XAI comparison</h3>{df_to_table(pd.DataFrame(summary), highlight_col='Pointing Game (mean)')}"
        html += safe_chart(chart_xai_comparison, df)

    note = '''<div class="note">
<strong>Two distinct app outputs:</strong><br>
① <strong>Symptom boundary</strong> (green contour) — UNet segmentation head → pixel-level leaf/symptom localization<br>
② <strong>Diagnostic attention</strong> (amber heatmap) — Grad-CAM++ on last encoder block → class-discriminative explanation
</div>'''
    html += note

    # Embedded XAI overlays
    for method in ["gradcamplusplus", "gradcam", "scorecam"]:
        method_dir = REPORTS_DIR / "xai" / method
        if not method_dir.exists():
            continue
        imgs = sorted(method_dir.glob("*.jpg"))[:6]
        if not imgs:
            continue
        html += f"<h3>{method} overlays (sample)</h3><div class='overlay-grid'>"
        for p in imgs:
            b64 = _img_to_b64(p)
            if b64:
                html += f'<div class="overlay-card"><img src="data:image/jpeg;base64,{b64}"><div class="overlay-lbl">{p.stem}</div></div>'
        html += "</div>"

    return html


def build_severity() -> str:
    csv_path = LOGS_DIR / "severity_reliability.csv"
    if not csv_path.exists():
        return "<p>Severity reliability analysis not found. Run evaluate_severity.py --analyze first.</p>"

    df = pd.read_csv(csv_path)
    if df.empty:
        return "<p>No data in severity reliability CSV.</p>"
    row = df.iloc[0]

    kappa = float(str(row.get("cohen_kappa", 0)).replace("nan","0") or 0)
    rho   = float(str(row.get("spearman_rho", 0)).replace("nan","0") or 0)
    kappa_q = "good" if kappa >= 0.6 else "warn" if kappa >= 0.4 else "bad"
    rho_q   = "good" if abs(rho) >= 0.6 else "warn" if abs(rho) >= 0.4 else "bad"

    html = f"""<div class="grid3">
{metric_card(f"{kappa:.3f}",           "Cohen's Kappa (inter-rater)", kappa_q)}
{metric_card(f"{rho:.3f}",             "Spearman ρ (HSV vs human)",   rho_q)}
{metric_card(str(row.get('n_images','N/A')), "Images rated")}
</div>"""

    interp = ("moderate-to-strong" if abs(rho) >= 0.6 else
              "weak-to-moderate"   if abs(rho) >= 0.4 else "weak")
    html += f'<div class="note">HSV-derived severity scores showed <strong>{interp} correlation (ρ = {rho:.3f})</strong> with expert visual ratings. Severity MAE should be interpreted as consistency with the HSV-derived proxy, not absolute agronomic accuracy.</div>'

    tbl_df = pd.DataFrame([{k: v for k, v in row.items()}])
    html += df_to_table(tbl_df)

    # Per-image scatter — previously this section only showed the aggregate
    # kappa/rho numbers above; the actual point-by-point relationship those
    # numbers summarize was never shown as a figure.
    analysis_csv = REPORTS_DIR / "severity_analysis.csv"
    if analysis_csv.exists():
        adf = pd.read_csv(analysis_csv)
        if not adf.empty and "human_norm" in adf.columns and "hsv_norm" in adf.columns:
            html += "<h3>Human vs. HSV-derived severity (per image)</h3>"
            html += safe_chart(chart_severity_scatter_inline, adf)

    return html


def build_deployment() -> str:
    exp_csv = EXPORTS_DIR / "export_report.csv"
    if not exp_csv.exists():
        return "<p>Export report not found. Run export_tflite.py first.</p>"

    df  = pd.read_csv(exp_csv)
    row = df.iloc[0] if not df.empty else {}

    tflite_ok = str(row.get("tflite_conversion","")) == "success"
    bouncer_ok = (EXPORTS_DIR.parent / "deploy" / "bouncer_model.tflite").exists()

    html = f"""<div class="grid3">
{metric_card(f"{row.get('tflite_size_mb','N/A')} MB", "Student TFLite size", "good" if tflite_ok else "bad")}
{metric_card(str(row.get('cpu_lat_mean_ms','N/A'))+" ms", "Student CPU latency")}
{metric_card("✓" if tflite_ok else "✗", "TFLite conversion", "good" if tflite_ok else "bad")}
</div>"""

    display = [c for c in ["encoder_variant","factory_mode","n_parameters",
                             "tflite_size_mb","cpu_lat_mean_ms","cls_agreement",
                             "quantization","export_duration_s"]
               if c in row.index]
    if display:
        html += df_to_table(pd.DataFrame([{c: row[c] for c in display}]))

    deploy_dir = EXPORTS_DIR.parent / "deploy"
    if deploy_dir.exists():
        files = sorted(deploy_dir.iterdir())
        html += "<h3>Deployment package contents</h3><ul style='margin-top:0.5rem;font-size:0.875rem;color:#b0b0c0'>"
        for f in files:
            size = f"{f.stat().st_size/1024:.1f} KB" if f.is_file() else ""
            html += f"<li><code>{f.name}</code> {size}</li>"
        html += "</ul>"

    html += '<div class="note">Models deployed to <code>exports/deploy/</code>. Copy <code>student_model.tflite</code> and <code>bouncer_model.tflite</code> into your Android Studio project\'s <code>assets/</code> folder.</div>'
    return html


# ══════════════════════════════════════════════════════════════════════════════
# MAIN REPORT ASSEMBLER
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import time as _t
    _start = _t.time()

    print("=" * 65)
    print("  Yellow MAIze | Evaluation Report Generator")
    print("=" * 65)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ts = _t.strftime("%Y-%m-%d %H:%M:%S")

    toc = """<div class="toc"><strong style="color:#4fc3f7">Contents</strong><ol style="margin-top:0.5rem;padding-left:1.5rem">
<li><a href="#preproc">Preprocessing Summary</a></li>
<li><a href="#bouncer">Bouncer Gate Comparison</a></li>
<li><a href="#teacher">Teacher Model Comparison</a></li>
<li><a href="#symptom">Symptom Teacher Comparison</a></li>
<li><a href="#factory">Factory Pseudo-Labeling QA</a></li>
<li><a href="#enc-abl">Student Encoder Ablation (Stage 1)</a></li>
<li><a href="#mode-abl">Student Mode Ablation (Stage 2)</a></li>
<li><a href="#best">Best Student — Full Test Results</a></li>
<li><a href="#xai">XAI Comparison</a></li>
<li><a href="#severity">Severity Reliability</a></li>
<li><a href="#deploy">Deployment Summary</a></li>
</ol></div>"""

    print("  Building report sections ...")
    section_defs = [
        ("1. Preprocessing Summary",            "preproc",   build_overview),
        ("2. Bouncer Gate Comparison",           "bouncer",   build_bouncer),
        ("3. Teacher Model Comparison",          "teacher",   build_teacher),
        ("4. Symptom Teacher Comparison",        "symptom",   build_symptom),
        ("5. Factory Pseudo-Labeling QA",        "factory",   build_factory),
        ("6. Student Encoder Ablation",          "enc-abl",   build_student_encoder),
        ("7. Student Pseudo-Label Mode Ablation","mode-abl",  build_student_mode),
        ("8. Best Student — Full Test Results",  "best",      build_student_best),
        ("9. XAI Method Comparison",             "xai",       build_xai),
        ("10. Severity Reliability Analysis",    "severity",  build_severity),
        ("11. Deployment Summary",               "deploy",    build_deployment),
    ]
    sections = []
    for title, anchor, builder in section_defs:
        print(f"    {title} ...", end=" ", flush=True)
        try:
            content_html = builder()
            sections.append(section(title, anchor, content_html))
            print("done")
        except Exception as e:
            sections.append(section(title, anchor,
                f"<div class='warn-box'>Section unavailable: {str(e)[:200]}</div>"))
            print(f"WARN: {e}")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Yellow MAIze — Evaluation Report</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
<h1>Yellow MAIze — Evaluation Report</h1>
<p class="subtitle">Generated: {ts} &nbsp;|&nbsp; MAIze: MobileNetV2-UNet + Grad-CAM XAI Framework for Maize Disease Detection</p>
{toc}
{"".join(sections)}
<p style="text-align:center;color:#4a4a7a;font-size:0.8rem;margin-top:2rem">
Yellow MAIze Thesis Pipeline &nbsp;|&nbsp; Generated in {round(_t.time()-_start,1)}s
</p>
</div>
</body>
</html>"""

    REPORT_PATH.write_text(html, encoding="utf-8")
    size_kb = round(REPORT_PATH.stat().st_size / 1024, 1)
    print(f"\n  Report generated: {REPORT_PATH}")
    print(f"  File size       : {size_kb} KB (self-contained)")
    print(f"  Duration        : {round(_t.time()-_start,1)}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
