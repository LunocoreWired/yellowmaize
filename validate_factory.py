"""
validate_factory.py — Post-Factory Pseudo-Label Visual Validator
Yellow MAIze Project | Phase 4 QA Tool

PURPOSE:
    Run this AFTER factory_master.py to visually audit whether the generated
    pseudo-masks correctly capture symptoms for HEALTHY, MSV, and MLN classes
    on yellow maize leaf images.

    Output: an HTML report showing raw image vs. pseudo-mask side by side,
    with per-image diagnostics and a global pass/fail summary.

USAGE:
    python validate_factory.py                        # auto-picks 10 images per class
    python validate_factory.py --n 20                 # 20 images per class
    python validate_factory.py --mode mode_b          # validate a specific factory mode
    python validate_factory.py --img path/to/img.jpg  # validate a single image
    python validate_factory.py --all-modes            # compare all 4 modes side by side

WHAT TO LOOK FOR (ground truth from literature):
─────────────────────────────────────────────────────────────────────────────
CLASS     | EXPECTED MASK BEHAVIOR
─────────────────────────────────────────────────────────────────────────────
HEALTHY   | Mask should be MOSTLY EMPTY (near-black).
          | Yellow maize leaves naturally have warm-yellow color — the
          | stricter HEALTHY thresholds (a≥140, b≥145) + the 3-band area
          | guard (zero <1%, dampen 1-4%, keep ≥4%) should suppress false
          | positives. If you see >4% mask fill on clearly healthy leaves
          | → LAB_GREEN_A_MAX or HEALTHY thresholds need tightening.
─────────────────────────────────────────────────────────────────────────────
MSV       | Mask should highlight NARROW, BROKEN STREAKS running parallel
          | to the veins. Streaks range from pale green through yellow to
          | white (eLife 2020). On yellow maize the streaks appear as
          | slightly paler/whiter interruptions of the yellow background.
          | Gabor filter (θ=0°,45°,90°,135°) should reinforce vertical
          | streak orientation. PROBLEMS TO FLAG:
          |   → Mask fills entire leaf uniformly (green-exclusion failing)
          |   → Mask is empty on visibly streaky leaves (LAB thresholds
          |     too strict or Gabor killing too much)
          |   → Mask covers midrib only (Gabor orientation bias)
─────────────────────────────────────────────────────────────────────────────
MLN       | Mask should highlight WIDER, MORE DIFFUSE yellowing + necrotic
          | brown/dark patches drying from leaf MARGINS toward midrib.
          | Unlike MSV streaks, MLN coverage tends to be broader and less
          | vein-aligned. Dark necrotic tissue (L*<110, a*≥125) should
          | also be caught by the dark_necrosis branch. PROBLEMS TO FLAG:
          |   → Only outer margin highlighted (dark_necrosis branch not
          |     firing on central necrosis)
          |   → Mask looks identical to MSV (directional kernel difference
          |     (1,15) vs (1,7) not distinguishing patterns)
          |   → Completely empty on brown/necrotic leaves
─────────────────────────────────────────────────────────────────────────────

ADJUSTMENT GUIDE (what knob to turn):
─────────────────────────────────────────────────────────────────────────────
PROBLEM                         | PARAMETER TO CHANGE            | FILE
─────────────────────────────────────────────────────────────────────────────
Too many false positives on     | Raise LAB_MSV_A_MIN (133→135)  | factory_master.py
healthy yellow leaves (MSV)     | Raise LAB_MSV_B_MIN (135→138)  |
─────────────────────────────────────────────────────────────────────────────
MSV streaks missed on yellow    | Lower LAB_MSV_B_MIN (135→130)  | factory_master.py
maize (mask too sparse)         | Lower GABOR_THRESHOLD (0.3→0.2)| config.py
─────────────────────────────────────────────────────────────────────────────
Gabor kills too many streaks    | Lower GABOR_THRESHOLD (0.3→0.2)| config.py
(soft-weight product <0.4)      | Or shorten GABOR_LAMBDA (10→8) |
─────────────────────────────────────────────────────────────────────────────
Directional opening destroys    | Shorten kernel: (1,7)→(1,5)    | factory_master.py
short MSV streaks               | in compute_lab_hard_mask MSV    |
─────────────────────────────────────────────────────────────────────────────
MLN necrosis (dark patches)     | Lower dark_necrosis L* ceiling  | factory_master.py
not caught                      | (110→120) or lower a* floor     |
                                | (125→120)                       |
─────────────────────────────────────────────────────────────────────────────
HEALTHY mask non-zero too often | Raise HEALTHY thresholds:       | factory_master.py
(warm-yellow false positives)   | a≥140→145, b≥145→150           |
─────────────────────────────────────────────────────────────────────────────
Midrib bright stripe captured   | LAB_GREEN_A_MAX already raised  | factory_master.py
as symptom                      | 124→121; try 118 if persists    |
─────────────────────────────────────────────────────────────────────────────
Silhouette too small / cuts     | Lower FACTORY_SILHOUETTE_       | config.py
into leaf edges                 | THRESHOLD (0.35→0.25)           |
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import argparse
import random
import json
import base64
from pathlib import Path
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np

# ── Try importing config from the project root ─────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
try:
    from config import (
        PSEUDO_DIR, GLOBAL_MANIFEST, FACTORY_MODES, CLASSES,
        STUDENT_IMG_SIZE, REPORTS_DIR,
    )
except ImportError:
    print("[FATAL] config.py not found. Run from the project root or add it to PYTHONPATH.")
    sys.exit(1)

# Inline LAB/config constants mirrored from factory_master.py for display
# (update these if you change them in factory_master.py)
DISPLAY_CONSTANTS = {
    "LAB_MSV_A_MIN":   133,
    "LAB_MSV_B_MIN":   135,
    "LAB_MLN_A_MIN":   130,
    "LAB_MLN_B_MIN":   138,
    "LAB_GREEN_A_MAX": 121,
    "HEALTHY_A_MIN":   140,
    "HEALTHY_B_MIN":   145,
    "FACTORY_SILHOUETTE_THRESHOLD": 0.35,
    "FACTORY_MIN_LEAF_COVERAGE":    0.15,
    "GABOR_THRESHOLD":              0.30,
    "GABOR_LAMBDA":                 10.0,
    "MSV_DIR_KERNEL":               "(1, 7)",
    "MLN_DIR_KERNEL":               "(1, 15)",
    "HEALTHY_3BAND_NOISE_FLOOR":    "< 1%  → zero | 1-4% → ×0.3 | ≥ 4% → keep",
    "MSV_CIMMYT_SCALE":             "1=<5% | 3=5-25% | 5=25-50% | 7=50-75% | 9=>75%",
    "MLN_CIMMYT_SCALE":             "1=<10% | 2=10-25% | 3=25-50% | 4=50-75% | 5=>75%",
    "OTSU_GUIDED_THRESH":           "Otsu per-image, clamped to fixed LAB floor",
    "CLAHE_L_LEAF_ONLY":            "clipLimit=2.0, tileGrid=(4,4), leaf mask only",
}

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _img_to_b64(arr: np.ndarray, quality: int = 85) -> str:
    """Convert an RGB or grayscale uint8 array to a base64 JPEG string."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        bgr = arr
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _load_pseudo_mask(stem: str, mode: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load silhouette (.npy) and symptom mask (png or .npy) for a given stem+mode.
    Returns (silhouette_arr, symptom_arr) both as uint8 [0,255] or None.
    """
    mode_dir = PSEUDO_DIR / mode
    sil_path = mode_dir / f"{stem}_silhouette.npy"
    sym_png   = mode_dir / f"{stem}_symptom.png"
    sym_npy   = mode_dir / f"{stem}_symptom.npy"
    sev_txt   = mode_dir / f"{stem}_sev.txt"
    weight_txt = mode_dir / f"{stem}_weight.txt"

    sil = None
    if sil_path.exists():
        raw = np.load(str(sil_path))
        sil = np.clip(raw * 255, 0, 255).astype(np.uint8)

    sym = None
    if sym_png.exists():
        sym = cv2.imread(str(sym_png), cv2.IMREAD_GRAYSCALE)
    elif sym_npy.exists():
        raw = np.load(str(sym_npy))
        if raw.max() <= 1.0:
            raw = raw * 255
        sym = np.clip(raw, 0, 255).astype(np.uint8)

    severity = float(sev_txt.read_text().strip()) if sev_txt.exists() else -1.0
    weight   = float(weight_txt.read_text().strip()) if weight_txt.exists() else -1.0
    grade_txt = mode_dir / f"{stem}_grade.txt"
    grade     = int(grade_txt.read_text().strip()) if grade_txt.exists() else -1

    return sil, sym, severity, weight, grade


def _overlay_symptom_on_raw(raw_rgb: np.ndarray, symptom: np.ndarray,
                              silhouette: np.ndarray | None,
                              alpha: float = 0.55) -> np.ndarray:
    """
    Create a colour overlay:
      - Symptom pixels → red tint
      - Silhouette (leaf area) boundary → thin green border
      - Rest → darkened background
    """
    out = raw_rgb.copy().astype(np.float32)

    # Dim background slightly
    out *= 0.75

    if silhouette is not None:
        sil_bin = (silhouette > 127).astype(np.uint8)
        out[sil_bin == 1] = raw_rgb[sil_bin == 1].astype(np.float32)

    if symptom is not None:
        sym_bin = (symptom > 30).astype(np.uint8)
        red_layer = np.zeros_like(out)
        red_layer[sym_bin == 1] = [255, 60, 60]
        out = out * (1 - alpha * sym_bin[:, :, None].astype(np.float32)) + \
              red_layer * alpha * sym_bin[:, :, None].astype(np.float32)

    # Silhouette boundary
    if silhouette is not None:
        sil_bin = (silhouette > 127).astype(np.uint8)
        contours, _ = cv2.findContours(sil_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out_u8 = np.clip(out, 0, 255).astype(np.uint8)
        cv2.drawContours(out_u8, contours, -1, (0, 220, 80), 1)
        return out_u8

    return np.clip(out, 0, 255).astype(np.uint8)


def _symptom_stats(symptom: np.ndarray, silhouette: np.ndarray | None) -> dict:
    """Compute coverage stats for a symptom mask."""
    if symptom is None:
        return {"sym_pct": 0.0, "sym_px": 0, "leaf_px": 0}
    sym_bin = (symptom > 30).astype(np.uint8)
    sym_px  = int(sym_bin.sum())
    leaf_px = int((silhouette > 127).sum()) if silhouette is not None else int(symptom.size)
    sym_pct = 100.0 * sym_px / max(leaf_px, 1)
    return {"sym_pct": round(sym_pct, 2), "sym_px": sym_px, "leaf_px": leaf_px}


def _auto_flag(category: str, stats: dict, severity: float) -> tuple[str, str]:
    """
    Return (flag_emoji, diagnostic_message) based on expected behaviour per class.
    These are heuristic rules — use them as prompts for human review, not hard verdicts.
    """
    pct = stats["sym_pct"]

    if category == "HEALTHY":
        if pct < 1.0:
            return "✅", "Good — mask near-empty as expected for HEALTHY."
        elif pct < 4.0:
            return "⚠️", f"Borderline ({pct:.1f}% coverage). Should be confidence-dampened (×0.3). Check if leaf has faint early-stage symptoms or is a false positive."
        else:
            return "🔴", f"HIGH false-positive risk ({pct:.1f}%). HEALTHY mask too full. Raise HEALTHY_A_MIN/B_MIN or tighten LAB_GREEN_A_MAX."

    elif category == "MSV":
        if pct < 1.0:
            return "🔴", f"UNDER-DETECTION ({pct:.1f}%). MSV streaks likely missed. Lower LAB_MSV_B_MIN, reduce GABOR_THRESHOLD, or shorten directional kernel (1,7)→(1,5)."
        elif pct < 5.0:
            return "⚠️", f"Low coverage ({pct:.1f}%). May be early-stage MSV or marginal detection. Visually confirm streaks present in raw image."
        elif pct < 40.0:
            return "✅", f"Reasonable MSV coverage ({pct:.1f}%). Verify streaks are vein-parallel, not solid fill."
        else:
            return "🔴", f"OVER-DETECTION ({pct:.1f}%). Mask covering most of leaf — likely green-exclusion failure or threshold too loose. Check LAB_GREEN_A_MAX."

    elif category == "MLN":
        if pct < 2.0:
            return "🔴", f"UNDER-DETECTION ({pct:.1f}%). MLN yellowing/necrosis likely missed. Lower LAB_MLN_B_MIN or dark_necrosis L*<110 ceiling."
        elif pct < 10.0:
            return "⚠️", f"Low-moderate coverage ({pct:.1f}%). May be early-stage MLN. Check for margin drying in raw image."
        elif pct < 60.0:
            return "✅", f"Reasonable MLN coverage ({pct:.1f}%). Confirm necrosis is margin-inward and necrotic patches captured."
        else:
            return "🔴", f"OVER-DETECTION ({pct:.1f}%). Nearly whole leaf masked — possible green-exclusion failure or MLN thresholds too loose."

    return "❓", "Unknown category."


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

def _discover_images(n_per_class: int, mode: str, single_img: str | None) -> list[dict]:
    """
    Returns a list of dicts: {stem, source_path, category}
    Tries global manifest first; falls back to scanning PSEUDO_DIR.
    """
    entries = []

    if single_img:
        p = Path(single_img)
        if not p.exists():
            print(f"[ERROR] Image not found: {single_img}")
            return []
        stem = p.stem
        entries.append({"stem": stem, "source_path": str(p), "category": "UNKNOWN"})
        return entries

    # Try manifest
    if GLOBAL_MANIFEST.exists():
        import pandas as pd
        df = pd.read_csv(GLOBAL_MANIFEST)
        # Sample from train/val (factory only processes these)
        df = df[df["split"] != "test"].reset_index(drop=True)

        for cls in CLASSES:
            cls_df = df[df["category"] == cls].reset_index(drop=True)
            # Only keep images that have pseudo masks
            mode_dir = PSEUDO_DIR / mode
            cls_df = cls_df[cls_df["filename"].apply(
                lambda f: (mode_dir / f"{Path(f).stem}_symptom.png").exists() or
                           (mode_dir / f"{Path(f).stem}_symptom.npy").exists()
            )]
            n = min(n_per_class, len(cls_df))
            if n == 0:
                print(f"[WARN] No processed pseudo-masks found for {cls} in {mode_dir}")
                continue
            sampled = cls_df.sample(n=n, random_state=42)
            for _, row in sampled.iterrows():
                entries.append({
                    "stem": Path(row["filename"]).stem,
                    "source_path": row.get("source_path", row.get("filename", "")),
                    "category": cls,
                })
        return entries

    # Fallback: scan pseudo dir
    print("[WARN] global_split_manifest.csv not found. Scanning PSEUDO_DIR directly.")
    mode_dir = PSEUDO_DIR / mode
    if not mode_dir.exists():
        print(f"[ERROR] Mode directory not found: {mode_dir}")
        return []

    seen_stems = set()
    for sym_file in mode_dir.glob("*_symptom.png"):
        stem = sym_file.stem.replace("_symptom", "")
        seen_stems.add(stem)
    for sym_file in mode_dir.glob("*_symptom.npy"):
        stem = sym_file.stem.replace("_symptom", "")
        seen_stems.add(stem)

    stems = list(seen_stems)
    random.shuffle(stems)
    for stem in stems[:n_per_class * len(CLASSES)]:
        # Guess category from stem name
        cat = "UNKNOWN"
        for c in CLASSES:
            if c.lower() in stem.lower():
                cat = c
                break
        entries.append({"stem": stem, "source_path": "", "category": cat})

    return entries


# ═══════════════════════════════════════════════════════════════════════════
# HTML REPORT BUILDER
# ═══════════════════════════════════════════════════════════════════════════

HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Yellow MAIze — Factory Pseudo-Label Validator</title>
<style>
  :root {{
    --bg:       #0d1117;
    --surface:  #161b22;
    --border:   #30363d;
    --text:     #e6edf3;
    --subtext:  #8b949e;
    --green:    #3fb950;
    --yellow:   #d29922;
    --red:      #f85149;
    --blue:     #58a6ff;
    --healthy:  #2ea043;
    --msv:      #d29922;
    --mln:      #f85149;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    font-size: 14px;
    line-height: 1.5;
  }}
  header {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 20px 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: sticky;
    top: 0;
    z-index: 100;
  }}
  header h1 {{
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.3px;
  }}
  header .meta {{ color: var(--subtext); font-size: 12px; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 32px; }}

  /* ── Summary Banner ── */
  .summary {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 32px;
  }}
  .summary h2 {{ font-size: 15px; font-weight: 600; margin-bottom: 12px; color: var(--blue); }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
  }}
  .stat-card {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 16px;
  }}
  .stat-card .label {{ color: var(--subtext); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .stat-card .value {{ font-size: 22px; font-weight: 700; margin-top: 2px; }}
  .good  {{ color: var(--green); }}
  .warn  {{ color: var(--yellow); }}
  .bad   {{ color: var(--red); }}

  /* ── Config Panel ── */
  .config-panel {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 24px;
    margin-bottom: 32px;
  }}
  .config-panel h2 {{ font-size: 14px; font-weight: 600; margin-bottom: 10px; color: var(--blue); }}
  .config-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 6px;
    font-family: 'Courier New', monospace;
    font-size: 12px;
  }}
  .config-row {{ display: flex; gap: 8px; }}
  .config-key {{ color: var(--subtext); min-width: 200px; }}
  .config-val {{ color: #79c0ff; }}

  /* ── Reference Guide ── */
  .ref-panel {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 24px;
    margin-bottom: 32px;
  }}
  .ref-panel h2 {{ font-size: 14px; font-weight: 600; margin-bottom: 12px; color: var(--blue); }}
  .ref-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
  .ref-card {{
    background: var(--bg);
    border-radius: 6px;
    padding: 14px;
    border-left: 3px solid;
  }}
  .ref-card.healthy {{ border-color: var(--healthy); }}
  .ref-card.msv     {{ border-color: var(--msv); }}
  .ref-card.mln     {{ border-color: var(--mln); }}
  .ref-card .cls-label {{ font-weight: 700; margin-bottom: 6px; font-size: 13px; }}
  .ref-card.healthy .cls-label {{ color: var(--healthy); }}
  .ref-card.msv .cls-label     {{ color: var(--msv); }}
  .ref-card.mln .cls-label     {{ color: var(--mln); }}
  .ref-card ul {{ padding-left: 16px; color: var(--subtext); font-size: 12px; }}
  .ref-card ul li {{ margin-bottom: 4px; }}

  /* ── Image Grid ── */
  .section-title {{
    font-size: 16px;
    font-weight: 700;
    margin: 28px 0 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .cls-badge {{
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
  }}
  .cls-badge.healthy {{ background: #143820; color: var(--healthy); }}
  .cls-badge.msv     {{ background: #2d2008; color: var(--msv); }}
  .cls-badge.mln     {{ background: #2d0e0e; color: var(--mln); }}

  .image-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
    gap: 20px;
    margin-bottom: 24px;
  }}
  .img-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }}
  .img-card .card-header {{
    padding: 10px 14px;
    background: #1a2030;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
  }}
  .img-card .stem {{ font-family: monospace; font-size: 11px; color: var(--subtext); word-break: break-all; }}
  .img-card .flag {{ font-size: 16px; }}
  .panel-row {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 0;
  }}
  .panel {{ position: relative; }}
  .panel img {{
    width: 100%;
    display: block;
    aspect-ratio: 1;
    object-fit: cover;
  }}
  .panel-label {{
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(0,0,0,0.7);
    font-size: 10px;
    text-align: center;
    padding: 3px;
    color: #ccc;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .card-diag {{
    padding: 10px 14px;
    font-size: 12px;
    border-top: 1px solid var(--border);
  }}
  .diag-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 4px; }}
  .diag-row .kv {{ display: flex; gap: 4px; }}
  .diag-row .k {{ color: var(--subtext); }}
  .diag-row .v {{ color: var(--text); font-weight: 600; font-family: monospace; }}
  .diag-msg {{ color: var(--subtext); font-size: 11px; margin-top: 4px; }}

  /* Adjust guide table */
  .adjust-panel {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 24px;
    margin-top: 32px;
  }}
  .adjust-panel h2 {{ font-size: 14px; font-weight: 600; margin-bottom: 12px; color: var(--blue); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{ background: var(--bg); padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); color: var(--subtext); font-weight: 600; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  td:nth-child(2) {{ font-family: monospace; color: #79c0ff; }}
  td:nth-child(3) {{ font-family: monospace; color: var(--subtext); }}

  footer {{
    text-align: center;
    padding: 24px;
    color: var(--subtext);
    font-size: 11px;
    border-top: 1px solid var(--border);
    margin-top: 40px;
  }}
</style>
</head>
<body>
"""

def _build_html_report(entries_data: list[dict], mode: str, all_modes: list[str]) -> str:
    """Build the full HTML validation report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_total = len(entries_data)
    n_good  = sum(1 for e in entries_data if e["flag"] == "✅")
    n_warn  = sum(1 for e in entries_data if e["flag"] == "⚠️")
    n_bad   = sum(1 for e in entries_data if e["flag"] == "🔴")
    n_missing = sum(1 for e in entries_data if e["missing"])

    html = HTML_HEAD.format()
    html += f"""
<header>
  <div>
    <h1>🌽 Yellow MAIze — Pseudo-Label Validator</h1>
    <div class="meta">Mode: <b>{mode}</b> &nbsp;|&nbsp; Generated: {ts} &nbsp;|&nbsp; Images: {n_total}</div>
  </div>
  <div class="meta">Post factory_master.py QA</div>
</header>
<div class="container">
"""

    # ── Summary ──
    pct_good = round(100 * n_good / max(n_total, 1))
    html += f"""
<div class="summary">
  <h2>📊 Summary</h2>
  <div class="summary-grid">
    <div class="stat-card"><div class="label">Total Images</div><div class="value">{n_total}</div></div>
    <div class="stat-card"><div class="label">Pass ✅</div><div class="value good">{n_good}</div></div>
    <div class="stat-card"><div class="label">Review ⚠️</div><div class="value warn">{n_warn}</div></div>
    <div class="stat-card"><div class="label">Flag 🔴</div><div class="value bad">{n_bad}</div></div>
    <div class="stat-card"><div class="label">Missing Masks</div><div class="value {"bad" if n_missing>0 else "good"}">{n_missing}</div></div>
    <div class="stat-card"><div class="label">Pass Rate</div><div class="value {"good" if pct_good>60 else "warn" if pct_good>30 else "bad"}">{pct_good}%</div></div>
  </div>
</div>
"""

    # ── Active Config ──
    html += """<div class="config-panel"><h2>⚙️ Active factory_master.py Thresholds (mirrored)</h2><div class="config-grid">"""
    for k, v in DISPLAY_CONSTANTS.items():
        html += f'<div class="config-row"><span class="config-key">{k}</span><span class="config-val">= {v}</span></div>'
    html += "</div></div>"

    # ── Reference Guide ──
    html += """
<div class="ref-panel">
  <h2>📖 What to Look For (Based on Published Symptom Literature)</h2>
  <div class="ref-grid">
    <div class="ref-card healthy">
      <div class="cls-label">HEALTHY — Yellow Maize</div>
      <ul>
        <li>Mask should be <b>near-empty</b> (black / &lt;1% fill)</li>
        <li>Uniform rich green or deep yellow leaf colour with bright midrib</li>
        <li>No pale streaks, no chlorotic patches, no margin browning</li>
        <li>3-band guard: &lt;1% → zero | 1–4% → ×0.3 | ≥4% → genuine early signal</li>
        <li>⚠️ If mask &gt;4%: raise HEALTHY_A_MIN (140→145) or HEALTHY_B_MIN (145→150)</li>
      </ul>
    </div>
    <div class="ref-card msv">
      <div class="cls-label">MSV — Maize Streak Virus</div>
      <ul>
        <li>Mask should show <b>narrow, broken, vein-parallel streaks</b></li>
        <li>Streak colours: pale green → yellow → white (not solid blotches)</li>
        <li>On yellow maize: paler/whiter interruptions of the yellow background</li>
        <li>Gabor (4 orientations) reinforces streak structure — should NOT mask whole leaf</li>
        <li>Early: scattered round spots (~0.5–2mm) that elongate into streaks (Philippines 2024)</li>
        <li>⚠️ Mask empty: lower LAB_MSV_B_MIN or GABOR_THRESHOLD | Mask full: raise LAB_MSV_A_MIN</li>
      </ul>
    </div>
    <div class="ref-card mln">
      <div class="cls-label">MLN — Maize Lethal Necrosis</div>
      <ul>
        <li>Mask should show <b>broad yellowing + margin-to-midrib necrosis</b></li>
        <li>Unlike MSV: streaks are <b>wider, less vein-bound</b></li>
        <li>Leaves dry from outer <b>margins inward</b> toward midrib</li>
        <li>Dark necrotic patches (brown/black) captured by dark_necrosis branch (L*&lt;110, a*≥125)</li>
        <li>Advanced: whole leaf dries; mottled mosaic pattern visible</li>
        <li>⚠️ Only margins masked: lower dark_necrosis L*&lt;110→120 | Mask full: raise LAB_MLN_A_MIN</li>
      </ul>
    </div>
  </div>
</div>
"""

    # ── Per-class image cards ──
    for cls in (CLASSES + ["UNKNOWN"]):
        cls_entries = [e for e in entries_data if e["category"] == cls]
        if not cls_entries:
            continue

        cls_lower = cls.lower()
        cls_badge = f'<span class="cls-badge {cls_lower}">{cls}</span>'
        n_cls = len(cls_entries)
        n_cls_bad = sum(1 for e in cls_entries if e["flag"] == "🔴")
        n_cls_warn = sum(1 for e in cls_entries if e["flag"] == "⚠️")

        html += f"""
<div class="section-title">{cls_badge} {cls} — {n_cls} images
  &nbsp;<span style="font-size:13px;font-weight:400;color:var(--subtext)">
    ✅ {n_cls - n_cls_bad - n_cls_warn} &nbsp; ⚠️ {n_cls_warn} &nbsp; 🔴 {n_cls_bad}
  </span>
</div>
<div class="image-grid">
"""
        for e in cls_entries:
            flag = e["flag"]
            stem = e["stem"]
            diag = e["diag"]
            stats = e.get("stats", {})
            severity = e.get("severity", -1)
            weight = e.get("weight", -1)
            grade = e.get("grade", -1)
            missing = e.get("missing", False)

            sym_pct = stats.get("sym_pct", 0)
            leaf_pct = round(100 * stats.get("leaf_px", 0) / max(stats.get("leaf_px", 1) + 100, 1), 1)

            raw_b64   = e.get("raw_b64", "")
            sym_b64   = e.get("sym_b64", "")
            ovly_b64  = e.get("ovly_b64", "")

            raw_img_tag  = f'<img src="data:image/jpeg;base64,{raw_b64}" alt="raw">'  if raw_b64  else '<div style="background:#111;aspect-ratio:1;display:flex;align-items:center;justify-content:center;color:#555;">No raw image</div>'
            sym_img_tag  = f'<img src="data:image/jpeg;base64,{sym_b64}" alt="mask">' if sym_b64  else '<div style="background:#111;aspect-ratio:1;display:flex;align-items:center;justify-content:center;color:#555;">No mask</div>'
            ovly_img_tag = f'<img src="data:image/jpeg;base64,{ovly_b64}" alt="overlay">' if ovly_b64 else '<div style="background:#111;aspect-ratio:1;display:flex;align-items:center;justify-content:center;color:#555;">No overlay</div>'

            missing_note = '<span style="color:var(--red);font-size:11px;">⚠️ PSEUDO-MASK NOT FOUND — image may have been filtered by bouncer or excluded</span>' if missing else ""

            html += f"""
  <div class="img-card">
    <div class="card-header">
      <div class="stem">{stem}</div>
      <div class="flag">{flag}</div>
    </div>
    <div class="panel-row">
      <div class="panel">{raw_img_tag}<div class="panel-label">Raw</div></div>
      <div class="panel">{sym_img_tag}<div class="panel-label">Symptom Mask</div></div>
      <div class="panel">{ovly_img_tag}<div class="panel-label">Overlay</div></div>
    </div>
    <div class="card-diag">
      <div class="diag-row">
        <div class="kv"><span class="k">Symptom:</span><span class="v">{sym_pct:.1f}%</span></div>
        <div class="kv"><span class="k">Severity:</span><span class="v">{severity:.2f}%</span></div>
        <div class="kv"><span class="k">Weight:</span><span class="v">{weight:.2f}</span></div>
        <div class="kv"><span class="k">Mode:</span><span class="v">{mode}</span></div>
      </div>
      {missing_note}
      <div class="diag-msg">{diag}</div>
    </div>
  </div>
"""
        html += "</div>"  # image-grid

    # ── Adjustment Guide Table ──
    html += """
<div class="adjust-panel">
  <h2>🔧 Adjustment Guide — What to Change When Something Looks Wrong</h2>
  <table>
    <thead>
      <tr><th>Problem Observed</th><th>Parameter to Change</th><th>File</th><th>Direction</th></tr>
    </thead>
    <tbody>
      <tr><td>Too many FP on HEALTHY yellow leaves (MSV path)</td><td>LAB_MSV_A_MIN</td><td>factory_master.py</td><td>Raise 133 → 135</td></tr>
      <tr><td>Too many FP on HEALTHY yellow leaves (MSV path)</td><td>LAB_MSV_B_MIN</td><td>factory_master.py</td><td>Raise 135 → 138</td></tr>
      <tr><td>MSV streaks entirely missed on visible streaky leaf</td><td>LAB_MSV_B_MIN</td><td>factory_master.py</td><td>Lower 135 → 130</td></tr>
      <tr><td>MSV streaks entirely missed on visible streaky leaf</td><td>GABOR_THRESHOLD</td><td>config.py</td><td>Lower 0.30 → 0.20</td></tr>
      <tr><td>Short MSV streaks destroyed by directional opening</td><td>MSV kernel in compute_lab_hard_mask</td><td>factory_master.py</td><td>(1,7) → (1,5)</td></tr>
      <tr><td>Midrib bright stripe captured as MSV symptom</td><td>LAB_GREEN_A_MAX</td><td>factory_master.py</td><td>Lower 121 → 118</td></tr>
      <tr><td>MLN dark necrotic patches not captured</td><td>dark_necrosis L* ceiling</td><td>factory_master.py</td><td>Raise 110 → 120</td></tr>
      <tr><td>MLN dark necrotic patches not captured</td><td>dark_necrosis a* floor</td><td>factory_master.py</td><td>Lower 125 → 120</td></tr>
      <tr><td>MLN fragmented necrotic patches still dropping out</td><td>MLN elliptical closing kernel</td><td>factory_master.py</td><td>(5,5) → (7,7)</td></tr>
      <tr><td>HEALTHY mask &gt;4% (warm-yellow FP)</td><td>HEALTHY a threshold</td><td>factory_master.py</td><td>Raise 140 → 145</td></tr>
      <tr><td>HEALTHY mask &gt;4% (warm-yellow FP)</td><td>HEALTHY b threshold</td><td>factory_master.py</td><td>Raise 145 → 150</td></tr>
      <tr><td>Silhouette cuts into leaf edges</td><td>FACTORY_SILHOUETTE_THRESHOLD</td><td>config.py</td><td>Lower 0.35 → 0.25</td></tr>
      <tr><td>Many images weight=-1 (low coverage filtered)</td><td>FACTORY_MIN_LEAF_COVERAGE</td><td>config.py</td><td>Lower 0.15 → 0.10</td></tr>
      <tr><td>Gabor kills streaks even with strong LAB hits</td><td>Weighted product threshold in compute_lab_hard_mask</td><td>factory_master.py</td><td>Lower 0.4 → 0.3</td></tr>
      <tr><td>L* normalisation over-amplifying yellow images</td><td>_normalize_L skip range</td><td>factory_master.py</td><td>Lower 180 → 150</td></tr>
    </tbody>
  </table>
</div>
"""

    html += f"""
<footer>
  Yellow MAIze Project · Phase 4 Pseudo-Label Validator · Mode: {mode} · {ts}<br>
  Run <code>python validate_factory.py --help</code> for options.
</footer>
</div></body></html>"""

    return html


# ═══════════════════════════════════════════════════════════════════════════
# PER-IMAGE PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def _process_entry(entry: dict, mode: str) -> dict:
    """Load raw image + mask, compute stats, build base64 thumbnails."""
    stem     = entry["stem"]
    src_path = entry["source_path"]
    category = entry["category"]

    result = {
        "stem": stem, "category": category,
        "flag": "❓", "diag": "", "missing": False,
        "stats": {}, "severity": -1.0, "weight": -1.0, "grade": -1,
        "raw_b64": "", "sym_b64": "", "ovly_b64": "",
    }

    # Load raw image
    raw_rgb = None
    if src_path and Path(src_path).exists():
        raw_bgr = cv2.imread(str(src_path))
        if raw_bgr is not None:
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

    # Load pseudo masks
    sil, sym, severity, weight, grade = _load_pseudo_mask(stem, mode)
    result["severity"] = severity
    result["weight"]   = weight
    result["grade"]    = grade

    if sym is None:
        result["missing"] = True
        result["flag"]  = "❓"
        result["diag"]  = "Pseudo-mask not found. Image may have been filtered by the Bouncer or is in the test split."
        if raw_rgb is not None:
            th = 300
            h, w = raw_rgb.shape[:2]
            scale = th / max(h, w)
            thumb = cv2.resize(raw_rgb, (int(w*scale), int(h*scale)))
            result["raw_b64"] = _img_to_b64(thumb)
        return result

    # Resize mask to match raw image if needed
    if raw_rgb is not None:
        h, w = raw_rgb.shape[:2]
        if sym is not None and (sym.shape[0] != h or sym.shape[1] != w):
            sym = cv2.resize(sym, (w, h), interpolation=cv2.INTER_NEAREST)
        if sil is not None and (sil.shape[0] != h or sil.shape[1] != w):
            sil = cv2.resize(sil, (w, h), interpolation=cv2.INTER_LINEAR)

    stats = _symptom_stats(sym, sil)
    result["stats"] = stats
    flag, diag = _auto_flag(category, stats, severity)
    result["flag"] = flag
    result["diag"] = diag

    # Build thumbnails (300px long side)
    th = 300
    if raw_rgb is not None:
        h, w = raw_rgb.shape[:2]
        scale = th / max(h, w)
        raw_thumb = cv2.resize(raw_rgb, (int(w*scale), int(h*scale)))
        result["raw_b64"] = _img_to_b64(raw_thumb)

        if sym is not None:
            sym_th   = cv2.resize(sym, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)
            sil_th   = cv2.resize(sil, (int(w*scale), int(h*scale))) if sil is not None else None
            result["sym_b64"]  = _img_to_b64(sym_th)
            ovly = _overlay_symptom_on_raw(raw_thumb, sym_th, sil_th)
            result["ovly_b64"] = _img_to_b64(ovly)
    else:
        # No raw image — show mask only
        if sym is not None:
            h, w = sym.shape[:2]
            scale = th / max(h, w)
            sym_th = cv2.resize(sym, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)
            result["sym_b64"] = _img_to_b64(sym_th)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def _discover_borderline_images(mode: str, margin: float = 5.0) -> list[dict]:
    """
    Sample images near CIMMYT severity grade boundaries for targeted human review.
    Boundary thresholds: 5%, 25%, 50%, 75% (MSV) and 10%, 25%, 50%, 75% (MLN).
    Returns images whose severity % falls within ±margin of any boundary.
    """
    import pandas as pd
    boundary_thresholds = [5.0, 10.0, 25.0, 50.0, 75.0]
    entries = []

    if not GLOBAL_MANIFEST.exists():
        print("[WARN] global_split_manifest.csv not found for borderline sampling.")
        return []

    df = pd.read_csv(GLOBAL_MANIFEST)
    df = df[df["split"] != "test"].reset_index(drop=True)
    mode_dir = PSEUDO_DIR / mode

    for _, row in df.iterrows():
        stem = Path(row.get("filename", row.get("source_path", ""))).stem
        sev_path = mode_dir / f"{stem}_sev.txt"
        if not sev_path.exists():
            continue
        try:
            sev = float(sev_path.read_text().strip())
        except ValueError:
            continue
        if sev < 0:
            continue
        # Check if within margin of any boundary
        if any(abs(sev - t) <= margin for t in boundary_thresholds):
            entries.append({
                "stem":        stem,
                "source_path": row.get("source_path", ""),
                "category":    row.get("category", "UNKNOWN"),
            })

    # Limit to max 30 per class to keep report manageable
    from collections import defaultdict
    by_class = defaultdict(list)
    for e in entries:
        by_class[e["category"]].append(e)
    result = []
    for cls_entries in by_class.values():
        result.extend(cls_entries[:30])
    print(f"  [Borderline] Found {len(result)} images within ±{margin}% of grade boundaries.")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate factory_master.py pseudo-label outputs visually.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n",          type=int,  default=10,       help="Images per class to sample (default: 10)")
    parser.add_argument("--mode",       type=str,  default="mode_b", help="Factory mode to validate (default: mode_b)")
    parser.add_argument("--all-modes",  action="store_true",          help="Validate all 4 modes (mode_a/b/c/d)")
    parser.add_argument("--img",        type=str,  default=None,      help="Path to a single raw image to validate")
    parser.add_argument("--out",        type=str,  default=None,      help="Output HTML path (default: reports/validate_factory_<mode>.html)")
    parser.add_argument("--borderline", action="store_true",
                        help="Sample images near CIMMYT grade boundaries (5%%, 25%%, 50%%, 75%%) for human review")
    args = parser.parse_args()

    modes_to_run = FACTORY_MODES if args.all_modes else [args.mode]

    for mode in modes_to_run:
        mode = mode.strip()
        print(f"\n{'='*60}")
        print(f"  Validating mode: {mode}")
        print(f"{'='*60}")

        if mode not in [m.strip() for m in FACTORY_MODES]:
            print(f"[WARN] '{mode}' not in FACTORY_MODES config: {FACTORY_MODES}")

        if args.borderline:
            entries = _discover_borderline_images(mode)
            if not entries:
                print(f"  [WARN] No borderline images found for mode {mode}. Run factory_master.py first.")
                continue
        else:
            entries = _discover_images(args.n, mode, args.img)
        if not entries:
            print(f"[ERROR] No images found for mode '{mode}'. Skipping.")
            continue

        print(f"  Found {len(entries)} images to validate...")
        entries_data = []
        for i, entry in enumerate(entries):
            processed = _process_entry(entry, mode)
            entries_data.append(processed)
            flag = processed["flag"]
            pct  = processed["stats"].get("sym_pct", 0)
            print(f"  [{i+1:>3}/{len(entries)}] {flag} {processed['category']:8s} | {processed['stem'][:50]:<50} | sym={pct:.1f}%")

        # Summary
        n_good = sum(1 for e in entries_data if e["flag"] == "✅")
        n_warn = sum(1 for e in entries_data if e["flag"] == "⚠️")
        n_bad  = sum(1 for e in entries_data if e["flag"] == "🔴")
        print(f"\n  RESULTS: ✅ {n_good}  ⚠️ {n_warn}  🔴 {n_bad}  / {len(entries_data)} total")

        # Build report
        html = _build_html_report(entries_data, mode, modes_to_run)

        if args.out:
            out_path = Path(args.out)
        else:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = REPORTS_DIR / f"validate_factory_{mode}.html"

        out_path.write_text(html, encoding="utf-8")
        print(f"\n  ✅ Report saved → {out_path}")
        print(f"     Open in a browser: file://{out_path.resolve()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
