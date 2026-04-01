#!/usr/bin/env python3
"""
scripts/06_analysis.py
=======================
Statistical analysis and publication-ready outputs.

Reads all model results from results/ directory and produces:
  - Full comparison table (LaTeX + CSV)
  - LPG / CLCS bar charts
  - KD ablation tables (T×α grid)
  - Statistical significance tests across models
  - thesis-ready figures (matplotlib)

CLUSTER SAFETY:
  - Only reads from results/ — no model loading
  - Output bounded in size
  - Matplotlib in non-interactive (Agg) backend — no display required

Usage:
    python scripts/06_analysis.py \
        --results-dir results/ \
        --output-dir analysis/
"""

import os
import sys
import csv
import json
import shutil
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def check_disk_space(path: Path, required_gb: float = 0.5, label: str = "") -> None:
    stat = shutil.disk_usage(path if path.exists() else path.parent)
    free_gb = stat.free / (1024 ** 3)
    if free_gb < required_gb:
        raise RuntimeError(f"DISK ERROR [{label}]: need {required_gb} GB, have {free_gb:.1f} GB.")


def is_path_approved(path: Path) -> bool:
    for var in ["HOME", "SCRATCH"]:
        root = os.environ.get(var)
        if root:
            try:
                path.resolve().relative_to(Path(root).resolve())
                return True
            except ValueError:
                continue
    return False


def safe_makedirs(path: Path) -> None:
    if not is_path_approved(path):
        raise PermissionError(f"SAFETY BLOCK: {path}")
    path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_results(results_dir: Path) -> list[dict]:
    """Scan results/ subfolders and collect all summary.json files."""
    records = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        summary_path = model_dir / "summary.json"
        if not summary_path.exists():
            log.warning(f"  No summary.json in {model_dir.name} — skipping.")
            continue
        with open(summary_path) as f:
            data = json.load(f)
        data["_model_dir"] = str(model_dir)
        records.append(data)
        log.info(f"  Loaded: {model_dir.name}")
    return records


# =============================================================================
# COMPARISON TABLE
# =============================================================================

METRIC_GROUPS = {
    "Pillar 1 (EN)": ["MMLU", "GSM8K", "TruthfulQA"],
    "Pillar 2 (AR)": ["ArabicMMLU", "ArabicQA_F1", "ArabicQA_EM"],
    "Pillar 3 (Cross-lingual)": ["Acc_EN", "Acc_AR", "LPG", "CLCS", "IFS", "RDI"],
    "Pillar 4 (Stress)": ["AR2EN_Adherence", "EN2AR_Adherence",
                           "CodeSwitch_Coherence", "Dialectal_Avg"],
    "Statistics": ["McNemar_p", "Cohen_d"],
}


def format_val(v, highlight_best=False):
    if v is None or v == "":
        return "--"
    try:
        f = float(v)
        return f"{f:.4f}"
    except (ValueError, TypeError):
        return str(v)


def build_latex_table(records: list[dict], output_path: Path) -> None:
    """Full comparison table in LaTeX booktabs format."""
    all_metrics = [m for group in METRIC_GROUPS.values() for m in group]
    model_names = [r["model"] for r in records]

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Full 4-Pillar Evaluation: Arabic KD Pipeline Comparison}",
        r"\label{tab:full_comparison}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(model_names) + "}",
        r"\toprule",
        "Metric & " + " & ".join(m.replace("_", "\\_") for m in model_names) + r" \\",
        r"\midrule",
    ]

    current_group = None
    for group_name, metrics in METRIC_GROUPS.items():
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{" + str(len(model_names)+1) + r"}{l}{\textit{" +
                     group_name + r"}} \\")
        for metric in metrics:
            vals = [format_val(r.get(metric)) for r in records]
            row = metric.replace("_", "\\_") + " & " + " & ".join(vals) + r" \\"
            lines.append(row)

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]

    check_disk_space(output_path.parent, 0.1, "LaTeX table")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    log.info(f"  📄 LaTeX table: {output_path}")


def build_csv_table(records: list[dict], output_path: Path) -> None:
    """Full comparison CSV."""
    all_metrics = [m for group in METRIC_GROUPS.values() for m in group]
    check_disk_space(output_path.parent, 0.1, "CSV table")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric"] + [r["model"] for r in records])
        for metric in all_metrics:
            writer.writerow([metric] + [format_val(r.get(metric)) for r in records])
    log.info(f"  📄 CSV: {output_path}")


# =============================================================================
# KD ABLATION TABLE (T × α grid)
# =============================================================================

def build_kd_ablation_table(records: list[dict], output_path: Path) -> None:
    """
    Extract token-KD runs and build T×α ablation table for a given metric (CLCS).
    Expects model names like: qwen7b_token_kd_T2_a05
    """
    import re

    kd_records = {}
    for r in records:
        name = r.get("model", "")
        m = re.search(r"token_kd_T(\d+)_a(\d+)", name)
        if m:
            T = int(m.group(1))
            alpha = int(m.group(2)) / 10
            kd_records[(T, alpha)] = r

    if not kd_records:
        log.info("  No token-KD records found — skipping ablation table.")
        return

    temperatures = sorted(set(k[0] for k in kd_records))
    alphas = sorted(set(k[1] for k in kd_records))
    metric = "CLCS"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Token-KD Ablation: CLCS by Temperature $T$ and $\alpha$}",
        r"\label{tab:kd_ablation}",
        r"\begin{tabular}{l" + "c" * len(alphas) + "}",
        r"\toprule",
        r"$T$ \textbackslash $\alpha$ & " +
        " & ".join(str(a) for a in alphas) + r" \\",
        r"\midrule",
    ]

    for T in temperatures:
        row_vals = [format_val(kd_records.get((T, a), {}).get(metric)) for a in alphas]
        lines.append(f"$T={T}$ & " + " & ".join(row_vals) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    check_disk_space(output_path.parent, 0.1)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    log.info(f"  📄 KD ablation table: {output_path}")


# =============================================================================
# FIGURES
# =============================================================================

def build_figures(records: list[dict], output_dir: Path) -> None:
    """Generate bar charts for LPG and CLCS across models."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive — safe for cluster
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        log.warning("matplotlib not available — skipping figures. "
                    "pip install matplotlib --break-system-packages")
        return

    check_disk_space(output_dir, 0.1, "figures")

    model_names = [r["model"] for r in records]
    x = np.arange(len(model_names))

    # ── LPG chart ─────────────────────────────────────────────────────────────
    lpg_vals = [float(r.get("LPG") or 0) for r in records]
    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 1.2), 4))
    bars = ax.bar(x, lpg_vals, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("LPG = |Acc_EN - Acc_AR|")
    ax.set_title("Language Performance Gap (lower = better)")
    for bar, val in zip(bars, lpg_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    lpg_path = output_dir / "lpg_comparison.pdf"
    plt.savefig(lpg_path, bbox_inches="tight")
    plt.close()
    log.info(f"  📊 LPG chart: {lpg_path}")

    # ── CLCS chart ────────────────────────────────────────────────────────────
    clcs_vals = [float(r.get("CLCS") or 0) for r in records]
    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 1.2), 4))
    bars = ax.bar(x, clcs_vals, color="seagreen", edgecolor="black", linewidth=0.5)
    ax.axhline(0.70, color="red", linestyle="--", linewidth=1.0, label="Target CLCS=0.70")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("CLCS")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-Lingual Consistency Score (higher = better)")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, clcs_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    clcs_path = output_dir / "clcs_comparison.pdf"
    plt.savefig(clcs_path, bbox_inches="tight")
    plt.close()
    log.info(f"  📊 CLCS chart: {clcs_path}")

    # ── Multi-metric radar (spider) chart ──────────────────────────────────
    try:
        radar_metrics = ["MMLU", "ArabicMMLU", "CLCS", "AR2EN_Adherence", "EN2AR_Adherence"]
        n_metrics = len(radar_metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(records)))

        for rec, color in zip(records, colors):
            values = [float(rec.get(m) or 0) for m in radar_metrics]
            values += values[:1]
            ax.plot(angles, values, linewidth=1.5, linestyle="solid",
                    label=rec["model"], color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_metrics, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title("Multi-Metric Model Comparison", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=6)
        plt.tight_layout()
        radar_path = output_dir / "radar_comparison.pdf"
        plt.savefig(radar_path, bbox_inches="tight")
        plt.close()
        log.info(f"  📊 Radar chart: {radar_path}")
    except Exception as e:
        log.warning(f"Radar chart failed: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--output-dir", default="analysis/")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not is_path_approved(output_dir):
        output_dir = Path(os.environ.get("HOME", ".")) / "projects/qwen-arabic-kd/analysis"
        log.warning(f"Output dir outside approved paths — redirecting to {output_dir}")

    safe_makedirs(output_dir)
    check_disk_space(output_dir, 0.5, "analysis output")

    if not results_dir.exists():
        log.error(f"Results directory not found: {results_dir}")
        log.error("Run 05_eval_4pillars.py first for at least one model.")
        sys.exit(1)

    log.info(f"\n{'='*60}")
    log.info(f"06_analysis.py")
    log.info(f"Results: {results_dir}")
    log.info(f"Output:  {output_dir}")
    log.info(f"{'='*60}\n")

    log.info("📥 Collecting results...")
    records = collect_results(results_dir)

    if not records:
        log.error("No results found. Run 05_eval_4pillars.py for at least one model first.")
        sys.exit(1)

    log.info(f"  Found {len(records)} model result(s): {[r['model'] for r in records]}")

    log.info("\n📊 Generating comparison tables...")
    build_latex_table(records, output_dir / "full_comparison.tex")
    build_csv_table(records, output_dir / "full_comparison.csv")

    log.info("\n🔬 KD ablation table...")
    build_kd_ablation_table(records, output_dir / "kd_ablation.tex")

    log.info("\n📈 Generating figures...")
    build_figures(records, output_dir)

    log.info(f"\n✅ Analysis complete → {output_dir}")
    log.info("  Files: full_comparison.tex, full_comparison.csv, "
             "kd_ablation.tex, lpg_comparison.pdf, clcs_comparison.pdf, radar_comparison.pdf")


if __name__ == "__main__":
    main()
