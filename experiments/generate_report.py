#!/usr/bin/env python3
import argparse, csv, os, math, json
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# -------------------------
# IO helpers
# -------------------------
def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# -------------------------
# Data helpers
# -------------------------
def to_float(x):
    try:
        return float(x)
    except:
        return float("nan")

def collect_groups(per_run_rows, metric_keys):
    groups = sorted({r["group"] for r in per_run_rows})
    seeds_by_group = {g: sorted({int(r["seed"]) for r in per_run_rows if r["group"]==g}) for g in groups}
    data = {g: {mk: [] for mk in metric_keys} for g in groups}
    for g in groups:
        for mk in metric_keys:
            vals = [to_float(r.get(mk,"")) for r in per_run_rows if r["group"]==g]
            vals = [v for v in vals if not math.isnan(v)]
            data[g][mk] = vals
    return groups, seeds_by_group, data

def pick_primary_candidates(all_metric_keys):
    # Prioritize common primary metrics if present
    prefs = ["bbox.AP", "bbox.AP50", "phenotype.fresh_weight.r2", "phenotype.fresh_weight.rmse"]
    for p in prefs:
        if p in all_metric_keys:
            return p
    return sorted(all_metric_keys)[0] if all_metric_keys else None

# -------------------------
# Plotting
# -------------------------
def plot_metric_box_violin(outdir, metric, groups, data):
    """Save boxplot + violin plot for metric."""
    values = [data[g][metric] for g in groups]
    # Skip empty metric
    if all(len(v)==0 for v in values):
        return []

    # Boxplot
    fig1, ax1 = plt.subplots(figsize=(7,4))
    ax1.boxplot(values, tick_labels=groups, showfliers=True)
    ax1.set_title(f"{metric} — boxplot")
    ax1.set_ylabel(metric)
    fig1.tight_layout()
    p1 = Path(outdir)/f"{metric.replace('.','_')}_box.png"
    fig1.savefig(p1, dpi=160)
    plt.close(fig1)

    # Violin
    fig2, ax2 = plt.subplots(figsize=(7,4))
    ax2.violinplot(values, showmeans=True, showmedians=True)
    ax2.set_xticks(range(1,len(groups)+1))
    ax2.set_xticklabels(groups)
    ax2.set_title(f"{metric} — violin")
    ax2.set_ylabel(metric)
    fig2.tight_layout()
    p2 = Path(outdir)/f"{metric.replace('.','_')}_violin.png"
    fig2.savefig(p2, dpi=160)
    plt.close(fig2)

    return [str(p1), str(p2)]

# -------------------------
# Formatting helpers
# -------------------------
def f4(x):
    try:
        return f"{float(x):.4g}"
    except:
        return "nan"

def rank_direction(metric_name):
    # larger-is-better for AP/r2; lower-is-better for RMSE/MAPE/latency
    name = metric_name.lower()
    if any(k in name for k in ["rmse","mape","latency","loss","mbe","nrmse"]):
        return "lower"
    return "higher"

def bold_if_best(group, metric, means_by_group):
    dir_ = rank_direction(metric)
    vals = {g: means_by_group[g][metric] for g in means_by_group}
    # filter NaNs
    vals = {g: v for g,v in vals.items() if not math.isnan(v)}
    if not vals: return False
    if dir_ == "higher":
        best_val = max(vals.values())
    else:
        best_val = min(vals.values())
    return abs(means_by_group[group][metric] - best_val) < 1e-12

# -------------------------
# Report generation
# -------------------------
def build_report(outdir, per_run_rows, group_summary_rows, pair_rows, figures_by_metric, primary_metric):
    groups = sorted({r["group"] for r in per_run_rows})

    # Compute mean±CI dict for quick lookup
    summary = {(r["group"], r["metric"]): r for r in group_summary_rows}

    # Mean table (primary metric)
    means_by_group = {g: {} for g in groups}
    for r in group_summary_rows:
        g, mk = r["group"], r["metric"]
        means_by_group[g][mk] = float(r["mean"]) if r["mean"] else float("nan")

    # Build significant winners map using Holm corrected p on t-test
    sig_pairs = defaultdict(list)  # metric -> list of (g1,g2,sign,adj_p)
    for r in pair_rows:
        mk = r["metric"]
        g1, g2 = r["group1"], r["group2"]
        p = float(r.get("t_p_holm", "nan")) if r.get("t_p_holm","")!="" else float("nan")
        if math.isnan(p):
            continue
        if p < 0.05:
            # Decide direction from mean difference
            m1 = means_by_group.get(g1, {}).get(mk, float("nan"))
            m2 = means_by_group.get(g2, {}).get(mk, float("nan"))
            if math.isnan(m1) or math.isnan(m2): 
                continue
            dir_ = rank_direction(mk)
            better = None
            if dir_ == "higher":
                if m1 > m2: better = g1
                elif m2 > m1: better = g2
            else:
                if m1 < m2: better = g1
                elif m2 < m1: better = g2
            if better is not None:
                other = g2 if better==g1 else g1
                sig_pairs[mk].append((better, other, p))

    # Markdown
    md = []
    md.append(f"# Experiment Report")
    md.append(f"- Source: `{outdir}`")
    md.append(f"- Primary metric: **{primary_metric}**")
    md.append("")
    md.append("## Quick Summary (primary metric)")
    md.append("| Group | n | mean | 95% CI | note |")
    md.append("|---|---:|---:|---:|---|")
    for g in groups:
        r = summary.get((g, primary_metric))
        if not r:
            continue
        n = int(r["n"])
        m = float(r["mean"]); lo = float(r["ci95_low"]); hi = float(r["ci95_high"])
        note = "**best**" if bold_if_best(g, primary_metric, means_by_group) else ""
        md.append(f"| {g} | {n} | {m:.4f} | [{lo:.4f}, {hi:.4f}] | {note} |")

    # Significant pairwise results for primary
    md.append("")
    md.append("### Significant Pairwise Differences (Holm-adjusted, primary metric)")
    primary_sig = [p for p in sig_pairs.get(primary_metric, [])]
    if not primary_sig:
        md.append("_No significant differences after correction._")
    else:
        md.append("| Better | Worse | Holm p |")
        md.append("|---|---|---:|")
        for b,w,p in sorted(primary_sig, key=lambda x: x[2]):
            md.append(f"| **{b}** | {w} | {p:.4g} |")

    # Figures
    md.append("")
    md.append("## Plots")
    if primary_metric in figures_by_metric:
        md.append(f"### {primary_metric}")
        for fig in figures_by_metric[primary_metric]:
            md.append(f"![{primary_metric}]({Path(fig).name})")
        md.append("")

    # Attach figures for a few more common metrics if available
    attach = []
    for cand in ("bbox.AP50","bbox.AP75","phenotype.fresh_weight.r2","phenotype.fresh_weight.rmse"):
        if cand in figures_by_metric and cand != primary_metric:
            attach.append(cand)
    for mk in attach:
        md.append(f"### {mk}")
        for fig in figures_by_metric[mk]:
            md.append(f"![{mk}]({Path(fig).name})")
        md.append("")

    # Interpretation cheat-sheet
    md.append("")
    md.append("## How to Read These Stats")
    md.append("- **mean ± 95% CI**: central performance and uncertainty across seeds.")
    md.append("- **bolded mean**: best average among groups for that metric.")
    md.append("- **Holm p < 0.05**: robust significant difference after multiple-comparison correction.")
    md.append("- **Higher-is-better** metrics: AP, AP50, AP75, r². **Lower-is-better**: RMSE, MAPE, latency, loss.")
    md.append("- If Wilcoxon/Mann–Whitney p is reported (in CSV), it’s a nonparametric confirmation of the t-test.")
    md.append("- If some p-values are `nan`, sample size or identical scores made the test undefined—treat as ‘no evidence of difference’.")
    md.append("")
    return "\n".join(md)

# -------------------------
# LaTeX table (optional)
# -------------------------
def write_latex_table(path, primary_metric, groups, summary):
    # \pm table for quick paste into thesis
    lines = []
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Group & $n$ & " + primary_metric.replace("_", r"\_") + r" (mean $\pm$ 95\% CI) \\")
    lines.append(r"\midrule")
    for g in groups:
        r = summary.get((g, primary_metric))
        if not r: 
            continue
        n = int(r["n"]); m = float(r["mean"]); lo = float(r["ci95_low"]); hi = float(r["ci95_high"])
        half = (hi - lo)/2.0
        lines.append(f"{g} & {n} & {m:.4f} $\\pm$ {half:.4f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    write_text(path, "\n".join(lines))

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="ind", required=True, help="Directory containing per_run_metrics.csv, group_summary.csv, pairwise_tests.csv")
    ap.add_argument("--out", dest="outd", required=True, help="Output directory for report and figures")
    ap.add_argument("--primary", default=None, help="Primary metric to highlight (default: auto-pick)")
    args = ap.parse_args()

    ind = Path(args.ind); outd = Path(args.outd)
    outd.mkdir(parents=True, exist_ok=True)

    per_run = read_csv(ind/"per_run_metrics.csv")
    group_summary = read_csv(ind/"group_summary.csv")
    pairwise = read_csv(ind/"pairwise_tests.csv")

    # Discover metric keys from per_run
    all_mk = sorted(k for k in per_run[0].keys() if k.startswith("bbox.") or k.startswith("phenotype."))
    primary_metric = args.primary or pick_primary_candidates(all_mk)

    # Build group data
    groups = sorted({r["group"] for r in per_run})
    summary_lookup = {(r["group"], r["metric"]): r for r in group_summary}

    # Collect values and draw plots
    _, _, data = collect_groups(per_run, all_mk)
    figures_by_metric = {}
    for mk in all_mk:
        figs = plot_metric_box_violin(outd, mk, groups, data)
        if figs:
            figures_by_metric[mk] = figs

    # Build markdown
    md = build_report(outd, per_run, group_summary, pairwise, figures_by_metric, primary_metric)
    write_text(outd/"REPORT.md", md)

    # Optional LaTeX table for primary metric
    write_latex_table(outd/"primary_metric_table.tex", primary_metric, groups, summary_lookup)

    print(f"[OK] Report written to {outd/'REPORT.md'}")
    print(f"[OK] Figures saved in {outd}")
    print(f"[OK] LaTeX table saved to {outd/'primary_metric_table.tex'}")

if __name__ == "__main__":
    main()
