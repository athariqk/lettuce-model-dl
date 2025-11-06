import argparse, csv, glob, json, math, os, sys
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Any

import numpy as np

try:
    from scipy import stats as spstats
    SCIPY = True
except Exception:
    spstats = None
    SCIPY = False


# ---------------------------
# Helpers
# ---------------------------

def read_best_epoch_metrics(run_dir: str) -> Dict[str, Any]:
    """
    Parse epoch_log.csv in a run directory, find the row with best_by_val==1,
    and return a flat dict of metrics:
      - coco bbox stats: AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl
      - phenotype metrics if available: nested dicts flattened as phenotype.<name>.<metric>
      - val_loss, latency_ms, param_count if available from run_results.json
    """
    epoch_csv = os.path.join(run_dir, "epoch_log.csv")
    if not os.path.exists(epoch_csv):
        raise FileNotFoundError(f"Missing epoch_log.csv in {run_dir}")

    rows = []
    with open(epoch_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise RuntimeError(f"No rows in {epoch_csv}")

    # Prefer best_by_val==1; fallback to min val_loss if none marked (e.g. no-validate case)
    best_rows = [r for r in rows if str(r.get("best_by_val", "0")) == "1"]
    if not best_rows:
        # fallback: min val_loss row among numeric losses
        numeric_rows = [r for r in rows if r.get("val_loss","nan") not in ("", "nan")]
        if not numeric_rows:
            best = rows[-1]
        else:
            best = min(numeric_rows, key=lambda r: float(r["val_loss"]))
    else:
        # In practice there should be exactly one; if multiple, take the first
        best = best_rows[0]

    out = {}

    # Core
    out["val_loss"] = float(best["val_loss"]) if best.get("val_loss","nan") not in ("", "nan") else float("nan")

    # Evaluation metrics are a JSON dict with keys like "bbox": [12 stats], "phenotype": {...}
    eval_json = best.get("eval_metrics_json", "")
    if eval_json:
        try:
            eval_dict = json.loads(eval_json)
        except Exception:
            eval_dict = {}
    else:
        eval_dict = {}

    # COCO bbox stats (if present)
    bbox_names = [
        "AP", "AP50", "AP75", "APs", "APm", "APl",
        "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"
    ]
    if "bbox" in eval_dict and isinstance(eval_dict["bbox"], list):
        stats = eval_dict["bbox"]
        for i, name in enumerate(bbox_names):
            if i < len(stats) and stats[i] is not None:
                out[f"bbox.{name}"] = float(stats[i])

    # Phenotype metrics (nested dict: phenotype -> per-target -> r2/rmse/mape)
    if "phenotype" in eval_dict and isinstance(eval_dict["phenotype"], dict):
        for pheno_name, metrics in eval_dict["phenotype"].items():
            if isinstance(metrics, dict):
                for mk, mv in metrics.items():
                    # mk in {"r2","rmse","mape"}; values may be lists/np scalars
                    try:
                        val = float(np.array(mv).item() if np.ndim(mv)==0 else np.mean(mv))
                    except Exception:
                        try:
                            val = float(mv)
                        except Exception:
                            continue
                    out[f"phenotype.{pheno_name}.{mk}"] = val

    # Run summary extras
    run_json = os.path.join(run_dir, "run_results.json")
    if os.path.exists(run_json):
        try:
            R = json.load(open(run_json, "r"))
            if "latency_ms" in R:
                out["latency_ms"] = float(R["latency_ms"])
            if "param_count" in R:
                out["param_count"] = float(R["param_count"])
            if "best_val_loss" in R:
                out["best_val_loss_snapshot"] = float(R["best_val_loss"])
        except Exception:
            pass

    return out


def mean_std_ci(x: np.ndarray, alpha=0.05) -> Tuple[float,float,Tuple[float,float]]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x)==0:
        return float("nan"), float("nan"), (float("nan"), float("nan"))
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    if SCIPY and len(x) > 1:
        tcrit = spstats.t.ppf(1 - alpha/2, df=len(x)-1)
        hw = tcrit * s / math.sqrt(len(x))
        return m, s, (m - hw, m + hw)
    else:
        # normal approx
        hw = 1.96 * s / math.sqrt(max(1,len(x)))
        return m, s, (m - hw, m + hw)


def cohen_d(a: np.ndarray, b: np.ndarray, paired=False) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if paired:
        d = a - b
        return float(np.mean(d) / (np.std(d, ddof=1) + 1e-12)) if len(d)>1 else float("nan")
    # unpaired pooled
    n1, n2 = len(a), len(b)
    if n1<2 or n2<2: return float("nan")
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = math.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
    return float((np.mean(a)-np.mean(b)) / (sp + 1e-12))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a)==0 or len(b)==0: return float("nan")
    # efficient-ish: sort and count
    A = np.sort(a); B = np.sort(b)
    i=j=more=less=0
    nA, nB = len(A), len(B)
    while i<nA:
        while j<nB and B[j] < A[i]:
            j += 1
        less += j
        # count equals
        k = j
        while k<nB and B[k] == A[i]:
            k += 1
        equal = k - j
        more += (nB - k)
        i += 1
    delta = (more - less) / (nA * nB)
    return float(delta)


def holm_bonferroni(pvals: List[float]) -> List[float]:
    # returns adjusted p-values in input order
    m = len(pvals)
    idx = np.argsort(pvals)
    adj = [0]*m
    prev = 0.0
    for rank, i in enumerate(idx, start=1):
        pv = pvals[i] * (m - rank + 1)
        pv = min(1.0, pv)
        # enforce monotonicity
        pv = max(pv, prev)
        adj[i] = pv
        prev = pv
    return adj


def paired_tests(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, float); b = np.asarray(b, float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    out = {}
    if len(a) < 2:
        out["t_stat"] = np.nan; out["t_p"] = np.nan
        out["wilcoxon_stat"] = np.nan; out["wilcoxon_p"] = np.nan
        return out
    if SCIPY:
        t = spstats.ttest_rel(a, b, alternative="two-sided", nan_policy="omit")
        out["t_stat"], out["t_p"] = float(t.statistic), float(t.pvalue)
        try:
            w = spstats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
            out["wilcoxon_stat"], out["wilcoxon_p"] = float(w.statistic), float(w.pvalue)
        except Exception:
            out["wilcoxon_stat"], out["wilcoxon_p"] = np.nan, np.nan
    else:
        # bootstrap mean difference CI & p-value (two-sided)
        rng = np.random.default_rng(123)
        dif = a - b
        obs = float(np.mean(dif))
        B = 20000
        boot = [np.mean(rng.choice(dif, size=len(dif), replace=True)) for _ in range(B)]
        p = 2*min(np.mean(np.array(boot) >= obs), np.mean(np.array(boot) <= obs))
        out["t_stat"], out["t_p"] = np.nan, p
        out["wilcoxon_stat"], out["wilcoxon_p"] = np.nan, np.nan
    return out


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True,
                    help="Format: Label:glob_pattern   (e.g., \"MV3-RGB:runs/mv3_rgb_seed*\")")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--metric", default="bbox.AP",
                    help="Primary metric to sort/summarize by (default bbox.AP)")
    ap.add_argument("--paired", action="store_true",
                    help="Treat runs as paired by index across groups (recommended if seeds align).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Resolve groups -> list of run dirs
    groups: Dict[str, List[str]] = {}
    for spec in args.group:
        if ":" not in spec:
            print(f"Bad --group spec: {spec}")
            sys.exit(2)
        label, pattern = spec.split(":", 1)
        dirs = sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])
        if not dirs:
            print(f"[WARN] No run dirs matched: {pattern}")
        groups[label] = dirs
        print(f"Group '{label}' -> {len(dirs)} runs")

    # Collect per-run metrics
    per_run_rows = []
    all_metric_keys = set(["val_loss","latency_ms","param_count","best_val_loss_snapshot"])
    for label, dirs in groups.items():
        for d in dirs:
            try:
                m = read_best_epoch_metrics(d)
            except Exception as e:
                print(f"[WARN] Skipping {d}: {e}")
                continue
            row = {"group": label, "run_dir": os.path.abspath(d)}
            row.update(m)
            per_run_rows.append(row)
            all_metric_keys.update(m.keys())

    if not per_run_rows:
        print("No runs parsed. Exiting.")
        return

    # Write per-run CSV
    per_run_csv = os.path.join(args.out, "per_run_metrics.csv")
    cols = ["group", "run_dir"] + sorted(all_metric_keys)
    with open(per_run_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in per_run_rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"Wrote {per_run_csv}")

    # Build arrays per-group per-metric
    # metrics_to_use: all 'bbox.' and 'phenotype.' keys present
    metric_keys = sorted([k for k in all_metric_keys if k.startswith("bbox.") or k.startswith("phenotype.")])

    # Group -> {metric -> np.array}
    grouped: Dict[str, Dict[str, np.ndarray]] = {g: {} for g in groups}
    for g in groups:
        rows = [r for r in per_run_rows if r["group"]==g]
        for mk in metric_keys:
            vals = []
            for r in rows:
                v = r.get(mk, None)
                if v is None or v=="":
                    vals.append(np.nan)
                else:
                    vals.append(float(v))
            grouped[g][mk] = np.array(vals, dtype=float)

    # Group summary (mean/std/CI)
    summary_rows = []
    for g in groups:
        for mk in metric_keys:
            m, s, (lo, hi) = mean_std_ci(grouped[g][mk])
            summary_rows.append({
                "group": g, "metric": mk,
                "n": int(np.sum(~np.isnan(grouped[g][mk]))),
                "mean": m, "std": s, "ci95_low": lo, "ci95_high": hi
            })

    group_csv = os.path.join(args.out, "group_summary.csv")
    with open(group_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group","metric","n","mean","std","ci95_low","ci95_high"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"Wrote {group_csv}")

    # Pairwise significance across groups (per metric)
    pair_rows = []
    group_list = list(groups.keys())
    for mk in metric_keys:
        pvals_for_holm = []
        tmp_rows = []

        for (g1, g2) in combinations(group_list, 2):
            a = grouped[g1][mk]
            b = grouped[g2][mk]

            # Trim to same length if paired requested
            if args.paired:
                n = min(len(a), len(b))
                a = a[:n]; b = b[:n]

            tests = paired_tests(a, b) if args.paired else (
                # Welch's t-test + Mann-Whitney as nonparam if SciPy available
                (lambda A,B: (
                    {"t_stat": spstats.ttest_ind(A[~np.isnan(A)], B[~np.isnan(B)], equal_var=False, nan_policy="omit").statistic,
                     "t_p": spstats.ttest_ind(A[~np.isnan(A)], B[~np.isnan(B)], equal_var=False, nan_policy="omit").pvalue,
                     "wilcoxon_stat": spstats.mannwhitneyu(A[~np.isnan(A)], B[~np.isnan(B)], alternative="two-sided").statistic if SCIPY else np.nan,
                     "wilcoxon_p": spstats.mannwhitneyu(A[~np.isnan(A)], B[~np.isnan(B)], alternative="two-sided").pvalue if SCIPY else np.nan
                    }) if SCIPY else
                    # Bootstrap fallback
                    (lambda A,B: _bootstrap_unpaired(A,B))
                )(a,b)
            )

            d = cohen_d(a, b, paired=args.paired)
            delta = cliffs_delta(a, b)

            row = {
                "metric": mk,
                "group1": g1, "group2": g2,
                "paired": int(args.paired),
                "n1": int(np.sum(~np.isnan(a))), "n2": int(np.sum(~np.isnan(b))),
                "t_stat": tests["t_stat"], "t_p": tests["t_p"],
                "rank_stat": tests["wilcoxon_stat"], "rank_p": tests["wilcoxon_p"],
                "cohen_d": d, "cliffs_delta": delta,
            }
            tmp_rows.append(row)
            if not math.isnan(row["t_p"]):
                pvals_for_holm.append(row["t_p"])
            else:
                pvals_for_holm.append(1.0)

        # Holm–Bonferroni adjust within this metric
        adj = holm_bonferroni(pvals_for_holm) if len(pvals_for_holm) > 0 else []
        for r, padj in zip(tmp_rows, adj):
            r["t_p_holm"] = padj
            pair_rows.append(r)

    pair_csv = os.path.join(args.out, "pairwise_tests.csv")
    with open(pair_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "metric","group1","group2","paired","n1","n2",
            "t_stat","t_p","t_p_holm","rank_stat","rank_p","cohen_d","cliffs_delta"
        ])
        w.writeheader()
        for r in pair_rows:
            w.writerow(r)
    print(f"Wrote {pair_csv}")

    # Markdown summary for the primary metric
    primary = args.metric
    md = [f"# Aggregate & Significance Summary",
          f"- Primary metric: **{primary}**",
          f"- Paired: **{args.paired}**",
          ""]
    md.append("## Group means (95% CI)")
    md.append("")
    md.append("| Group | n | mean | 95% CI |")
    md.append("|---|---:|---:|---:|")
    for g in group_list:
        row = next((r for r in summary_rows if r["group"]==g and r["metric"]==primary), None)
        if row:
            md.append(f"| {g} | {row['n']} | {row['mean']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |")
    md.append("")
    md.append("## Pairwise tests (Holm–Bonferroni adjusted on t-test p)")
    md.append("")
    md.append("| g1 | g2 | t_p | t_p_holm | rank_p | Cohen d | Cliff Δ |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for r in pair_rows:
        if r["metric"]==primary:
            md.append(f"| {r['group1']} | {r['group2']} | "
                      f"{r['t_p']:.4g} | {r['t_p_holm']:.4g} | "
                      f"{(r['rank_p'] if not math.isnan(r['rank_p']) else float('nan')):.4g} | "
                      f"{r['cohen_d']:.3f} | {r['cliffs_delta']:.3f} |")
    md_path = os.path.join(args.out, "README_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Wrote {md_path}")


def _bootstrap_unpaired(A, B):
    A = np.asarray(A, float); B = np.asarray(B, float)
    A = A[~np.isnan(A)]; B = B[~np.isnan(B)]
    if len(A) < 2 or len(B) < 2:
        return {"t_stat": np.nan, "t_p": np.nan, "wilcoxon_stat": np.nan, "wilcoxon_p": np.nan}
    rng = np.random.default_rng(123)
    obs = float(np.mean(A) - np.mean(B))
    Bn = 20000
    boot = []
    for _ in range(Bn):
        a = rng.choice(A, size=len(A), replace=True)
        b = rng.choice(B, size=len(B), replace=True)
        boot.append(np.mean(a) - np.mean(b))
    boot = np.array(boot)
    p = 2*min(np.mean(boot >= obs), np.mean(boot <= obs))
    return {"t_stat": np.nan, "t_p": float(p), "wilcoxon_stat": np.nan, "wilcoxon_p": np.nan}


if __name__ == "__main__":
    main()
