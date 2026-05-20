from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover - keeps the script usable without scipy
    wilcoxon = None


def cohen_dz(diff: np.ndarray) -> float:
    """Return Cohen's dz for paired samples."""
    diff = diff.astype(float)
    sd = diff.std(ddof=1)
    if diff.size < 2 or sd == 0 or math.isnan(sd):
        return float("nan")
    return float(diff.mean() / sd)


def paired_wilcoxon(diff: np.ndarray) -> float:
    """Return a paired Wilcoxon p-value, or NaN if scipy is unavailable."""
    if wilcoxon is None or diff.size < 2:
        return float("nan")
    try:
        return float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def numeric_metrics(df: pd.DataFrame, excluded: Iterable[str]) -> list[str]:
    excluded = set(excluded)
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def compare_against_reference(
    df: pd.DataFrame,
    reference: str,
    dataset_col: str,
    task_col: str,
    fold_col: str,
    method_col: str,
    metrics: list[str],
) -> pd.DataFrame:
    rows = []
    groups = [dataset_col]
    if task_col in df.columns:
        groups.append(task_col)

    for group_key, group_df in df.groupby(groups, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_info = dict(zip(groups, group_key))
        ref_df = group_df[group_df[method_col] == reference]
        if ref_df.empty:
            continue

        for method, method_df in group_df.groupby(method_col):
            if method == reference:
                continue

            merged = ref_df[[fold_col] + metrics].merge(
                method_df[[fold_col] + metrics],
                on=fold_col,
                suffixes=("_ref", "_cmp"),
            )
            if merged.empty:
                continue

            for metric in metrics:
                ref_values = merged[f"{metric}_ref"].dropna().to_numpy(dtype=float)
                cmp_values = merged[f"{metric}_cmp"].dropna().to_numpy(dtype=float)
                valid = np.isfinite(ref_values) & np.isfinite(cmp_values)
                ref_values = ref_values[valid]
                cmp_values = cmp_values[valid]
                if ref_values.size == 0:
                    continue

                diff = ref_values - cmp_values
                rows.append(
                    {
                        **group_info,
                        "metric": metric,
                        "reference": reference,
                        "comparison": method,
                        "n_folds": int(ref_values.size),
                        "reference_mean": float(ref_values.mean()),
                        "reference_std": float(ref_values.std(ddof=1))
                        if ref_values.size > 1
                        else 0.0,
                        "comparison_mean": float(cmp_values.mean()),
                        "comparison_std": float(cmp_values.std(ddof=1))
                        if cmp_values.size > 1
                        else 0.0,
                        "mean_paired_difference": float(diff.mean()),
                        "cohen_dz": cohen_dz(diff),
                        "wilcoxon_p": paired_wilcoxon(diff),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Fold-wise result CSV.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--reference", default="MMDiffuzzy", help="Reference method.")
    parser.add_argument("--dataset-col", default="dataset")
    parser.add_argument("--task-col", default="task")
    parser.add_argument("--fold-col", default="fold")
    parser.add_argument("--method-col", default="method")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Metrics to test. Defaults to all numeric non-index columns.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    excluded = [args.dataset_col, args.task_col, args.fold_col, args.method_col]
    metrics = args.metrics or numeric_metrics(df, excluded)
    if not metrics:
        raise ValueError("No numeric metric columns found.")

    result = compare_against_reference(
        df=df,
        reference=args.reference,
        dataset_col=args.dataset_col,
        task_col=args.task_col,
        fold_col=args.fold_col,
        method_col=args.method_col,
        metrics=metrics,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"Saved {len(result)} statistical comparisons to {out}")


if __name__ == "__main__":
    main()