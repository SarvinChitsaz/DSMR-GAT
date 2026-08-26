import statistics as stats
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def bootstrap_r2_ci(y_true, y_pred, n_bootstrap=1000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)

    bootstrap_r2_scores = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        bootstrap_r2_scores.append(r2_score(y_true[indices], y_pred[indices]))

    lower = np.percentile(bootstrap_r2_scores, (100 - ci) / 2)
    upper = np.percentile(bootstrap_r2_scores, 100 - (100 - ci) / 2)
    return np.array(bootstrap_r2_scores), lower, upper


def summarize_seed_variance(r2_by_seed):
    values = list(r2_by_seed.values())
    return {
        "mean": stats.mean(values),
        "std": stats.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "values_by_seed": r2_by_seed,
    }


def mean_absolute_error_by_group(df, predicted_column, actual_column, group_columns=("CANCER_TYPE", "DRUG_NAME"), min_count=3):
    df = df.copy()
    df["abs_error"] = (df[predicted_column] - df[actual_column]).abs()

    group_columns = list(group_columns)
    pair_counts = df.groupby(group_columns).size()
    valid_pairs = pair_counts[pair_counts >= min_count].index

    indexed = df.set_index(group_columns)
    indexed = indexed.loc[indexed.index.isin(valid_pairs)]

    return indexed.groupby(group_columns)["abs_error"].mean().reset_index()


def build_error_heatmap_table(mae_table, top_n_drugs=18, top_n_cancers=20):
    drug_coverage = mae_table["DRUG_NAME"].value_counts().head(top_n_drugs).index.tolist()
    cancer_coverage = mae_table["CANCER_TYPE"].value_counts().head(top_n_cancers).index.tolist()

    filtered = mae_table[
        mae_table["DRUG_NAME"].isin(drug_coverage) & mae_table["CANCER_TYPE"].isin(cancer_coverage)
    ]
    pivot = filtered.pivot(index="CANCER_TYPE", columns="DRUG_NAME", values="abs_error")

    cancer_order = pivot.mean(axis=1).sort_values(ascending=False).index
    drug_order = pivot.mean(axis=0).sort_values(ascending=False).index
    return pivot.loc[cancer_order, drug_order]
