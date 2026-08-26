import json
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from configs.config import PROCESSED_DIR, DEVICE


def compute_validation_mae(model, val_loader):
    model.eval()
    val_predictions = []
    val_labels = []

    with torch.no_grad():
        for drug_batch, expressions, labels in val_loader:
            drug_batch = drug_batch.to(DEVICE)
            expressions = expressions.to(DEVICE)

            predictions = model(
                drug_batch.x,
                drug_batch.edge_index_bond,
                drug_batch.edge_index_ring,
                drug_batch.edge_index_fg,
                drug_batch.batch,
                expressions,
            )
            val_predictions.extend(predictions.cpu().tolist())
            val_labels.extend(labels.tolist())

    return mean_absolute_error(val_labels, val_predictions)


def build_sensitivity_probabilities(test_df, predictions, train_df, mae_val):
    test_df = test_df.reset_index(drop=True).copy()
    test_df["predicted_ln_ic50"] = predictions

    drug_thresholds = train_df.groupby("DRUG_NAME")["LN_IC50"].median()
    test_df["drug_threshold"] = test_df["DRUG_NAME"].map(drug_thresholds)

    test_df["sensitivity_probability"] = 1 / (
        1 + np.exp(-(test_df["drug_threshold"] - test_df["predicted_ln_ic50"]) / mae_val)
    )
    test_df["actual_sensitive"] = (test_df["LN_IC50"] < test_df["drug_threshold"]).astype(int)
    return test_df, drug_thresholds


def personalized_ranking_hit_rate(test_df_with_predictions):
    correct = 0
    total = 0
    for _, group in test_df_with_predictions.groupby("SANGER_MODEL_ID"):
        if len(group) < 2:
            continue
        top_row = group.loc[group["sensitivity_probability"].idxmax()]
        total += 1
        correct += int(top_row["actual_sensitive"] == 1)

    hit_rate = correct / total
    print(f"Personalized top-1 recommendation: {correct}/{total} = {hit_rate:.1%}")
    return hit_rate, correct, total


def non_personalized_baseline_hit_rate(train_df, test_df_with_predictions, drug_thresholds):
    train_df_labeled = train_df.copy()
    train_df_labeled["drug_threshold"] = train_df_labeled["DRUG_NAME"].map(drug_thresholds)
    train_df_labeled["actual_sensitive"] = (train_df_labeled["LN_IC50"] < train_df_labeled["drug_threshold"]).astype(int)

    drug_broad_effectiveness = train_df_labeled.groupby("DRUG_NAME")["actual_sensitive"].mean().sort_values(ascending=False)
    most_broadly_effective_drug = drug_broad_effectiveness.index[0]

    baseline_subset = test_df_with_predictions[test_df_with_predictions["DRUG_NAME"] == most_broadly_effective_drug]
    baseline_correct = baseline_subset["actual_sensitive"].sum()
    baseline_total = len(baseline_subset)
    baseline_hit_rate = baseline_correct / baseline_total if baseline_total > 0 else float("nan")

    print(f"Non-personalized baseline ({most_broadly_effective_drug}): {baseline_correct}/{baseline_total} = {baseline_hit_rate:.1%}")
    return most_broadly_effective_drug, baseline_hit_rate, baseline_correct, baseline_total


def build_dashboard_summary(personalized_hit_rate, personalized_correct, personalized_total,
                             most_broadly_effective_drug, baseline_hit_rate, baseline_correct,
                             baseline_total, mae_val):
    gap = (personalized_hit_rate - baseline_hit_rate) * 100
    print(f"Gap: {gap:.1f} percentage points favoring personalized ranking")

    return {
        "personalized_hit_rate": personalized_hit_rate,
        "personalized_correct": int(personalized_correct),
        "personalized_total": int(personalized_total),
        "most_broadly_effective_drug": most_broadly_effective_drug,
        "baseline_hit_rate": float(baseline_hit_rate),
        "baseline_correct": int(baseline_correct),
        "baseline_total": int(baseline_total),
        "validation_mae": mae_val,
    }


def save_dashboard_summary(summary):
    with open(f"{PROCESSED_DIR}/dashboard_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def load_dashboard_summary():
    with open(f"{PROCESSED_DIR}/dashboard_summary.json", "r") as f:
        return json.load(f)
