import os
import json
import pickle
import torch
from torch.utils.data import DataLoader

from configs.config import (
    PROCESSED_DIR,
    RESULTS_DIR,
    CHECKPOINT_DIR,
    ATOM_FEATURE_DIM,
    GENE_FEATURE_DIM,
    HIDDEN_DIM,
    BATCH_SIZE,
    VARIANCE_SEEDS,
    DEVICE,
)
from data.preprocessing import build_dataset, normalize_expression
from data.smiles_resolution import (
    resolve_all_drug_smiles,
    save_smiles_cache,
    filter_dataset_to_resolved_drugs,
)
from data.graph_utils import build_functional_group_patterns, build_all_drug_graphs, save_drug_graphs
from data.dataset import DrugResponseDataset, collate_fn, split_by_cell_line

from src.train import (
    train_dsmrgat,
    train_two_relation_ablation,
    train_single_relation_baseline,
    save_checkpoint,
    set_all_seeds,
)
from src.eval import evaluate_dsmrgat, evaluate_two_relation, evaluate_single_relation
from src.metrics import bootstrap_r2_ci, summarize_seed_variance
from src.relation_diagnostics import (
    diagnose_relation_density,
    compute_output_magnitude_table,
    summarize_output_magnitude,
    compute_pointwise_relation_importance,
    build_relation_contribution_summary,
    save_relation_contribution_summary,
)
from src.dashboard import (
    compute_validation_mae,
    build_sensitivity_probabilities,
    personalized_ranking_hit_rate,
    non_personalized_baseline_hit_rate,
    build_dashboard_summary,
    save_dashboard_summary,
)
from src.visualize import (
    plot_r2_comparison_bar,
    plot_bootstrap_distribution,
    plot_relation_contribution,
    plot_ablation_drop,
    plot_parity,
)


def run_pipeline():
    # 1. Data preparation ---------------------------------------------------
    final_dataset, top_genes, train_cell_lines, val_cell_lines, test_cell_lines = build_dataset()

    # Requires screened_compounds_rel_8.5.csv (drug synonyms) to be loaded separately;
    # see data/README.md for the full raw-data layout.
    import pandas as pd
    from configs.config import RAW_DIR
    screened_compounds = pd.read_csv(f"{RAW_DIR}/screened_compounds_rel_8.5.csv")

    drug_to_smiles = resolve_all_drug_smiles(final_dataset, screened_compounds)
    save_smiles_cache(drug_to_smiles)
    final_dataset_with_smiles = filter_dataset_to_resolved_drugs(final_dataset, drug_to_smiles)

    functional_group_patterns = build_functional_group_patterns()
    drug_graphs, failed_drugs = build_all_drug_graphs(drug_to_smiles)
    save_drug_graphs(drug_graphs)

    valid_drugs = set(drug_graphs.keys())
    final_dataset_with_smiles = final_dataset_with_smiles[
        final_dataset_with_smiles["DRUG_NAME"].isin(valid_drugs)
    ]

    cell_line_expression_normalized, _, _ = normalize_expression(
        final_dataset_with_smiles, top_genes, train_cell_lines
    )

    train_df, val_df, test_df = split_by_cell_line(
        final_dataset_with_smiles, train_cell_lines, val_cell_lines, test_cell_lines
    )

    train_dataset = DrugResponseDataset(train_df, drug_graphs, cell_line_expression_normalized)
    val_dataset = DrugResponseDataset(val_df, drug_graphs, cell_line_expression_normalized)
    test_dataset = DrugResponseDataset(test_df, drug_graphs, cell_line_expression_normalized)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 2. Train the full three-relation DSMR-GAT model ------------------------
    model, optimizer = train_dsmrgat(train_loader, val_loader, seed=VARIANCE_SEEDS[0])
    r2_full, preds_full, labels_full = evaluate_dsmrgat(model, test_loader)
    print(f"DSMR-GAT Test R2: {r2_full:.4f}")

    save_checkpoint(model, optimizer, r2_full, "drug_response_model_dsmr_gat.ckpt",
                     atom_feature_dim=ATOM_FEATURE_DIM, gene_feature_dim=GENE_FEATURE_DIM, hidden_dim=HIDDEN_DIM)

    bootstrap_scores, ci_lower, ci_upper = bootstrap_r2_ci(labels_full, preds_full)
    print(f"95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # 3. Relation-level ablation ---------------------------------------------
    model_br, opt_br = train_two_relation_ablation("bond", "ring", train_loader, val_loader, seed=VARIANCE_SEEDS[0])
    r2_bond_ring, _, _ = evaluate_two_relation(model_br, test_loader, "bond", "ring")
    save_checkpoint(model_br, opt_br, r2_bond_ring, "model_bond_ring.ckpt")

    model_bf, opt_bf = train_two_relation_ablation("bond", "fg", train_loader, val_loader, seed=VARIANCE_SEEDS[0])
    r2_bond_fg, _, _ = evaluate_two_relation(model_bf, test_loader, "bond", "fg")
    save_checkpoint(model_bf, opt_bf, r2_bond_fg, "model_bond_fg.ckpt")

    model_single, opt_single = train_single_relation_baseline(train_loader, val_loader, seed=VARIANCE_SEEDS[0])
    r2_single, _, _ = evaluate_single_relation(model_single, test_loader)
    save_checkpoint(model_single, opt_single, r2_single, "model_single_relation.ckpt")

    ablation_summary = {
        "single_relation_gat_bond_only": r2_single,
        "bond_plus_ring": r2_bond_ring,
        "bond_plus_fg": r2_bond_fg,
        "bond_plus_ring_plus_fg_full": r2_full,
    }
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(f"{PROCESSED_DIR}/ablation_summary.json", "w") as f:
        json.dump(ablation_summary, f, indent=2)

    # 3b. Training-run (seed) variance, on top of the primary seed above -----
    r2_full_by_seed = {VARIANCE_SEEDS[0]: r2_full}
    r2_br_by_seed = {VARIANCE_SEEDS[0]: r2_bond_ring}
    r2_bf_by_seed = {VARIANCE_SEEDS[0]: r2_bond_fg}
    r2_single_by_seed = {VARIANCE_SEEDS[0]: r2_single}

    for seed in VARIANCE_SEEDS[1:]:
        m, _ = train_dsmrgat(train_loader, val_loader, seed=seed)
        r2_full_by_seed[seed], _, _ = evaluate_dsmrgat(m, test_loader)

        m, _ = train_two_relation_ablation("bond", "ring", train_loader, val_loader, seed=seed)
        r2_br_by_seed[seed], _, _ = evaluate_two_relation(m, test_loader, "bond", "ring")

        m, _ = train_two_relation_ablation("bond", "fg", train_loader, val_loader, seed=seed)
        r2_bf_by_seed[seed], _, _ = evaluate_two_relation(m, test_loader, "bond", "fg")

        m, _ = train_single_relation_baseline(train_loader, val_loader, seed=seed)
        r2_single_by_seed[seed], _, _ = evaluate_single_relation(m, test_loader)

    seed_variance_summary = {
        "seeds_used": VARIANCE_SEEDS,
        "full_dsmrgat": summarize_seed_variance(r2_full_by_seed),
        "bond_plus_ring": summarize_seed_variance(r2_br_by_seed),
        "bond_plus_fg": summarize_seed_variance(r2_bf_by_seed),
        "single_relation": summarize_seed_variance(r2_single_by_seed),
    }
    with open(f"{PROCESSED_DIR}/seed_variance_summary.json", "w") as f:
        json.dump(seed_variance_summary, f, indent=2)

    # 4. Relation contribution diagnostics ------------------------------------
    density_diagnostics = diagnose_relation_density(drug_graphs)
    magnitude_df = compute_output_magnitude_table(model, drug_graphs)
    magnitude_summary = summarize_output_magnitude(magnitude_df)
    pointwise_importance = compute_pointwise_relation_importance(model)

    relation_contribution_summary = build_relation_contribution_summary(
        density_diagnostics, magnitude_summary, pointwise_importance
    )
    save_relation_contribution_summary(relation_contribution_summary)

    # 5. Clinical decision dashboard -------------------------------------------
    mae_val = compute_validation_mae(model, val_loader)
    test_df_with_predictions, drug_thresholds = build_sensitivity_probabilities(test_df, preds_full, train_df, mae_val)
    personalized_hit_rate, personalized_correct, personalized_total = personalized_ranking_hit_rate(test_df_with_predictions)
    most_broadly_effective_drug, baseline_hit_rate, baseline_correct, baseline_total = non_personalized_baseline_hit_rate(
        train_df, test_df_with_predictions, drug_thresholds
    )
    dashboard_summary = build_dashboard_summary(
        personalized_hit_rate, personalized_correct, personalized_total,
        most_broadly_effective_drug, baseline_hit_rate, baseline_correct, baseline_total, mae_val,
    )
    save_dashboard_summary(dashboard_summary)

    # 6. Save headline result figures -------------------------------------------
    os.makedirs(f"{RESULTS_DIR}/ablation", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/model_comparison", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/relation_diagnostics", exist_ok=True)

    plot_r2_comparison_bar(r2_bond_fg, r2_bond_ring, r2_full, r2_single, ci_lower, ci_upper,
                            f"{RESULTS_DIR}/ablation/fig_r2_comparison")
    plot_bootstrap_distribution(bootstrap_scores, r2_full, ci_lower, ci_upper,
                                 f"{RESULTS_DIR}/model_comparison/fig_bootstrap_distribution")
    plot_relation_contribution(magnitude_summary, pointwise_importance,
                                f"{RESULTS_DIR}/relation_diagnostics/fig_relation_contribution")
    plot_ablation_drop(r2_bond_ring, r2_bond_fg, r2_full,
                        f"{RESULTS_DIR}/ablation/fig_ablation_drop")
    plot_parity(labels_full, preds_full, r2_full,
                f"{RESULTS_DIR}/model_comparison/fig_parity_plot")

    # 7. Final results summary ---------------------------------------------------
    final_results_summary = {
        "main_model_r2": r2_full,
        "bootstrap_ci_95": [ci_lower, ci_upper],
        "seed_variance": seed_variance_summary,
        "ablation": ablation_summary,
        "relation_contribution": relation_contribution_summary,
        "clinical_dashboard": dashboard_summary,
    }
    with open(f"{PROCESSED_DIR}/final_results_summary.json", "w") as f:
        json.dump(final_results_summary, f, indent=2)

    print("\nPipeline complete. See data/processed/final_results_summary.json for the full summary,")
    print("and assets/results/ for the saved figures.")
    print("\nFor gene attribution (SLFN11), atom-level ring-relation attribution, and error analysis,")
    print("see src/explainability.py and src/metrics.py — these require the Camptothecin test rows")
    print("and are run separately since they target specific drugs/analyses (see README.md).")


if __name__ == "__main__":
    run_pipeline()
