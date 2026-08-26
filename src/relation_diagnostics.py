import json
import torch
import pandas as pd
from torch_geometric.utils import add_self_loops
from configs.config import PROCESSED_DIR, RESULTS_DIR, DEVICE, RELATIONS


def diagnose_relation_density(drug_graphs):
    results = {name: {"total_atoms": 0, "self_loop_only_atoms": 0} for name in RELATIONS}

    for graph in drug_graphs.values():
        num_atoms = graph.x.size(0)
        for rel_name in RELATIONS:
            edge_index = getattr(graph, f"edge_index_{rel_name}")
            edge_index_sl, _ = add_self_loops(edge_index, num_nodes=num_atoms)
            target = edge_index_sl[1]

            degree_per_node = torch.bincount(target, minlength=num_atoms)
            self_loop_only = (degree_per_node == 1).sum().item()

            results[rel_name]["total_atoms"] += num_atoms
            results[rel_name]["self_loop_only_atoms"] += self_loop_only

    for rel_name in RELATIONS:
        total = results[rel_name]["total_atoms"]
        self_only = results[rel_name]["self_loop_only_atoms"]
        print(f"{rel_name}: {self_only}/{total} atoms ({self_only / total:.1%}) isolated in this relation")

    return results


def compute_relation_output_magnitude(model, graph, drug_name):
    model.eval()
    with torch.no_grad():
        x = graph.x.to(DEVICE)
        layer1 = model.drug_encoder.layer1

        z_bond = layer1.attn_bond(x, graph.edge_index_bond.to(DEVICE))
        z_ring = layer1.attn_ring(x, graph.edge_index_ring.to(DEVICE))
        z_fg = layer1.attn_fg(x, graph.edge_index_fg.to(DEVICE))

    return {
        "drug_name": drug_name,
        "magnitude_bond": z_bond.norm(dim=1).mean().item(),
        "magnitude_ring": z_ring.norm(dim=1).mean().item(),
        "magnitude_fg": z_fg.norm(dim=1).mean().item(),
    }


def compute_output_magnitude_table(model, drug_graphs):
    summaries = [compute_relation_output_magnitude(model, graph, name) for name, graph in drug_graphs.items()]
    return pd.DataFrame(summaries)


def summarize_output_magnitude(magnitude_df):
    mean_mag_bond = magnitude_df["magnitude_bond"].mean()
    mean_mag_ring = magnitude_df["magnitude_ring"].mean()
    mean_mag_fg = magnitude_df["magnitude_fg"].mean()
    total = mean_mag_bond + mean_mag_ring + mean_mag_fg

    print("Relation output magnitude (percentage of total):")
    print(f"Bond: {mean_mag_bond:.4f} ({mean_mag_bond / total:.1%})")
    print(f"Ring: {mean_mag_ring:.4f} ({mean_mag_ring / total:.1%})")
    print(f"Functional Group: {mean_mag_fg:.4f} ({mean_mag_fg / total:.1%})")

    return {"bond": mean_mag_bond, "ring": mean_mag_ring, "fg": mean_mag_fg}


def compute_pointwise_relation_importance(model):
    layer1_weight = model.drug_encoder.layer1.pointwise.weight.detach().cpu()
    hidden_dim = layer1_weight.shape[1] // 3

    bond_norm = layer1_weight[:, 0:hidden_dim].norm().item()
    ring_norm = layer1_weight[:, hidden_dim:2 * hidden_dim].norm().item()
    fg_norm = layer1_weight[:, 2 * hidden_dim:3 * hidden_dim].norm().item()

    total = bond_norm + ring_norm + fg_norm
    print("Pointwise weight norm per relation (Layer 1):")
    print(f"Bond: {bond_norm:.4f} ({bond_norm / total:.1%})")
    print(f"Ring: {ring_norm:.4f} ({ring_norm / total:.1%})")
    print(f"Functional Group: {fg_norm:.4f} ({fg_norm / total:.1%})")

    return {"bond_norm": bond_norm, "ring_norm": ring_norm, "fg_norm": fg_norm}


def build_relation_contribution_summary(density_diagnostics, magnitude_summary, pointwise_importance):
    return {
        "self_loop_fraction": {
            name: density_diagnostics[name]["self_loop_only_atoms"] / density_diagnostics[name]["total_atoms"]
            for name in RELATIONS
        },
        "output_magnitude": magnitude_summary,
        "pointwise_weight_norm": pointwise_importance,
    }


def save_relation_contribution_summary(summary):
    with open(f"{PROCESSED_DIR}/relation_contribution_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def load_relation_contribution_summary():
    with open(f"{PROCESSED_DIR}/relation_contribution_summary.json", "r") as f:
        return json.load(f)
