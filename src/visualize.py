import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
from scipy.stats import gaussian_kde
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageDraw, ImageFont
from data.graph_utils import get_bond_edges, get_ring_edges, get_functional_group_edges

RELATION_COLORS = {"bond": "#4C72B0", "ring": "#55A868", "fg": "#DD8452"}


def build_relation_graph(num_atoms, edge_list):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_atoms))
    graph.add_edges_from(edge_list)
    return graph


def _atom_layout(mol):
    num_atoms = mol.GetNumAtoms()
    AllChem.Compute2DCoords(mol)
    conformer = mol.GetConformer()
    positions = {i: (conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y) for i in range(num_atoms)}
    labels = {i: mol.GetAtomWithIdx(i).GetSymbol() for i in range(num_atoms)}
    return num_atoms, positions, labels


# ---------------------------------------------------------------------------
# Multi-relation molecular graph decomposition (Figs. 2/3 of the paper)
# ---------------------------------------------------------------------------


def plot_multi_relation_graph_grid(example_drug_names, drug_to_smiles, functional_group_patterns, save_path):
    """One row per drug, four panels: Bond / Ring / Functional Group / Combined."""
    n = len(example_drug_names)
    fig, axes = plt.subplots(n, 4, figsize=(24, 6 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row_idx, drug_name in enumerate(example_drug_names):
        mol = Chem.MolFromSmiles(drug_to_smiles[drug_name])
        num_atoms, positions, labels = _atom_layout(mol)

        bond_edges = get_bond_edges(mol)
        ring_edges = get_ring_edges(mol)
        fg_edges = get_functional_group_edges(mol, functional_group_patterns)

        bond_graph = build_relation_graph(num_atoms, bond_edges)
        ring_graph = build_relation_graph(num_atoms, ring_edges)
        fg_graph = build_relation_graph(num_atoms, fg_edges)

        node_kwargs = dict(node_color="lightgray", node_size=250, edgecolors="black")

        nx.draw(bond_graph, pos=positions, ax=axes[row_idx, 0], labels=labels,
                edge_color="steelblue", width=2, font_size=7, **node_kwargs)
        axes[row_idx, 0].set_title(f"{drug_name}\nBond ({len(bond_edges)} edges)")

        nx.draw(ring_graph, pos=positions, ax=axes[row_idx, 1], labels=labels,
                edge_color="seagreen", width=2, style="dashed", font_size=7, **node_kwargs)
        axes[row_idx, 1].set_title(f"{drug_name}\nRing ({len(ring_edges)} edges)")

        nx.draw(fg_graph, pos=positions, ax=axes[row_idx, 2], labels=labels,
                edge_color="darkorange", width=2, style="dotted", font_size=7, **node_kwargs)
        axes[row_idx, 2].set_title(f"{drug_name}\nFunctional Group ({len(fg_edges)} edges)")

        nx.draw(bond_graph, pos=positions, ax=axes[row_idx, 3], labels=labels,
                edge_color="steelblue", width=2, font_size=7, **node_kwargs)
        nx.draw_networkx_edges(ring_graph, pos=positions, ax=axes[row_idx, 3],
                                edge_color="seagreen", width=1.5, style="dashed")
        nx.draw_networkx_edges(fg_graph, pos=positions, ax=axes[row_idx, 3],
                                edge_color="darkorange", width=1.5, style="dotted")
        axes[row_idx, 3].set_title(f"{drug_name}\nCombined (Bond + Ring + FG)")

    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_path}.png", dpi=500, bbox_inches="tight")
    plt.close()


def plot_multi_relation_single_example(drug_name, drug_to_smiles, functional_group_patterns, save_path):
    """Bond / Ring / Functional Group / Combined for a single drug (Fig. 2)."""
    mol = Chem.MolFromSmiles(drug_to_smiles[drug_name])
    num_atoms, positions, labels = _atom_layout(mol)

    bond_edges = get_bond_edges(mol)
    ring_edges = get_ring_edges(mol)
    fg_edges = get_functional_group_edges(mol, functional_group_patterns)

    bond_graph = build_relation_graph(num_atoms, bond_edges)
    ring_graph = build_relation_graph(num_atoms, ring_edges)
    fg_graph = build_relation_graph(num_atoms, fg_edges)

    node_kwargs = dict(node_color="lightgray", node_size=180, edgecolors="black")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))

    nx.draw(bond_graph, pos=positions, ax=axes[0], labels=labels,
            edge_color="steelblue", width=1.5, font_size=6, **node_kwargs)
    axes[0].set_title(f"Bond\n({len(bond_edges)} edges)", fontsize=10)

    nx.draw(ring_graph, pos=positions, ax=axes[1], labels=labels,
            edge_color="seagreen", width=1.5, style="dashed", font_size=6, **node_kwargs)
    axes[1].set_title(f"Ring\n({len(ring_edges)} edges)", fontsize=10)

    nx.draw(fg_graph, pos=positions, ax=axes[2], labels=labels,
            edge_color="darkorange", width=1.5, style="dotted", font_size=6, **node_kwargs)
    axes[2].set_title(f"Functional Group\n({len(fg_edges)} edges)", fontsize=10)

    nx.draw(bond_graph, pos=positions, ax=axes[3], labels=labels,
            edge_color="steelblue", width=1.5, font_size=6, **node_kwargs)
    nx.draw_networkx_edges(ring_graph, pos=positions, ax=axes[3], edge_color="seagreen", width=1.2, style="dashed")
    nx.draw_networkx_edges(fg_graph, pos=positions, ax=axes[3], edge_color="darkorange", width=1.2, style="dotted")
    axes[3].set_title("Combined\n(Bond + Ring + FG)", fontsize=10)

    fig.suptitle(drug_name, fontsize=12, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_path}.png", dpi=500, bbox_inches="tight")
    plt.close()


def plot_multi_relation_combined_only(example_drug_names, drug_to_smiles, functional_group_patterns, save_path):
    """2x2 grid showing only the combined (Bond+Ring+FG) view per drug (Fig. 3)."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 6.5))
    axes = axes.flatten()
    node_kwargs = dict(node_color="lightgray", node_size=90, edgecolors="black", linewidths=0.6)

    for idx, drug_name in enumerate(example_drug_names):
        mol = Chem.MolFromSmiles(drug_to_smiles[drug_name])
        num_atoms, positions, labels = _atom_layout(mol)

        bond_edges = get_bond_edges(mol)
        ring_edges = get_ring_edges(mol)
        fg_edges = get_functional_group_edges(mol, functional_group_patterns)

        bond_graph = build_relation_graph(num_atoms, bond_edges)
        ring_graph = build_relation_graph(num_atoms, ring_edges)
        fg_graph = build_relation_graph(num_atoms, fg_edges)

        ax = axes[idx]
        nx.draw(bond_graph, pos=positions, ax=ax, labels=labels,
                edge_color="steelblue", width=1.1, font_size=5.5, **node_kwargs)
        nx.draw_networkx_edges(ring_graph, pos=positions, ax=ax, edge_color="seagreen", width=0.9, style="dashed")
        nx.draw_networkx_edges(fg_graph, pos=positions, ax=ax, edge_color="darkorange", width=0.9, style="dotted")
        ax.set_title(drug_name, fontsize=10, fontweight="bold")

    legend_elements = [
        Line2D([0], [0], color="steelblue", lw=1.5, label="Bond"),
        Line2D([0], [0], color="seagreen", lw=1.5, linestyle="dashed", label="Ring"),
        Line2D([0], [0], color="darkorange", lw=1.5, linestyle="dotted", label="Functional Group"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Multi-Relation Molecular Graph (Combined View)", fontsize=13, fontweight="bold", y=1.0)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_path}.png", dpi=500, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Functional-group SMARTS pattern verification (Fig. 4 of the paper)
# ---------------------------------------------------------------------------


def find_functional_group_examples(functional_group_patterns, drug_to_smiles, pattern_names, randomize_seed=None):
    """For each named pattern, finds one drug in which it matches (first match,
    or a random match among candidates if randomize_seed is given)."""
    examples = {}
    for pattern_name in pattern_names:
        pattern = functional_group_patterns[pattern_name]
        candidates = []
        for drug_name, smiles in drug_to_smiles.items():
            if smiles is None:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            matches = mol.GetSubstructMatches(pattern)
            if len(matches) > 0:
                candidates.append((drug_name, smiles, mol, matches[0]))
                if randomize_seed is None:
                    break
        if not candidates:
            continue
        if randomize_seed is not None:
            random.seed(randomize_seed)
            examples[pattern_name] = random.choice(candidates)
        else:
            examples[pattern_name] = candidates[0]
    return examples


def _mol_to_svg_fragment(mol, match, size):
    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    drawer.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=list(match))
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return re.search(r"<svg.*?>(.*)</svg>", svg, re.DOTALL).group(1)


def build_functional_group_verification_svg(verification_examples, save_path, n_cols=2, n_rows=2,
                                              panel_size=320, title_height=34, main_title_height=40, gap=20):
    """True-vector SVG grid verifying that named SMARTS patterns match the
    expected substructure on real drug molecules (Fig. 4 in the paper)."""
    selected_keys = list(verification_examples.keys())[: n_cols * n_rows]
    canvas_width = panel_size * n_cols + gap * (n_cols - 1)
    canvas_height = main_title_height + (title_height + panel_size + gap) * n_rows

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" '
        f'viewBox="0 0 {canvas_width} {canvas_height}">',
        f'<rect width="{canvas_width}" height="{canvas_height}" fill="white"/>',
        f'<text x="{canvas_width / 2}" y="{main_title_height * 0.65}" font-family="Helvetica" '
        f'font-size="20" font-weight="bold" text-anchor="middle">Functional Group Pattern Verification</text>',
    ]

    for idx, pattern_name in enumerate(selected_keys):
        drug_name, smiles, mol, match = verification_examples[pattern_name]
        row, col = idx // n_cols, idx % n_cols
        x_offset = col * (panel_size + gap)
        y_offset = main_title_height + row * (title_height + panel_size + gap)

        label = f"{pattern_name} ({drug_name})"
        svg_parts.append(
            f'<text x="{x_offset + panel_size / 2}" y="{y_offset + title_height * 0.7}" '
            f'font-family="Helvetica" font-size="13" text-anchor="middle">{label}</text>'
        )
        svg_parts.append(
            f'<g transform="translate({x_offset},{y_offset + title_height})">'
            f'{_mol_to_svg_fragment(mol, match, panel_size)}</g>'
        )

    svg_parts.append("</svg>")
    final_svg = "\n".join(svg_parts)
    with open(save_path, "w") as f:
        f.write(final_svg)
    return final_svg


# ---------------------------------------------------------------------------
# Model comparison / ablation figures (Section 4.5)
# ---------------------------------------------------------------------------


def plot_r2_comparison_bar(r2_bond_fg, r2_bond_ring, r2_full, r2_single_relation,
                            ci_lower_full, ci_upper_full, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))

    model_names = ["Bond + FG", "Bond + Ring", "Bond + Ring + FG\n(Full DeepRelCDR)", "Single-Relation\nGAT (Bond-only)"]
    r2_values = [r2_bond_fg, r2_bond_ring, r2_full, r2_single_relation]
    colors = ["#DD8452", "#4C72B0", "#2C4E7C", "#8C8C8C"]
    x_pos = np.arange(len(model_names))

    bars = ax.bar(x_pos, r2_values, color=colors, edgecolor="white", width=0.6, zorder=2)
    ax.errorbar(2, r2_full, yerr=[[r2_full - ci_lower_full], [ci_upper_full - r2_full]],
                fmt="none", ecolor="black", capsize=6, linewidth=1.5, zorder=4)

    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.4f}",
                ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel("Test R²", fontsize=12)
    ax.set_ylim(0, 0.88)
    ax.set_title("Model Comparison — Test R²", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_path}.png", dpi=500, bbox_inches="tight")
    plt.close()


def plot_bootstrap_distribution(bootstrap_scores, r2, ci_lower, ci_upper, save_path):
    kde = gaussian_kde(bootstrap_scores)
    x_range = np.linspace(bootstrap_scores.min(), bootstrap_scores.max(), 300)
    density = kde(x_range)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(x_range, density, color="#4C72B0", alpha=0.35)
    ax.plot(x_range, density, color="#1f4e79", linewidth=2)

    ax.axvline(r2, color="black", linewidth=2, label=f"Observed R² = {r2:.4f}")
    ax.axvspan(ci_lower, ci_upper, color="gray", alpha=0.15, label=f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}]")

    ax.set_xlabel("Bootstrap R² (1000 resamples)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Bootstrap Sampling Distribution of Test R²", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_path}.png", dpi=500, bbox_inches="tight")
    plt.close()


def plot_relation_contribution(magnitude_summary, pointwise_importance, save_path):
    relations = ["Bond", "Ring", "Functional Group"]
    relation_colors = ["#4C72B0", "#55A868", "#DD8452"]

    total_mag = sum(magnitude_summary.values())
    total_pw = sum(pointwise_importance.values())

    magnitude_pct = [magnitude_summary["bond"] / total_mag * 100,
                      magnitude_summary["ring"] / total_mag * 100,
                      magnitude_summary["fg"] / total_mag * 100]
    pointwise_pct = [pointwise_importance["bond_norm"] / total_pw * 100,
                      pointwise_importance["ring_norm"] / total_pw * 100,
                      pointwise_importance["fg_norm"] / total_pw * 100]

    y_pos = np.arange(len(relations))
    bar_height = 0.35
    fig, ax = plt.subplots(figsize=(10, 6.5))

    bars1 = ax.barh(y_pos + bar_height / 2, magnitude_pct, height=bar_height,
                     color=relation_colors, edgecolor="white", label="Output Magnitude")
    bars2 = ax.barh(y_pos - bar_height / 2, pointwise_pct, height=bar_height,
                     color=relation_colors, alpha=0.5, edgecolor="white", label="Pointwise Weight Norm")

    for bar, val in zip(bars1, magnitude_pct):
        ax.text(val + 0.7, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
    for bar, val in zip(bars2, pointwise_pct):
        ax.text(val + 0.7, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(relations, fontsize=12)
    ax.set_xlabel("Relative Contribution (%)", fontsize=12)
    ax.set_xlim(0, 42)
    ax.legend(fontsize=10, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.01))
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)

    fig.suptitle("Relation Contribution — Two Independent Metrics", fontsize=13, fontweight="bold", x=0.5, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_path}.png", dpi=500, bbox_inches="tight")
    plt.close()


def plot_ablation_drop(r2_bond_ring, r2_bond_fg, r2_full, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))

    comparisons = [
        ("Bond + Ring\n(FG removed)", r2_bond_ring, "#DD8452"),
        ("Bond + FG\n(Ring removed)", r2_bond_fg, "#55A868"),
    ]
    y_pos = np.arange(len(comparisons))

    for i, (label, ablated_r2, removed_color) in enumerate(comparisons):
        ax.plot([ablated_r2, r2_full], [i, i], color="#d9d9d9", linewidth=4, zorder=1, solid_capstyle="round")
        ax.scatter(ablated_r2, i, s=280, color=removed_color, zorder=3, edgecolor="white", linewidth=2)
        ax.scatter(r2_full, i, s=280, color="#2C4E7C", zorder=3, edgecolor="white", linewidth=2)

        delta = r2_full - ablated_r2
        mid_x = (ablated_r2 + r2_full) / 2
        ax.annotate(f"Δ = {delta:+.4f}", xy=(mid_x, i + 0.24), ha="center", fontsize=10,
                    fontweight="bold", color="#555555")
        ax.text(ablated_r2, i - 0.24, f"{ablated_r2:.4f}", ha="center", fontsize=10.5, fontweight="bold")
        ax.text(r2_full, i - 0.24, f"{r2_full:.4f}", ha="center", fontsize=10.5, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c[0] for c in comparisons], fontsize=12)
    ax.set_xlabel("Test R²", fontsize=12)
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlim(min(r2_bond_fg, r2_bond_ring) - 0.02, r2_full + 0.02)
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2C4E7C", markersize=12,
               markeredgecolor="white", markeredgewidth=1.5, label="Full model (Bond+Ring+FG)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#DD8452", markersize=12,
               markeredgecolor="white", markeredgewidth=1.5, label="Ring removed (FG remains)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#55A868", markersize=12,
               markeredgecolor="white", markeredgewidth=1.5, label="FG removed (Ring remains)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, frameon=True, framealpha=0.9,
              edgecolor="#dddddd", loc="upper left", bbox_to_anchor=(0.01, 0.99))

    fig.suptitle("Relation Ablation — Accuracy Drop When a Relation Is Removed", fontsize=13.5, fontweight="bold", x=0.5, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_path}.png", dpi=500, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Biological / clinical interpretability figures (Sections 4.7, 4.9)
# ---------------------------------------------------------------------------


def plot_slfn11_attribution(slfn11_results, save_path):
    gene_base_color = "#4C72B0"
    slfn11_color = "#DD8452"

    fig, axes = plt.subplots(1, len(slfn11_results), figsize=(20, 5.5))
    if len(slfn11_results) == 1:
        axes = [axes]

    for ax, (cancer_type, result) in zip(axes, slfn11_results.items()):
        genes = [g[0] for g in result["top_5_genes"]][::-1]
        values = [g[1] for g in result["top_5_genes"]][::-1]
        bar_colors = [slfn11_color if g == "SLFN11" else gene_base_color for g in genes]

        y_pos = np.arange(len(genes))
        bars = ax.barh(y_pos, values, color=bar_colors, edgecolor="white", height=0.6, zorder=2)

        for bar, val, gene in zip(bars, values, genes):
            weight = "bold" if gene == "SLFN11" else "normal"
            ax.text(val + max(values) * 0.03, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                    va="center", fontsize=9, fontweight=weight)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(genes, fontsize=11)
        ax.set_title(f"{cancer_type}\n(n={result['n_samples']})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Gradient Importance", fontsize=10)
        ax.set_xlim(0, max(values) * 1.2)
        ax.grid(axis="x", alpha=0.25, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Camptothecin Gene Attribution — SLFN11 (red) Ranks #1 Across Cancer Types", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_path}.png", dpi=500, bbox_inches="tight")
    plt.close()


def plot_dashboard_comparison(personalized_correct, personalized_total, baseline_correct, baseline_total,
                               most_broadly_effective_drug, test_df_with_predictions, save_path):
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4])

    ax1 = fig.add_subplot(gs[0])
    panels = [
        ("Personalized\nRanking", personalized_correct, personalized_total, "#DD8452"),
        (f"Non-Personalized\nBaseline\n({most_broadly_effective_drug})", baseline_correct, baseline_total, "#8C8C8C"),
    ]
    for i, (label, correct, total, color) in enumerate(panels):
        hit_rate = correct / total
        center = (i * 1.3, 0)
        ax1.pie([hit_rate, 1 - hit_rate], radius=0.55, center=center, colors=[color, "#eeeeee"],
                startangle=90, counterclock=False, wedgeprops=dict(width=0.32, edgecolor="white", linewidth=2))
        ax1.text(center[0], center[1], f"{hit_rate:.0%}", ha="center", va="center", fontsize=17, fontweight="bold")
        ax1.text(center[0], center[1] - 0.85, f"{label}\n({correct}/{total})", ha="center", va="center", fontsize=11)

    ax1.set_xlim(-0.8, 2.1)
    ax1.set_ylim(-1.4, 0.8)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("Top-1 Hit Rate", fontsize=14, fontweight="bold")

    ax2 = fig.add_subplot(gs[1])
    patient_level = []
    for _, group in test_df_with_predictions.groupby("SANGER_MODEL_ID"):
        if len(group) < 2:
            continue
        top_row = group.loc[group["sensitivity_probability"].idxmax()]
        patient_level.append({"probability": top_row["sensitivity_probability"], "correct": bool(top_row["actual_sensitive"] == 1)})

    patient_df = pd.DataFrame(patient_level).sort_values("probability").reset_index(drop=True)
    dot_colors = patient_df["correct"].map({True: "#DD8452", False: "#4C72B0"})

    ax2.scatter(patient_df["probability"], range(len(patient_df)), c=dot_colors, s=28,
                alpha=0.85, edgecolor="white", linewidth=0.3)
    ax2.axvline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Predicted Sensitivity Probability (Top-1 Drug)", fontsize=12)
    ax2.set_ylabel("Patients (sorted by probability)", fontsize=12)
    ax2.set_title("Per-Patient Top Recommendation", fontsize=14, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#DD8452", markersize=9, label="Correct"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0", markersize=9, label="Incorrect"),
    ]
    ax2.legend(handles=legend_elements, fontsize=10, frameon=False, loc="upper left")

    gap_pp = (personalized_correct / personalized_total - baseline_correct / baseline_total) * 100
    plt.suptitle(f"Clinical Decision Dashboard — Gap: {gap_pp:.1f} pp", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_path}.png", dpi=500, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Accuracy diagnostics (Sections 4.3, 4.4)
# ---------------------------------------------------------------------------


def plot_parity(y_true, y_pred, r2, save_path):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    xy = np.vstack([y_true, y_pred])
    density = gaussian_kde(xy)(xy)
    sort_idx = density.argsort()
    y_true_sorted, y_pred_sorted, density_sorted = y_true[sort_idx], y_pred[sort_idx], density[sort_idx]

    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(y_true_sorted, y_pred_sorted, c=density_sorted, cmap="Blues", s=14, alpha=0.75, edgecolor="none")

    lims = [min(y_true.min(), y_pred.min()) - 0.5, max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, color="#C44E52", linewidth=1.8, linestyle="--", label=r"Perfect prediction ($y=\hat{y}$)")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual LN_IC50", fontsize=12)
    ax.set_ylabel("Predicted LN_IC50", fontsize=12)
    ax.set_title(f"Predicted vs. Actual Response ($R^2$ = {r2:.4f})", fontsize=13.5, fontweight="bold")
    ax.legend(fontsize=9.5, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Point density", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_path}.png", dpi=500, bbox_inches="tight")
    plt.close()


def plot_error_heatmap(heatmap_pivot, save_path):
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(heatmap_pivot.values, cmap="Oranges", aspect="auto")

    ax.set_xticks(np.arange(len(heatmap_pivot.columns)))
    ax.set_xticklabels(heatmap_pivot.columns, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(heatmap_pivot.index)))
    ax.set_yticklabels(heatmap_pivot.index, fontsize=8)

    ax.set_xlabel("Drug", fontsize=12)
    ax.set_ylabel("Cancer Type", fontsize=12)
    ax.set_title("Mean Absolute Error by Cancer Type and Drug", fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean Absolute Error", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_path}.png", dpi=500, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Learned drug-embedding projection (2D PCA)
# ---------------------------------------------------------------------------


def extract_drug_embeddings(model, drug_graphs, device):
    import torch

    model.eval()
    embeddings = {}
    with torch.no_grad():
        for drug_name, graph in drug_graphs.items():
            x = graph.x.to(device)
            edge_bond = graph.edge_index_bond.to(device)
            edge_ring = graph.edge_index_ring.to(device)
            edge_fg = graph.edge_index_fg.to(device)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

            embedding = model.drug_encoder(x, edge_bond, edge_ring, edge_fg, batch)
            embeddings[drug_name] = embedding.cpu().numpy().flatten()
    return embeddings


def compute_pca_2d(X):
    X_centered = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:2].T


def plot_pca_drug_embeddings(embedding_matrix, pathway_grouped, save_path):
    coords = compute_pca_2d(embedding_matrix)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    palette = ["#4C72B0", "#DD8452", "#55A868", "#8172B2", "#937860", "#DA8BC3", "#CCB974", "#64B5CD", "#8C8C8C"]
    unique_groups = sorted(set(pathway_grouped), key=lambda g: (g == "Other", g))

    for i, group in enumerate(unique_groups):
        mask = [g == group for g in pathway_grouped]
        color = palette[i % len(palette)] if group != "Other" else "#CCCCCC"
        ax.scatter(coords[mask, 0], coords[mask, 1], label=group, color=color,
                   s=45, alpha=0.8, edgecolor="white", linewidth=0.4)

    ax.set_xlabel("PC-1", fontsize=12)
    ax.set_ylabel("PC-2", fontsize=12)
    ax.set_title("PCA Projection of Learned Drug Embeddings", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5), title="Pathway", title_fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_path}.png", dpi=500, bbox_inches="tight")
    plt.close()
