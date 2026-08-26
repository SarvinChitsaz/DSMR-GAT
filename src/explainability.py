import re
import json
import torch
import numpy as np
import matplotlib.cm as cm
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from configs.config import PROCESSED_DIR, DEVICE

# ---------------------------------------------------------------------------
# Gene-level attribution (SLFN11 biological validation, Section 4.7)
# ---------------------------------------------------------------------------


def compute_gene_attribution(model, graph, cell_expression_tensor):
    model.eval()

    drug_x = graph.x.to(DEVICE)
    edge_bond = graph.edge_index_bond.to(DEVICE)
    edge_ring = graph.edge_index_ring.to(DEVICE)
    edge_fg = graph.edge_index_fg.to(DEVICE)
    drug_batch = torch.zeros(drug_x.size(0), dtype=torch.long, device=DEVICE)

    cell_expression_tensor = cell_expression_tensor.clone().detach().to(DEVICE)
    cell_expression_tensor.requires_grad_(True)

    prediction = model(drug_x, edge_bond, edge_ring, edge_fg, drug_batch, cell_expression_tensor.unsqueeze(0))
    prediction.backward()

    return cell_expression_tensor.grad.abs().cpu().numpy()


def compute_slfn11_attribution(model, drug_graphs, drug_to_smiles, camptothecin_test_rows,
                                cell_line_expression_normalized, gene_names, top_n_cancer_types=3, top_k_genes=5):
    graph = drug_graphs["Camptothecin"]
    cancer_types_available = camptothecin_test_rows["CANCER_TYPE"].value_counts()
    selected_cancer_types = cancer_types_available.head(top_n_cancer_types).index.tolist()

    results = {}
    for cancer_type in selected_cancer_types:
        subset = camptothecin_test_rows[camptothecin_test_rows["CANCER_TYPE"] == cancer_type]

        gradients = []
        for _, row in subset.iterrows():
            cell_expr = torch.tensor(
                cell_line_expression_normalized.loc[row["SANGER_MODEL_ID"]].values, dtype=torch.float
            )
            gradients.append(compute_gene_attribution(model, graph, cell_expr))

        mean_gradient = np.mean(gradients, axis=0)
        top_gene_indices = np.argsort(mean_gradient)[::-1][:top_k_genes]

        results[cancer_type] = {
            "n_samples": len(subset),
            "top_5_genes": [(gene_names[idx], float(mean_gradient[idx])) for idx in top_gene_indices],
        }

        print(f"\nCamptothecin — {cancer_type} (n={len(subset)}):")
        for rank, idx in enumerate(top_gene_indices, start=1):
            marker = "  <-- SLFN11" if gene_names[idx] == "SLFN11" else ""
            print(f"  {rank}. {gene_names[idx]}: {mean_gradient[idx]:.4f}{marker}")

    return results


def save_slfn11_attribution(results):
    with open(f"{PROCESSED_DIR}/slfn11_attribution_summary.json", "w") as f:
        json.dump(results, f, indent=2)


def load_slfn11_attribution():
    with open(f"{PROCESSED_DIR}/slfn11_attribution_summary.json", "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Atom-level attribution (ring-relation interpretability, Section 4.8)
# ---------------------------------------------------------------------------


def get_atom_output_magnitude(model, graph, relation_name="ring"):
    x = graph.x.to(DEVICE)
    edge_index = getattr(graph, f"edge_index_{relation_name}").to(DEVICE)

    layer1 = model.drug_encoder.layer1
    attn_module = getattr(layer1, f"attn_{relation_name}")

    with torch.no_grad():
        z = attn_module(x, edge_index)

    return z.norm(dim=1).cpu().numpy()


def mol_to_svg_fragment_colored(mol, atom_colors, size):
    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    drawer.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightBonds=[],
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    inner = re.search(r"<svg.*?>(.*)</svg>", svg, re.DOTALL).group(1)
    return inner


def build_node_importance_svg(model, drug_graphs, drug_to_smiles, example_drugs,
                               relation_name="ring", panel_size=320, gap=25,
                               title_height=28, main_title_height=40):
    num_examples = len(example_drugs)
    canvas_width = panel_size * num_examples + gap * (num_examples - 1)
    canvas_height = main_title_height + title_height + panel_size

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" '
        f'viewBox="0 0 {canvas_width} {canvas_height}">',
        f'<rect width="{canvas_width}" height="{canvas_height}" fill="white"/>',
    ]

    main_title = f"Node Importance Overlay ({relation_name.capitalize()} Relation) — Output Magnitude"
    svg_parts.append(
        f'<text x="{canvas_width / 2}" y="{main_title_height * 0.65}" font-family="Helvetica" '
        f'font-size="20" font-weight="bold" text-anchor="middle">{main_title}</text>'
    )

    for i, drug_name in enumerate(example_drugs):
        smiles = drug_to_smiles[drug_name]
        mol = Chem.MolFromSmiles(smiles)

        atom_magnitude = get_atom_output_magnitude(model, drug_graphs[drug_name], relation_name=relation_name)
        norm_scores = (atom_magnitude - atom_magnitude.min()) / (atom_magnitude.max() - atom_magnitude.min() + 1e-8)
        cmap = cm.get_cmap("Oranges")
        atom_colors = {j: cmap(norm_scores[j])[:3] for j in range(mol.GetNumAtoms())}

        x_offset = i * (panel_size + gap)
        y_offset = main_title_height

        svg_parts.append(
            f'<text x="{x_offset + panel_size / 2}" y="{y_offset + title_height * 0.7}" '
            f'font-family="Helvetica" font-size="16" text-anchor="middle">{drug_name}</text>'
        )
        mol_svg_inner = mol_to_svg_fragment_colored(mol, atom_colors, panel_size)
        svg_parts.append(f'<g transform="translate({x_offset},{y_offset + title_height})">{mol_svg_inner}</g>')

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)
