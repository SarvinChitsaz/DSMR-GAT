import inspect
import pickle
import torch
from rdkit import Chem
from rdkit.Chem import Fragments
from torch_geometric.data import Data
from configs.config import PROCESSED_DIR

POSSIBLE_ATOMS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "H"]
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]

# Ring-scaffold and metabolic-site Fragments patterns are excluded because they
# substantially overlap with the ring relation built from RDKit's own SSSR.
RING_SCAFFOLD_PATTERNS = {
    "fr_benzene", "fr_furan", "fr_thiophene", "fr_pyridine",
    "fr_imidazole", "fr_thiazole", "fr_oxazole", "fr_tetrazole",
    "fr_piperdine", "fr_piperzine", "fr_morpholine",
    "fr_benzodiazepine", "fr_barbitur", "fr_dihydropyridine",
    "fr_Nhpyrrole", "fr_bicyclic", "fr_epoxide",
}
METABOLISM_SITE_PATTERNS = {
    "fr_Ndealkylation1", "fr_Ndealkylation2", "fr_allylic_oxid", "fr_para_hydroxylation",
}


def one_hot_encode(value, choices):
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    return encoding


def get_atom_features(atom):
    features = one_hot_encode(atom.GetSymbol(), POSSIBLE_ATOMS)
    features.append(atom.GetDegree())
    features.append(atom.GetFormalCharge())
    features.append(int(atom.GetIsAromatic()))
    features.append(atom.GetTotalNumHs())
    features.append(int(atom.IsInRing()))
    features.extend(one_hot_encode(str(atom.GetHybridization()), HYBRIDIZATIONS))
    return features


def get_bond_edges(mol):
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])
    return edges


def get_ring_edges(mol):
    ring_info = mol.GetRingInfo()
    edges = set()
    for ring in ring_info.AtomRings():
        for i in ring:
            for j in ring:
                if i != j:
                    edges.add((i, j))
    return list(edges)


def build_functional_group_patterns():
    excluded_patterns = RING_SCAFFOLD_PATTERNS | METABOLISM_SITE_PATTERNS
    fragment_function_names = [name for name in dir(Fragments) if name.startswith("fr_")]
    fragment_function_names = [name for name in fragment_function_names if name not in excluded_patterns]

    patterns = {}
    for name in fragment_function_names:
        func = getattr(Fragments, name)
        sig = inspect.signature(func)
        pattern = sig.parameters["pattern"].default
        if pattern is not None:
            patterns[name] = pattern

    # Stricter fr_ether: excludes ester/lactone oxygens.
    patterns["fr_ether"] = Chem.MolFromSmarts("[OD2;!$(OC=O)]([#6])[#6]")
    # Stricter fr_amide: excludes carbamate/urea nitrogens.
    patterns["fr_amide"] = Chem.MolFromSmarts("[NX3][CX3](=[OX1])[#6]")
    # Boronic acid, not covered by the default RDKit Fragments library.
    patterns["fr_boronic_acid_custom"] = Chem.MolFromSmarts("[#5](-[OX2])-[OX2]")

    return patterns


def get_functional_group_edges(mol, functional_group_patterns):
    edges = set()
    for pattern in functional_group_patterns.values():
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            for i in match:
                for j in match:
                    if i != j:
                        edges.add((i, j))
    return list(edges)


def edges_to_tensor(edge_list):
    if len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


def smiles_to_multi_relation_graph(smiles: str, functional_group_patterns):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float)

    bond_edges = get_bond_edges(mol)
    ring_edges = get_ring_edges(mol)
    fg_edges = get_functional_group_edges(mol, functional_group_patterns)

    return Data(
        x=x,
        edge_index_bond=edges_to_tensor(bond_edges),
        edge_index_ring=edges_to_tensor(ring_edges),
        edge_index_fg=edges_to_tensor(fg_edges),
        num_nodes=x.size(0),
    )


def build_all_drug_graphs(drug_to_smiles):
    functional_group_patterns = build_functional_group_patterns()
    drug_graphs = {}
    failed_drugs = []

    for drug_name, smiles in drug_to_smiles.items():
        if smiles is None:
            continue
        graph = smiles_to_multi_relation_graph(smiles, functional_group_patterns)
        if graph is not None:
            drug_graphs[drug_name] = graph
        else:
            failed_drugs.append(drug_name)

    return drug_graphs, failed_drugs


def save_drug_graphs(drug_graphs):
    with open(f"{PROCESSED_DIR}/drug_graphs_multirelation.pkl", "wb") as f:
        pickle.dump(drug_graphs, f)


def load_drug_graphs():
    with open(f"{PROCESSED_DIR}/drug_graphs_multirelation.pkl", "rb") as f:
        return pickle.load(f)
