import torch
from sklearn.metrics import r2_score
from configs.config import DEVICE


def evaluate_deeprelcdr(model, test_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for drug_batch, expressions, labels in test_loader:
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
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    test_r2 = r2_score(all_labels, all_predictions)
    return test_r2, all_predictions, all_labels


def evaluate_two_relation(model, test_loader, relation_a_name, relation_b_name):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for drug_batch, expressions, labels in test_loader:
            drug_batch = drug_batch.to(DEVICE)
            expressions = expressions.to(DEVICE)

            edge_a = getattr(drug_batch, f"edge_index_{relation_a_name}")
            edge_b = getattr(drug_batch, f"edge_index_{relation_b_name}")

            predictions = model(drug_batch.x, edge_a, edge_b, drug_batch.batch, expressions)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    test_r2 = r2_score(all_labels, all_predictions)
    return test_r2, all_predictions, all_labels


def evaluate_single_relation(model, test_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for drug_batch, expressions, labels in test_loader:
            drug_batch = drug_batch.to(DEVICE)
            expressions = expressions.to(DEVICE)

            predictions = model(drug_batch.x, drug_batch.edge_index_bond, drug_batch.batch, expressions)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    test_r2 = r2_score(all_labels, all_predictions)
    return test_r2, all_predictions, all_labels


def load_deeprelcdr_checkpoint(checkpoint_path, model_class, atom_feature_dim=20,
                             gene_feature_dim=1000, hidden_dim=128):
    model = model_class(
        atom_feature_dim=atom_feature_dim, gene_feature_dim=gene_feature_dim, hidden_dim=hidden_dim
    ).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
