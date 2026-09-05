import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from sklearn.metrics import r2_score
from configs.config import (LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, EARLY_STOPPING_PATIENCE, CHECKPOINT_DIR, DEVICE)
from models.deeprelcdr import DrugResponseModelDeepRelCDR, EarlyStopping
from models.ablation_models import DrugResponseModel_TwoRelation, DrugResponseModel_SingleRelation


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_deeprelcdr(train_loader, val_loader, atom_feature_dim=20, gene_feature_dim=1000,
                   hidden_dim=128, epochs=MAX_EPOCHS, seed=None):
    """Trains the full three-relation DeepRelCDR model."""
    if seed is not None:
        set_all_seeds(seed)

    model = DrugResponseModelDeepRelCDR(
        atom_feature_dim=atom_feature_dim, gene_feature_dim=gene_feature_dim, hidden_dim=hidden_dim
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        running_train_loss = 0
        for drug_batch, expressions, labels in train_loader:
            drug_batch = drug_batch.to(DEVICE)
            expressions = expressions.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(
                drug_batch.x,
                drug_batch.edge_index_bond,
                drug_batch.edge_index_ring,
                drug_batch.edge_index_fg,
                drug_batch.batch,
                expressions,
            )
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * len(labels)
        train_loss = running_train_loss / len(train_loader.dataset)

        model.eval()
        running_val_loss = 0
        with torch.no_grad():
            for drug_batch, expressions, labels in val_loader:
                drug_batch = drug_batch.to(DEVICE)
                expressions = expressions.to(DEVICE)
                labels = labels.to(DEVICE)
                predictions = model(
                    drug_batch.x,
                    drug_batch.edge_index_bond,
                    drug_batch.edge_index_ring,
                    drug_batch.edge_index_fg,
                    drug_batch.batch,
                    expressions,
                )
                loss = criterion(predictions, labels)
                running_val_loss += loss.item() * len(labels)
        val_loss = running_val_loss / len(val_loader.dataset)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")

        early_stopping.step(val_loss, model)
        if early_stopping.stop:
            print("Early stopping triggered.")
            break

    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)
        print(f"Restored best model weights (Val Loss: {early_stopping.best_loss:.4f})")

    return model, optimizer


def train_two_relation_ablation(relation_a_name, relation_b_name, train_loader, val_loader,
                                 atom_feature_dim=20, gene_feature_dim=1000, hidden_dim=128,
                                 epochs=MAX_EPOCHS, seed=None):
    """Trains the Bond+Ring or Bond+FG ablation variant."""
    if seed is not None:
        set_all_seeds(seed)

    model = DrugResponseModel_TwoRelation(
        atom_feature_dim=atom_feature_dim, gene_feature_dim=gene_feature_dim, hidden_dim=hidden_dim
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        running_train_loss = 0
        for drug_batch, expressions, labels in train_loader:
            drug_batch = drug_batch.to(DEVICE)
            expressions = expressions.to(DEVICE)
            labels = labels.to(DEVICE)

            edge_a = getattr(drug_batch, f"edge_index_{relation_a_name}")
            edge_b = getattr(drug_batch, f"edge_index_{relation_b_name}")

            optimizer.zero_grad()
            predictions = model(drug_batch.x, edge_a, edge_b, drug_batch.batch, expressions)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * len(labels)
        train_loss = running_train_loss / len(train_loader.dataset)

        model.eval()
        running_val_loss = 0
        with torch.no_grad():
            for drug_batch, expressions, labels in val_loader:
                drug_batch = drug_batch.to(DEVICE)
                expressions = expressions.to(DEVICE)
                labels = labels.to(DEVICE)

                edge_a = getattr(drug_batch, f"edge_index_{relation_a_name}")
                edge_b = getattr(drug_batch, f"edge_index_{relation_b_name}")

                predictions = model(drug_batch.x, edge_a, edge_b, drug_batch.batch, expressions)
                loss = criterion(predictions, labels)
                running_val_loss += loss.item() * len(labels)
        val_loss = running_val_loss / len(val_loader.dataset)

        elapsed = time.time() - start_time
        print(f"[{relation_a_name}+{relation_b_name}] Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")

        early_stopping.step(val_loss, model)
        if early_stopping.stop:
            print("Early stopping triggered.")
            break

    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)

    return model, optimizer


def train_single_relation_baseline(train_loader, val_loader, atom_feature_dim=20, gene_feature_dim=1000, hidden_dim=128,
                                    epochs=MAX_EPOCHS, seed=None):
    if seed is not None:
        set_all_seeds(seed)

    model = DrugResponseModel_SingleRelation(
        atom_feature_dim=atom_feature_dim, gene_feature_dim=gene_feature_dim, hidden_dim=hidden_dim
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for drug_batch, expressions, labels in train_loader:
            drug_batch = drug_batch.to(DEVICE)
            expressions = expressions.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(drug_batch.x, drug_batch.edge_index_bond, drug_batch.batch, expressions)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        running_val_loss = 0
        with torch.no_grad():
            for drug_batch, expressions, labels in val_loader:
                drug_batch = drug_batch.to(DEVICE)
                expressions = expressions.to(DEVICE)
                labels = labels.to(DEVICE)
                predictions = model(drug_batch.x, drug_batch.edge_index_bond, drug_batch.batch, expressions)
                loss = criterion(predictions, labels)
                running_val_loss += loss.item() * len(labels)
        val_loss = running_val_loss / len(val_loader.dataset)

        print(f"[single-relation] Epoch {epoch + 1}/{epochs} | Val Loss: {val_loss:.4f}")

        early_stopping.step(val_loss, model)
        if early_stopping.stop:
            print("Early stopping triggered.")
            break

    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)

    return model, optimizer


def save_checkpoint(model, optimizer, test_r2, filename, **extra_fields):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_r2": test_r2,
        **extra_fields,
    }
    torch.save(checkpoint, f"{CHECKPOINT_DIR}/{filename}")
    print(f"Saved checkpoint: {filename} (test_r2={test_r2:.4f})")
