import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, BatchNorm
from torch_geometric.utils import softmax, add_self_loops


class RelationDepthwiseAttention(MessagePassing):
    def __init__(self, in_dim, negative_slope=0.2, dropout=0.2):
        super().__init__(aggr="add", node_dim=0)
        self.attn = nn.Linear(2 * in_dim, 1, bias=False)
        self.negative_slope = negative_slope
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, index, ptr, size_i):
        e = self.attn(torch.cat([x_i, x_j], dim=-1))
        e = F.leaky_relu(e, negative_slope=self.negative_slope)
        alpha = softmax(e, index, ptr, size_i)
        alpha = self.dropout(alpha)
        return alpha * x_j


class DSMRGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.attn_bond = RelationDepthwiseAttention(in_dim, dropout=dropout)
        self.attn_ring = RelationDepthwiseAttention(in_dim, dropout=dropout)
        self.attn_fg = RelationDepthwiseAttention(in_dim, dropout=dropout)
        self.pointwise = nn.Linear(in_dim * 3, out_dim)

    def forward(self, x, edge_index_bond, edge_index_ring, edge_index_fg):
        z_bond = self.attn_bond(x, edge_index_bond)
        z_ring = self.attn_ring(x, edge_index_ring)
        z_fg = self.attn_fg(x, edge_index_fg)
        z = torch.cat([z_bond, z_ring, z_fg], dim=1)
        return self.pointwise(z)


class DrugEncoderDSMRGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.layer1 = DSMRGATLayer(input_dim, hidden_dim, dropout=dropout)
        self.bn1 = BatchNorm(hidden_dim)
        self.layer2 = DSMRGATLayer(hidden_dim, hidden_dim, dropout=dropout)
        self.bn2 = BatchNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index_bond, edge_index_ring, edge_index_fg, batch):
        x = self.layer1(x, edge_index_bond, edge_index_ring, edge_index_fg)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.dropout(x)
        x = self.layer2(x, edge_index_bond, edge_index_ring, edge_index_fg)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = global_mean_pool(x, batch)
        return x


class CellLineEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        return x


class SimpleFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, drug_vec, cell_vec):
        combined = torch.cat([drug_vec, cell_vec], dim=1)
        return F.relu(self.fc(combined))


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False
        self.best_state_dict = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


class DrugResponseModelDSMRGAT(nn.Module):
    def __init__(self, atom_feature_dim, gene_feature_dim, hidden_dim=128):
        super().__init__()
        self.drug_encoder = DrugEncoderDSMRGAT(atom_feature_dim, hidden_dim)
        self.cell_encoder = CellLineEncoder(gene_feature_dim, hidden_dim)
        self.fusion = SimpleFusion(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, drug_x, edge_index_bond, edge_index_ring, edge_index_fg, drug_batch, cell_expression):
        drug_vec = self.drug_encoder(drug_x, edge_index_bond, edge_index_ring, edge_index_fg, drug_batch)
        cell_vec = self.cell_encoder(cell_expression)
        fused = self.fusion(drug_vec, cell_vec)
        prediction = self.output_layer(fused)
        return prediction.squeeze(-1)
