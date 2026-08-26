import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, BatchNorm
from models.dsmr_gat import RelationDepthwiseAttention, CellLineEncoder, SimpleFusion


class DSMRGATLayerTwoRelation(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.attn_a = RelationDepthwiseAttention(in_dim, dropout=dropout)
        self.attn_b = RelationDepthwiseAttention(in_dim, dropout=dropout)
        self.pointwise = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, edge_index_a, edge_index_b):
        z_a = self.attn_a(x, edge_index_a)
        z_b = self.attn_b(x, edge_index_b)
        z = torch.cat([z_a, z_b], dim=1)
        return self.pointwise(z)


class DrugEncoderDSMR_TwoRelation(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.layer1 = DSMRGATLayerTwoRelation(input_dim, hidden_dim, dropout=dropout)
        self.bn1 = BatchNorm(hidden_dim)
        self.layer2 = DSMRGATLayerTwoRelation(hidden_dim, hidden_dim, dropout=dropout)
        self.bn2 = BatchNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index_a, edge_index_b, batch):
        x = self.layer1(x, edge_index_a, edge_index_b)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.dropout(x)
        x = self.layer2(x, edge_index_a, edge_index_b)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = global_mean_pool(x, batch)
        return x


class DrugResponseModel_TwoRelation(nn.Module):
    def __init__(self, atom_feature_dim, gene_feature_dim, hidden_dim=128):
        super().__init__()
        self.drug_encoder = DrugEncoderDSMR_TwoRelation(atom_feature_dim, hidden_dim)
        self.cell_encoder = CellLineEncoder(gene_feature_dim, hidden_dim)
        self.fusion = SimpleFusion(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, drug_x, edge_index_a, edge_index_b, drug_batch, cell_expression):
        drug_vec = self.drug_encoder(drug_x, edge_index_a, edge_index_b, drug_batch)
        cell_vec = self.cell_encoder(cell_expression)
        fused = self.fusion(drug_vec, cell_vec)
        prediction = self.output_layer(fused)
        return prediction.squeeze(-1)


class SingleRelationGATAttention(nn.Module):
    def __init__(self, in_dim, out_dim, negative_slope=0.2, dropout=0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * out_dim, 1, bias=False)
        self.negative_slope = negative_slope
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        from torch_geometric.utils import add_self_loops, softmax

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.W(x)

        src, dst = edge_index
        e = self.attn(torch.cat([x[dst], x[src]], dim=-1))
        e = F.leaky_relu(e, negative_slope=self.negative_slope)
        alpha = softmax(e, dst, num_nodes=x.size(0))
        alpha = self.dropout(alpha)

        out = torch.zeros_like(x)
        out.index_add_(0, dst, alpha * x[src])
        return out


class DrugEncoderSingleRelation(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.layer1 = SingleRelationGATAttention(input_dim, hidden_dim, dropout=dropout)
        self.bn1 = BatchNorm(hidden_dim)
        self.layer2 = SingleRelationGATAttention(hidden_dim, hidden_dim, dropout=dropout)
        self.bn2 = BatchNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index_bond, batch):
        x = self.layer1(x, edge_index_bond)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.dropout(x)
        x = self.layer2(x, edge_index_bond)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = global_mean_pool(x, batch)
        return x


class DrugResponseModel_SingleRelation(nn.Module):
    def __init__(self, atom_feature_dim, gene_feature_dim, hidden_dim=128):
        super().__init__()
        self.drug_encoder = DrugEncoderSingleRelation(atom_feature_dim, hidden_dim)
        self.cell_encoder = CellLineEncoder(gene_feature_dim, hidden_dim)
        self.fusion = SimpleFusion(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, drug_x, edge_index_bond, drug_batch, cell_expression):
        drug_vec = self.drug_encoder(drug_x, edge_index_bond, drug_batch)
        cell_vec = self.cell_encoder(cell_expression)
        fused = self.fusion(drug_vec, cell_vec)
        prediction = self.output_layer(fused)
        return prediction.squeeze(-1)
