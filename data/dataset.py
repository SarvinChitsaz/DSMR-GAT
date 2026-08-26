import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class DrugResponseDataset(Dataset):
    def __init__(self, dataframe, drug_graphs, cell_line_expression):
        self.data = dataframe.reset_index(drop=True)
        self.drug_graphs = drug_graphs
        self.cell_line_expression = cell_line_expression

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        drug_graph = self.drug_graphs[row["DRUG_NAME"]]
        expression = torch.tensor(
            self.cell_line_expression.loc[row["SANGER_MODEL_ID"]].values,
            dtype=torch.float,
        )
        label = torch.tensor(row["LN_IC50"], dtype=torch.float)
        return drug_graph, expression, label


def collate_fn(batch):
    graphs = [item[0] for item in batch]
    expressions = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    drug_batch = Batch.from_data_list(graphs)
    return drug_batch, expressions, labels


def split_by_cell_line(final_dataset_with_smiles, train_cell_lines, val_cell_lines, test_cell_lines):
    train_df = final_dataset_with_smiles[final_dataset_with_smiles["SANGER_MODEL_ID"].isin(train_cell_lines)]
    val_df = final_dataset_with_smiles[final_dataset_with_smiles["SANGER_MODEL_ID"].isin(val_cell_lines)]
    test_df = final_dataset_with_smiles[final_dataset_with_smiles["SANGER_MODEL_ID"].isin(test_cell_lines)]
    return train_df, val_df, test_df
