import torch
import os, gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn.models import EdgeCNN
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import polars as pl
import re
from tqdm import tqdm

class DataConverter:
    # This class is used to convert the data from csv to parquet.
    @staticmethod
    def to_parquet(csv_file, parquet_file):
        dummy_df = pl.scan_csv(csv_file)

        new_schema = {}
        for key, value in dummy_df.schema.items():
            if key.startswith("reactivity"):
                new_schema[key] = pl.Float32
            else:
                new_schema[key] = value

        df = pl.scan_csv(csv_file, schema=new_schema)
        
        df.sink_parquet(
                parquet_file,
                compression='uncompressed',
                row_group_size=10,
        )

class SimpleGraphDataset():
    def __init__(self, parquet_name, edge_distance=5, process_type = 'train', root=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__()
        # Set csv name
        self.parquet_name = parquet_name
        # Set edge distance
        self.edge_distance = edge_distance
        # Initialize one hot encoder
        self.node_encoder = OneHotEncoder(sparse_output=False, max_categories=5)
        # For one-hot encoder to possible values
        self.node_encoder.fit(np.array(['A', 'G', 'U', 'C']).reshape(-1, 1))
        # Load dataframe
        self.df = pl.read_parquet(self.parquet_name)
        self.sequence_df = self.df.select("sequence")
        self.process_type = process_type

        if self.process_type == 'train':
            self.df = self.df.filter(pl.col("SN_filter") == 1.0)
            # Get reactivity columns names
            reactivity_match = re.compile('(reactivity_[0-9])')
            reactivity_names = [col for col in self.df.columns if reactivity_match.match(col)]
            self.reactivity_df = self.df.select(reactivity_names) 
        elif self.process_type == 'test':
            self.id_min_df = self.df.select("id_min")
        else:
            raise ValueError("process_type must be either 'train' or 'test'")

    def nearest_adjacency(self, sequence_length, n=2, loops=True):
        base = np.arange(sequence_length)
        connections = []
        for i in range(-n, n + 1):
            if i == 0 and not loops:
                continue
            elif i == 0 and loops:
                stack = np.vstack([base, base])
                connections.append(stack)
                continue
            neighbours = base.take(range(i, sequence_length + i), mode='wrap')
            stack = np.vstack([base, neighbours])
            if i < 0:
                connections.append(stack[:, -i:])
            elif i > 0:
                connections.append(stack[:, :-i])
        return np.hstack(connections)

    def parse_row(self, idx):
        # Parse row
        if self.process_type == 'train': # For training
            sequence_row = self.sequence_df.row(idx)
            reactivity_row = self.reactivity_df.row(idx)
            sequence = np.array(list(sequence_row[0])).reshape(-1, 1)
            encoded_sequence = self.node_encoder.transform(sequence)
            sequence_length = len(sequence)
            edges_np = self.nearest_adjacency(sequence_length, n=self.edge_distance, loops=False)
            edge_index = torch.tensor(edges_np, dtype=torch.long)
            reactivity = np.array(reactivity_row, dtype=np.float32)[0:sequence_length]
            valid_mask = np.argwhere(~np.isnan(reactivity)).reshape(-1)
            torch_valid_mask = torch.tensor(valid_mask, dtype=torch.long)
            reactivity = np.nan_to_num(reactivity, copy=False, nan=0.0)
            node_features = torch.Tensor(encoded_sequence)
            targets = torch.Tensor(reactivity)
            return Data(x=node_features, edge_index=edge_index, y=targets, valid_mask=torch_valid_mask)
        
        elif self.process_type == 'test': # For inference
            sequence_row = self.sequence_df.row(idx)  
            id_min = self.id_min_df.row(idx)[0]
            sequence = np.array(list(sequence_row[0])).reshape(-1, 1)
            encoded_sequence = self.node_encoder.transform(sequence)
            sequence_length = len(sequence)
            edges_np = self.nearest_adjacency(sequence_length, n=self.edge_distance, loops=False)
            edge_index = torch.tensor(edges_np, dtype=torch.long)
            node_features = torch.Tensor(encoded_sequence)
            ids = torch.arange(id_min, id_min+sequence_length, 1)
            return Data(x=node_features, edge_index=edge_index, ids=ids)

    def len(self):
        return len(self.df)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.parse_row(idx)

    def get(self, idx):
        return self.parse_row(idx)

class RNAGraphModel:
    
    def __init__(self, train_parquet_file, test_parquet_file, edge_distance=4):
        self.train_parquet_file = train_parquet_file
        self.test_parquet_file = test_parquet_file
        self.edge_distance = edge_distance
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EdgeCNN(in_channels=4, hidden_channels=128, num_layers=4, out_channels=1).to(self.device)
        self.inference_ids = None # Will be set after inference
        self.inference_preds = None # Will be set after inference
        self.SimpleGraphDataset = SimpleGraphDataset

    def loss_fn(self, output, target):
        clipped_target = torch.clip(target, min=0, max=1)
        return F.mse_loss(output, clipped_target, reduction='mean')

    def mae_fn(self, output, target):
        clipped_target = torch.clip(target, min=0, max=1)
        return F.l1_loss(output, clipped_target, reduction='mean')


    def train_model(self, n_epochs=10):
        # Initialize dataset and dataloaders
        full_train_dataset = self.SimpleGraphDataset(parquet_name=self.train_parquet_file, edge_distance=self.edge_distance)
        
        generator1 = torch.Generator().manual_seed(42)
        len_full_train_dataset = full_train_dataset.len()  # Assuming you've defined __len__ in your Dataset class
        
        len_train_dataset = int(0.9 * len_full_train_dataset)
        len_val_dataset = len_full_train_dataset - len_train_dataset

        train_dataset, val_dataset = random_split(full_train_dataset, [len_train_dataset, len_val_dataset], generator=generator1)
        
        train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
        val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)

        # Initialize optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003, weight_decay=5e-4)

        best_val_loss = float('inf')

        for epoch in range(n_epochs):
            self.model.train()
            # Training loop
            for batch in tqdm(train_dataloader):
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index)
                out = torch.squeeze(out)
                loss = self.mae_fn(out[batch.valid_mask], batch.y[batch.valid_mask])
                #loss = self.loss_fn(out[batch.valid_mask], batch.y[batch.valid_mask])
                loss.backward()
                optimizer.step()

            # Validation loop
            self.model.eval()
            val_losses = []
            val_mae = []
            for batch in tqdm(val_dataloader):
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index)
                out = torch.squeeze(out)
                loss = self.loss_fn(out[batch.valid_mask], batch.y[batch.valid_mask])
                mae = self.mae_fn(out[batch.valid_mask], batch.y[batch.valid_mask])
                val_losses.append(loss.detach().cpu().numpy())

            mean_val_loss = np.mean(val_losses)
            print(f"Epoch {epoch} val loss: ", np.mean(mean_val_loss))
            print(f"Epoch {epoch} val mae: ", np.mean(mae))
            
            # Save model if validation loss is improved
            if mean_val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss} to {mean_val_loss}. Saving model.")
                best_val_loss = mean_val_loss
                self.save_model("best_model.pth")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def inference(self):
        infer_dataset = self.SimpleGraphDataset(parquet_name=self.test_parquet_file, edge_distance=self.edge_distance, process_type = 'test')
        infer_dataloader = DataLoader(infer_dataset, batch_size=128, shuffle=False, num_workers=4)
        self.model.eval().to(self.device)
        ids = np.empty(shape=(0, 1), dtype=int)
        preds = np.empty(shape=(0, 1), dtype=np.float32)
        for batch in tqdm(infer_dataloader):
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index).detach().cpu().numpy()
            rounded_out = np.round(out, 4)  # Rounding to 4 decimal places
            ids = np.append(ids, batch.ids.detach().cpu().numpy())
            preds = np.append(preds, rounded_out)  # Using rounded_out instead of out
        self.inference_ids = ids
        self.inference_preds = preds

    def save_submission(self, submission_file_path):
        submission_df = pl.DataFrame({"id": self.inference_ids, "reactivity_DMS_MaP": self.inference_preds, "reactivity_2A3_MaP": self.inference_preds})
        submission_df.write_csv(submission_file_path)

