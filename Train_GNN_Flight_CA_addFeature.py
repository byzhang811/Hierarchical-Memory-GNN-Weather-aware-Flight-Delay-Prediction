#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Graph Neural Network for California flight risk prediction
- Adds time-based features
- Handles class imbalance (pos_weight)
- Early stopping
- GraphSAGE architecture
Author: Boyang Zhang + ChatGPT
Date: 2025-10-27
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader   # ✅ new API
from torch_geometric.nn import SAGEConv

# ================================================================
# Argument parser
# ================================================================
parser = argparse.ArgumentParser(description="Train enhanced GNN on California flight network")
parser.add_argument("--data_root", type=str, required=True,
                    help="Root directory containing yearly flight data (2022/2023/2024 subfolders)")
parser.add_argument("--output_dir", type=str, default="./gnn_output", help="Output directory for logs and models")
parser.add_argument("--delay_threshold", type=int, default=60, help="Minutes threshold for 'dangerous' flight label")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--pos_weight", type=float, default=0.0,
                    help="Positive class weight for BCEWithLogitsLoss; 0 = auto-compute")
args = parser.parse_args()

# ================================================================
# Setup
# ================================================================
np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.makedirs(args.output_dir, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print(f"📂 Data root: {args.data_root}")
print(f"📦 Output dir: {args.output_dir}")
print(f"🚀 Device: {DEVICE}")
print("=" * 70)

# ================================================================
# 1. Load all CSV files
# ================================================================
csv_files = glob.glob(os.path.join(args.data_root, "**/*.csv"), recursive=True)
if len(csv_files) == 0:
    raise RuntimeError("No CSV files found under " + args.data_root)

dfs = []
for f in tqdm(csv_files, desc="Loading CSV files"):
    try:
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)
    except Exception as e:
        print(f"⚠️ Error loading {f}: {e}")

df_all = pd.concat(dfs, ignore_index=True)
df_all.columns = [c.strip() for c in df_all.columns]
print(f"✅ Loaded {len(df_all)} total rows from {len(csv_files)} files")

# ================================================================
# 2. Filter California flights
# ================================================================
def filter_ca_flights(df):
    cols = df.columns
    if "ORIGIN_STATE_ABR" in cols and "DEST_STATE_ABR" in cols:
        return df[(df["ORIGIN_STATE_ABR"] == "CA") & (df["DEST_STATE_ABR"] == "CA")]
    elif "ORIGIN_CITY_NAME" in cols and "DEST_CITY_NAME" in cols:
        return df[df["ORIGIN_CITY_NAME"].str.contains(", CA", na=False) &
                  df["DEST_CITY_NAME"].str.contains(", CA", na=False)]
    else:
        raise ValueError("Cannot find state columns.")
        
df_ca = filter_ca_flights(df_all).copy()
print(f"✈️ California-internal flights: {len(df_ca)} rows")

# ================================================================
# 3. Label generation
# ================================================================
for col in ["ARR_DELAY", "CANCELLED", "DIVERTED"]:
    if col not in df_ca.columns:
        df_ca[col] = 0.0
df_ca["label"] = ((df_ca["ARR_DELAY"].fillna(0) > args.delay_threshold) |
                  (df_ca["CANCELLED"] == 1) | (df_ca["DIVERTED"] == 1)).astype(int)
print("Label counts:", df_ca["label"].value_counts().to_dict())

# ================================================================
# 4. Add time-related edge features
# ================================================================
def parse_dep_hour(v):
    try:
        v = int(v)
        return max(0, min(23, v // 100))
    except:
        return 0

def parse_month_day(fl_date):
    try:
        dt = pd.to_datetime(fl_date)
        return dt.month, dt.dayofweek
    except:
        return 1, 0

df_ca["DEP_HOUR"] = df_ca.get("CRS_DEP_TIME", 0).apply(parse_dep_hour)
tmp_md = df_ca.get("FL_DATE", pd.Series(["1/1/2024"] * len(df_ca))).apply(parse_month_day)
df_ca["MONTH"] = tmp_md.apply(lambda x: x[0])
df_ca["DOW"] = tmp_md.apply(lambda x: x[1])
df_ca["IS_WEEKEND"] = ((df_ca["DOW"] >= 5).astype(int))

# ================================================================
# 5. Build nodes and node features
# ================================================================
airports = pd.Index(df_ca["ORIGIN"].tolist() + df_ca["DEST"].tolist()).unique()
node_idx = {ap: i for i, ap in enumerate(airports)}

def safe_mean(series):
    return float(series.dropna().mean()) if len(series.dropna()) > 0 else 0.0

node_features_list = []
for ap in airports:
    out_mask = df_ca["ORIGIN"] == ap
    in_mask = df_ca["DEST"] == ap
    out_cnt = out_mask.sum()
    in_cnt = in_mask.sum()
    avg_arr = safe_mean(df_ca.loc[in_mask, "ARR_DELAY"])
    avg_dep = safe_mean(df_ca.loc[out_mask, "DEP_DELAY"]) if "DEP_DELAY" in df_ca.columns else 0.0
    node_features_list.append([ap, out_cnt + in_cnt, out_cnt, in_cnt, avg_arr, avg_dep])

node_df = pd.DataFrame(node_features_list, columns=["AP", "TOTAL_FLIGHTS", "OUT_CNT", "IN_CNT", "AVG_ARR", "AVG_DEP"])
scaler = StandardScaler()
x = torch.tensor(scaler.fit_transform(node_df[["TOTAL_FLIGHTS", "OUT_CNT", "IN_CNT", "AVG_ARR", "AVG_DEP"]].values),
                 dtype=torch.float)

# ================================================================
# 6. Build edges and edge features
# ================================================================
edge_index, edge_attr, edge_label = [[], []], [], []

if "OP_UNIQUE_CARRIER" in df_ca.columns:
    le_carrier = LabelEncoder()
    df_ca["CARRIER_LE"] = le_carrier.fit_transform(df_ca["OP_UNIQUE_CARRIER"].fillna("NA"))
else:
    df_ca["CARRIER_LE"] = 0

edge_feature_cols = [
    "DISTANCE","CRS_DEP_TIME","CRS_ARR_TIME","ACTUAL_ELAPSED_TIME",
    "AIR_TIME","CARRIER_LE","DEP_HOUR","MONTH","DOW","IS_WEEKEND"
]
for c in edge_feature_cols:
    if c not in df_ca.columns:
        df_ca[c] = 0

for _, row in df_ca.iterrows():
    o, d = row["ORIGIN"], row["DEST"]
    if o not in node_idx or d not in node_idx:
        continue
    edge_index[0].append(node_idx[o])
    edge_index[1].append(node_idx[d])
    edge_attr.append([row[c] if not pd.isna(row[c]) else 0.0 for c in edge_feature_cols])
    edge_label.append(int(row["label"]))

edge_index = torch.tensor(edge_index, dtype=torch.long)
edge_attr = torch.tensor(np.array(edge_attr, dtype=float), dtype=torch.float)
edge_label = torch.tensor(np.array(edge_label, dtype=int), dtype=torch.long)

print(f"Graph summary: {len(airports)} nodes, {edge_index.shape[1]} edges")

# ================================================================
# 7. Split train/val/test
# ================================================================
indices = np.arange(len(edge_label))
train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=edge_label.numpy(), random_state=args.seed)
train_idx, val_idx = train_test_split(train_idx, test_size=0.125, stratify=edge_label.numpy()[train_idx],
                                      random_state=args.seed)
print(f"Split edges: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

def build_dataloader(edge_indices, batch_size, shuffle=True):
    data_list = []
    for i in range(0, len(edge_indices), batch_size):
        sel = edge_indices[i:i+batch_size]
        batch = Data(x=x, edge_index=edge_index[:, sel],
                     edge_attr=edge_attr[sel], y=edge_label[sel])
        data_list.append(batch)
    return DataLoader(data_list, batch_size=1, shuffle=shuffle)

train_loader = build_dataloader(train_idx, args.batch_size)
val_loader = build_dataloader(val_idx, args.batch_size, shuffle=False)
test_loader = build_dataloader(test_idx, args.batch_size, shuffle=False)

# ================================================================
# 8. Model definition
# ================================================================
class FlightGNN(nn.Module):
    def __init__(self, in_node_dim, edge_attr_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_node_dim, 96)
        self.conv2 = SAGEConv(96, 96)
        self.bn = nn.BatchNorm1d(96)
        self.mlp = nn.Sequential(
            nn.Linear(96*2 + edge_attr_dim, 192),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 1)
        )
    def forward(self, x, full_ei, edge_index_batch, edge_attr_batch):
        h = F.relu(self.conv1(x, full_ei))
        h = self.bn(F.relu(self.conv2(h, full_ei)))
        src, dst = h[edge_index_batch[0]], h[edge_index_batch[1]]
        cat = torch.cat([src, dst, edge_attr_batch], dim=1)
        return self.mlp(cat).squeeze(-1)

model = FlightGNN(x.shape[1], edge_attr.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# ================================================================
# 9. Weighted loss for imbalance
# ================================================================
neg = int((df_ca["label"] == 0).sum())
pos = int((df_ca["label"] == 1).sum())
auto_ratio = max(1.0, neg / max(1, pos))
pos_w = args.pos_weight if args.pos_weight > 0 else float(np.sqrt(auto_ratio))
print(f"Using pos_weight={pos_w:.2f}")
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=DEVICE))

# ================================================================
# 10. Training with early stopping
# ================================================================
def evaluate(loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            x_b = batch.x.to(DEVICE)
            logits = model(x_b, edge_index.to(DEVICE),
                           batch.edge_index.to(DEVICE),
                           batch.edge_attr.to(DEVICE))
            prob = torch.sigmoid(logits)
            ys.append(batch.y.numpy())
            ps.append(prob.cpu().numpy())
    y_true, y_prob = np.concatenate(ys), np.concatenate(ps)
    acc = ((y_prob >= 0.5) == y_true).mean()
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return acc, auc

print("\nStart training...\n")
best_val_auc = 0.0
patience = 5
no_improve = 0
best_state = None

for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch.x.to(DEVICE),
                       edge_index.to(DEVICE),
                       batch.edge_index.to(DEVICE),
                       batch.edge_attr.to(DEVICE))
        loss = criterion(logits, batch.y.to(DEVICE).float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    val_acc, val_auc = evaluate(val_loader)
    test_acc, test_auc = evaluate(test_loader)
    print(f"Epoch {epoch:02d}: loss={total_loss/len(train_loader):.4f}, "
          f"val_auc={val_auc:.4f}, val_acc={val_acc:.4f}, test_auc={test_auc:.4f}, test_acc={test_acc:.4f}")
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best val AUC={best_val_auc:.4f})")
            break

if best_state is not None:
    model.load_state_dict(best_state)
torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
print(f"\n✅ Training finished. Best validation AUC: {best_val_auc:.4f}")
