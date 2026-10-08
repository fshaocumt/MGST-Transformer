"""Original sequence-model comparison code: MLP, RNN (LSTM/GRU), TCN and
Transformer baselines with their training and evaluation loop and metric
export (RMSE, MAE, sMAPE, WAPE, R2, EVS).
"""

import os
import copy
import json
import random
import pickle
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)

FLOW_PATH = r"/root/autodl-fs/pems07_flow.csv"
SPEED_PATH = r"/root/autodl-fs/speed_272.xlsx"
SENSOR_PATH = r"/root/autodl-fs/sensor_800+.csv"

DATA_DIR = r"/root/autodl-fs/preprocess_result"
RESULT_ROOT = r"/root/autodl-fs/model_compare_results18年"

INPUT_LEN = 12
PRED_LEN = 6
TEST_RATIO = 0.20
VAL_RATIO = 0.15

EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 12

BATCH_SIZE = 64
NUM_WORKERS = 0
BASE_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_autocast_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

class TrafficWindowDataset(Dataset):
    def __init__(self, X, y, input_len=INPUT_LEN, pred_len=PRED_LEN):
        self.X = X
        self.y = y
        self.input_len = input_len
        self.pred_len = pred_len
        self.T = X.shape[0]

    def __len__(self):
        return self.T - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.input_len]
        y = self.y[idx + self.input_len: idx + self.input_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def load_preprocessed_data():
    data = np.load(os.path.join(DATA_DIR, "dataset.npz"))

    X_train = data["X_train"].astype(np.float32)
    X_val = data["X_val"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    y_val = data["y_val"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    with open(os.path.join(DATA_DIR, "scalers.pkl"), "rb") as f:
        scaler_dict = pickle.load(f)

    y_scaler = scaler_dict["y_scaler"]
    common_sensors = pd.read_csv(os.path.join(DATA_DIR, "common_sensors.csv"))["sensor_id"].tolist()

    return X_train, X_val, X_test, y_train, y_val, y_test, y_scaler, common_sensors

def load_test_timestamps(data_dir=DATA_DIR):
    test_ts_df = pd.read_csv(os.path.join(data_dir, "test_timestamps.csv"))
    test_ts = pd.to_datetime(test_ts_df["timestamp"], errors="coerce").dropna().reset_index(drop=True)

    window_num = len(test_ts) - INPUT_LEN - PRED_LEN + 1
    if window_num <= 0:
        raise ValueError("Test timestamps are too short to form sliding windows.")

    pred_start_ts = test_ts[INPUT_LEN: INPUT_LEN + window_num].reset_index(drop=True)
    return pred_start_ts, test_ts

def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test,
                       batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, seed=BASE_SEED):
    train_ds = TrafficWindowDataset(X_train, y_train)
    val_ds = TrafficWindowDataset(X_val, y_val)
    test_ds = TrafficWindowDataset(X_test, y_test)

    use_pin = torch.cuda.is_available()
    persistent = num_workers > 0
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_pin,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=persistent,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_pin,
        num_workers=num_workers,
        persistent_workers=persistent,
        worker_init_fn=seed_worker if num_workers > 0 else None
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_pin,
        num_workers=num_workers,
        persistent_workers=persistent,
        worker_init_fn=seed_worker if num_workers > 0 else None
    )

    print(f"DataLoader: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    return train_loader, val_loader, test_loader

def inverse_transform_3d(scaler, arr):
    shape = arr.shape
    arr2 = arr.reshape(-1, 1)
    inv = scaler.inverse_transform(arr2).reshape(shape)
    return inv

def safe_mape(y_true, y_pred, eps=1e-8):
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

def smape(y_true, y_pred, eps=1e-8):
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0

def wape(y_true, y_pred, eps=1e-8):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps) * 100.0

def compute_metrics(y_true, y_pred):
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    mse = mean_squared_error(yt, yp)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(yt, yp)
    mape = safe_mape(yt, yp)
    smape_v = smape(yt, yp)
    r2 = r2_score(yt, yp)
    evs = explained_variance_score(yt, yp)
    wape_v = wape(yt, yp)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE(%)": mape,
        "sMAPE(%)": smape_v,
        "R2": r2,
        "EVS": evs,
        "WAPE(%)": wape_v
    }

def evaluate_model(model, loader, y_scaler, device):
    model.eval()
    preds_all = []
    trues_all = []

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with get_autocast_context(device):
                pred = model(x)

            preds_all.append(pred.detach().cpu().numpy())
            trues_all.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds_all, axis=0)
    trues = np.concatenate(trues_all, axis=0)

    preds_inv = inverse_transform_3d(y_scaler, preds)
    trues_inv = inverse_transform_3d(y_scaler, trues)

    overall_metrics = compute_metrics(trues_inv, preds_inv)

    horizon_metrics = {}
    for h in range(preds_inv.shape[1]):
        horizon_metrics[f"H{h + 1}"] = compute_metrics(trues_inv[:, h, :], preds_inv[:, h, :])

    return overall_metrics, horizon_metrics, preds_inv, trues_inv

class FlattenMLP(nn.Module):
    def __init__(self, input_len, num_nodes, in_dim, pred_len=PRED_LEN,
                 hidden_dims=(256, 128), dropout=0.1):
        super().__init__()
        self.input_len = input_len
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.pred_len = pred_len

        flat_dim = input_len * num_nodes * in_dim
        layers = []
        prev = flat_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev = h
        layers.append(nn.Linear(prev, pred_len * num_nodes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        B = x.shape[0]
        out = self.mlp(x.reshape(B, -1))
        return out.reshape(B, self.pred_len, self.num_nodes)

class SequenceEncoder(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim=64):
        super().__init__()
        self.step_dim = num_nodes * in_dim
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(self.step_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, T, N, F = x.shape
        x = x.reshape(B, T, N * F)
        x = self.input_proj(x)
        x = self.norm(x)
        return x

class RNNBaseline(nn.Module):
    def __init__(self, num_nodes, in_dim, pred_len=PRED_LEN,
                 hidden_dim=64, num_layers=2, dropout=0.1, cell_type="lstm"):
        super().__init__()
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type.lower()

        self.encoder = SequenceEncoder(num_nodes, in_dim, hidden_dim)

        rnn_dropout = dropout if num_layers > 1 else 0.0
        if self.cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout
            )
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout
            )
        else:
            raise ValueError("cell_type must be either 'lstm' or 'gru'")

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * num_nodes)
        )

    def forward(self, x):
        x = self.encoder(x)  # [B, T, H]
        out, h = self.rnn(x)

        if self.cell_type == "lstm":
            rep = h[0][-1]   # last layer hidden state
        else:
            rep = h[-1]

        y = self.head(rep)
        return y.reshape(x.shape[0], self.pred_len, self.num_nodes)

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)

        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = F.relu(out)
        out = self.dropout(out)

        res = self.downsample(x)
        out = self.norm(out + res)
        return F.relu(out)

class TCNBaseline(nn.Module):
    def __init__(self, num_nodes, in_dim, pred_len=PRED_LEN,
                 hidden_dim=64, levels=3, dropout=0.1, kernel_size=3):
        super().__init__()
        self.pred_len = pred_len
        self.num_nodes = num_nodes

        self.encoder = SequenceEncoder(num_nodes, in_dim, hidden_dim)

        blocks = []
        for i in range(levels):
            dilation = 2 ** i
            blocks.append(
                TemporalBlock(
                    in_ch=hidden_dim,
                    out_ch=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        self.tcn = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * num_nodes)
        )

    def forward(self, x):
        x = self.encoder(x)         # [B, T, H]
        x = x.transpose(1, 2)       # [B, H, T]
        feat = self.tcn(x)          # [B, H, T]
        rep = feat[:, :, -1]        # [B, H]
        y = self.head(rep)
        return y.reshape(x.shape[0], self.pred_len, self.num_nodes)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerBaseline(nn.Module):
    def __init__(self, num_nodes, in_dim, pred_len=PRED_LEN,
                 d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.num_nodes = num_nodes

        self.encoder = SequenceEncoder(num_nodes, in_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * num_nodes)
        )

    def forward(self, x):
        x = self.encoder(x)     # [B, T, H]
        x = self.pos_enc(x)
        out = self.transformer(x)   # [B, T, H]
        rep = out.mean(dim=1)       
        y = self.head(rep)
        return y.reshape(x.shape[0], self.pred_len, self.num_nodes)

def train_model(model, train_loader, val_loader, epochs, lr, weight_decay, patience, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    model = model.to(device)
    best_val = float("inf")
    best_state = None
    bad_count = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train = 0.0
        n_train = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with get_autocast_context(device):
                pred = model(x)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train += loss.item()
            n_train += 1

        train_loss = total_train / max(n_train, 1)
        train_losses.append(train_loss)

        model.eval()
        total_val = 0.0
        n_val = 0

        with torch.inference_mode():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with get_autocast_context(device):
                    pred = model(x)
                    loss = criterion(pred, y)

                total_val += loss.item()
                n_val += 1

        val_loss = total_val / max(n_val, 1)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch:03d}/{epochs}] | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_count = 0
        else:
            bad_count += 1
            if bad_count >= patience:
                print(f"Early stopping triggered: epoch={epoch}, best validation loss={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses

def save_results(exp_dir, preds_inv, trues_inv, overall_metrics, horizon_metrics,
                 pred_start_ts=None, train_losses=None, val_losses=None,
                 cfg=None, common_sensors=None):
    os.makedirs(exp_dir, exist_ok=True)

    save_dict = {"preds": preds_inv, "trues": trues_inv}
    if pred_start_ts is not None:
        pred_start_str = pd.Series(pred_start_ts).astype(str).values
        save_dict["pred_start_ts"] = pred_start_str
        pd.DataFrame({"pred_start_ts": pred_start_str}).to_csv(
            os.path.join(exp_dir, "pred_start_timestamps18年.csv"), index=False
        )

    np.savez_compressed(os.path.join(exp_dir, "test_predictions18年.npz"), **save_dict)
    pd.DataFrame([overall_metrics]).to_csv(os.path.join(exp_dir, "overall_metrics18年.csv"), index=False)

    horizon_rows = []
    for h, met in horizon_metrics.items():
        row = {"Horizon": h}
        row.update(met)
        horizon_rows.append(row)
    pd.DataFrame(horizon_rows).to_csv(os.path.join(exp_dir, "horizon_metrics18年.csv"), index=False)

    if train_losses and val_losses:
        loss_df = pd.DataFrame({
            "Epoch": np.arange(1, len(train_losses) + 1),
            "Train_MSE": train_losses,
            "Val_MSE": val_losses
        })
        loss_df.to_csv(os.path.join(exp_dir, "loss_curve18年.csv"), index=False)

    if cfg is not None:
        with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    if common_sensors is not None:
        pd.Series(common_sensors).to_csv(
            os.path.join(exp_dir, "common_sensors18年.csv"),
            index=False,
            header=["sensor_id"]
        )

    print(f"Results saved to: {exp_dir}")

def build_model(spec, num_nodes, in_dim):
    name = spec["name"]
    if name == "LSTM":
        return RNNBaseline(
            num_nodes=num_nodes,
            in_dim=in_dim,
            pred_len=PRED_LEN,
            hidden_dim=spec.get("hidden_dim", 64),
            num_layers=spec.get("num_layers", 2),
            dropout=spec.get("dropout", 0.1),
            cell_type="lstm"
        )

    if name == "GRU":
        return RNNBaseline(
            num_nodes=num_nodes,
            in_dim=in_dim,
            pred_len=PRED_LEN,
            hidden_dim=spec.get("hidden_dim", 64),
            num_layers=spec.get("num_layers", 2),
            dropout=spec.get("dropout", 0.1),
            cell_type="gru"
        )

    if name == "TCN":
        return TCNBaseline(
            num_nodes=num_nodes,
            in_dim=in_dim,
            pred_len=PRED_LEN,
            hidden_dim=spec.get("hidden_dim", 64),
            levels=spec.get("levels", 3),
            dropout=spec.get("dropout", 0.1),
            kernel_size=spec.get("kernel_size", 3)
        )

    if name == "Transformer":
        return TransformerBaseline(
            num_nodes=num_nodes,
            in_dim=in_dim,
            pred_len=PRED_LEN,
            d_model=spec.get("d_model", 64),
            nhead=spec.get("nhead", 4),
            num_layers=spec.get("num_layers", 2),
            dim_feedforward=spec.get("dim_feedforward", 128),
            dropout=spec.get("dropout", 0.1)
        )

    raise ValueError(f"Unknown model: {name}")

def run_single_model(spec, train_loader, val_loader, test_loader,
                     num_nodes, in_dim, y_scaler, pred_start_ts, common_sensors):
    set_seed(BASE_SEED)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_name = spec["name"]
    exp_dir = os.path.join(RESULT_ROOT, model_name)
    os.makedirs(exp_dir, exist_ok=True)

    print("\n" + "=" * 100)
    print(f"Starting baseline experiment: {model_name}")
    print("=" * 100)

    model = build_model(spec, num_nodes=num_nodes, in_dim=in_dim).to(DEVICE)
    print(f"Model parameter device: {next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else 'no trainable parameters'}")

    trainable = spec.get("trainable", True)

    if trainable:
        model, train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            epochs=spec.get("epochs", EPOCHS),
            lr=spec.get("lr", LR),
            weight_decay=spec.get("weight_decay", WEIGHT_DECAY),
            patience=spec.get("patience", PATIENCE),
            device=DEVICE
        )
    else:
        train_losses, val_losses = [], []

    overall_metrics, horizon_metrics, preds_inv, trues_inv = evaluate_model(
        model, test_loader, y_scaler, DEVICE
    )

    print("\n========== Overall metrics ==========")
    for k, v in overall_metrics.items():
        print(f"{k}: {v:.6f}")

    save_results(
        exp_dir=exp_dir,
        preds_inv=preds_inv,
        trues_inv=trues_inv,
        overall_metrics=overall_metrics,
        horizon_metrics=horizon_metrics,
        pred_start_ts=pred_start_ts,
        train_losses=train_losses,
        val_losses=val_losses,
        cfg=spec,
        common_sensors=common_sensors
    )

    return overall_metrics

def main():
    os.makedirs(RESULT_ROOT, exist_ok=True)
    set_seed(BASE_SEED)

    X_train, X_val, X_test, y_train, y_val, y_test, y_scaler, common_sensors = load_preprocessed_data()
    pred_start_ts, test_ts = load_test_timestamps()

    num_nodes = len(common_sensors)
    in_dim = X_train.shape[-1]

    print(f"Input dimension: {in_dim}")
    print(f"Number of nodes: {num_nodes}")
    print(f"Training windows: {len(TrafficWindowDataset(X_train, y_train))}")
    print(f"Validation windows: {len(TrafficWindowDataset(X_val, y_val))}")
    print(f"Test windows: {len(TrafficWindowDataset(X_test, y_test))}")

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, seed=BASE_SEED
    )

    model_specs = [
        {
            "name": "LSTM",
            "trainable": True,
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "epochs": 80,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "patience": 12
        },
        {
            "name": "GRU",
            "trainable": True,
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "epochs": 80,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "patience": 12
        },
        {
            "name": "TCN",
            "trainable": True,
            "hidden_dim": 64,
            "levels": 3,
            "kernel_size": 3,
            "dropout": 0.1,
            "epochs": 80,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "patience": 12
        },
        {
            "name": "Transformer",
            "trainable": True,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "epochs": 80,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "patience": 12
        }
    ]

    all_results = []
    for spec in model_specs:
        metrics = run_single_model(
            spec=spec,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_nodes=num_nodes,
            in_dim=in_dim,
            y_scaler=y_scaler,
            pred_start_ts=pred_start_ts,
            common_sensors=common_sensors
        )
        row = {"Model": spec["name"]}
        row.update(metrics)
        all_results.append(row)

    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values(by="RMSE", ascending=True).reset_index(drop=True)

    summary_path = os.path.join(RESULT_ROOT, "model_comparison_summary18年.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print("Baseline comparison summary (sorted by RMSE)")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    print(f"\nSummary saved to: {summary_path}")
    print("Baseline experiments finished.")

if __name__ == "__main__":
    main()
