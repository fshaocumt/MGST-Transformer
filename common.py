"""Shared utilities: data loading, windowing, training loop, evaluation metrics,
coordinate-descent hyper-parameter search and result export.
"""

"""Shared utilities for the MGST-Transformer experiments.

Protocol used throughout (identical to the original notebooks, kept unchanged
so that old and new numbers remain comparable):

  * look-back window   phi   = 12 steps (60 min)
  * forecast horizon   theta = 6 steps (30 min)
  * chronological split 65 / 15 / 20 (train / val / test), no shuffling
  * x-scaler, y-scaler, correlation matrices and K-means clusters are fitted
    on the TRAINING split only
  * early stopping on validation MSE, patience = 12, best checkpoint restored
  * every metric is computed AFTER inverse-transforming y with y_scaler,
    i.e. in the original veh/(lane*hour) unit

All models share one entry interface:

    model(x)              -> (B, theta, N)      temporal baselines
    model(x, A_list)      -> (B, theta, N)      graph-aware models

so that a single train()/evaluate() pair can drive every baseline.
"""

import copy
import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch.utils.data import DataLoader, Dataset

BUILD = "2026-10-02e"

if os.environ.get("MGST_QUIET_BUILD") != "1":
    print(f"[build] mgst/common.py {BUILD}", flush=True)

def set_seed(seed=42, deterministic=True):
    """deterministic=True  for the final runs (reproducible numbers);
    deterministic=False for the search trials (lets cuDNN pick the fast
    kernels -- here we only need a ranking, not bit-exact numbers).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_device(gpu=None):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrafficWindowDataset(Dataset):

    def __init__(self, X, y, input_len=12, pred_len=6):
        self.X = X
        self.y = y
        self.input_len = input_len
        self.pred_len = pred_len
        self.T = X.shape[0]

    def __len__(self):
        return self.T - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.input_len]                                  # [phi, N, F]
        y = self.y[idx + self.input_len: idx + self.input_len + self.pred_len]  # [theta, N]
        return torch.from_numpy(np.ascontiguousarray(x)), \
            torch.from_numpy(np.ascontiguousarray(y))

def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test,
                       batch_size=64, num_workers=0, seed=42, input_len=12, pred_len=6):
    train_ds = TrafficWindowDataset(X_train, y_train, input_len, pred_len)
    val_ds = TrafficWindowDataset(X_val, y_val, input_len, pred_len)
    test_ds = TrafficWindowDataset(X_test, y_test, input_len, pred_len)

    use_pin = torch.cuda.is_available()
    persistent = num_workers > 0
    g = torch.Generator()
    g.manual_seed(seed)

    common = dict(
        pin_memory=use_pin,
        num_workers=num_workers,
        persistent_workers=persistent,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, generator=g, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader

_TAG_CANDIDATES = ["", "18年", "24年", "18", "24", "_18", "_24"]

def resolve_tag(data_dir, tag=None):
    """Return the filename suffix whose `dataset<suffix>.npz` really exists.

    `tag=None` (auto) probes the directory. This removes the whole class of
    "silently loaded the wrong year / FileNotFoundError" bugs: the printed
    suffix tells you exactly which subset is being trained on.
    """
    probe = "dataset{}.npz"
    if tag is not None and os.path.exists(os.path.join(data_dir, probe.format(tag))):
        return tag
    found = [c for c in _TAG_CANDIDATES
             if os.path.exists(os.path.join(data_dir, probe.format(c)))]
    if not found:
        import glob as _glob
        hits = sorted(_glob.glob(os.path.join(data_dir, "dataset*.npz")))
        if hits:
            base = os.path.basename(hits[0])
            found = [base[len("dataset"):-len(".npz")]]
            print(f"[tag] suffix not in the known list; globbed '{found[0]}' "
                  f"from {base}", flush=True)
    if not found:
        listing = sorted(os.listdir(data_dir))[:20] if os.path.isdir(data_dir) else []
        raise FileNotFoundError(
            f"no dataset*.npz found in {data_dir} (tried suffixes "
            f"{[tag] + _TAG_CANDIDATES}). Directory contains: {listing}")
    if tag is not None and tag not in found:
        print(f"[tag] requested suffix '{tag}' not found in {data_dir}; "
              f"auto-selected '{found[0]}'", flush=True)
    else:
        print(f"[tag] using suffix '{found[0]}' in {data_dir}", flush=True)
    return found[0]

def load_preprocessed(data_dir, tag="", input_len=12, pred_len=6):
    tag = resolve_tag(data_dir, tag)
    npz = np.load(os.path.join(data_dir, f"dataset{tag}.npz"))
    data = {
        "X_train": npz["X_train"].astype(np.float32),
        "X_val": npz["X_val"].astype(np.float32),
        "X_test": npz["X_test"].astype(np.float32),
        "y_train": npz["y_train"].astype(np.float32),
        "y_val": npz["y_val"].astype(np.float32),
        "y_test": npz["y_test"].astype(np.float32),
    }
    with open(os.path.join(data_dir, f"scalers{tag}.pkl"), "rb") as f:
        data["y_scaler"] = pickle.load(f)["y_scaler"]

    data["sensors"] = pd.read_csv(
        os.path.join(data_dir, f"common_sensors{tag}.csv"))["sensor_id"].tolist()

    g = np.load(os.path.join(data_dir, f"graph_mats{tag}.npz"))
    data["graphs"] = {
        "distance": g["A_distance"].astype(np.float32),
        "flow_corr": g["A_flow_corr"].astype(np.float32),
        "speed_corr": g["A_speed_corr"].astype(np.float32),
        "region": g["A_region"].astype(np.float32),
    }
    data["input_len"] = input_len
    data["pred_len"] = pred_len
    data["tag"] = tag            # resolved suffix, recorded in the run config
    return data

def inverse_transform_3d(scaler, arr):
    shape = arr.shape
    inv = scaler.inverse_transform(arr.reshape(-1, 1))
    return inv.reshape(shape)

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
    return {
        "MSE": mean_squared_error(yt, yp),
        "RMSE": np.sqrt(mean_squared_error(yt, yp)),
        "MAE": mean_absolute_error(yt, yp),
        "MAPE(%)": safe_mape(yt, yp),
        "sMAPE(%)": smape(yt, yp),
        "R2": r2_score(yt, yp),
        "EVS": explained_variance_score(yt, yp),
        "WAPE(%)": wape(yt, yp),
    }

def _predict(model, x, A_list, teacher=None):
    if A_list is not None:
        return model(x, A_list)
    if teacher is not None and getattr(model, "accepts_teacher", False):
        return model(x, teacher)
    return model(x)

@torch.inference_mode()
def predict_loader(model, loader, device, A_list=None):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        pred = _predict(model, x, A_list)
        preds.append(pred.detach().cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(preds, 0), np.concatenate(trues, 0)

def evaluate_model(model, loader, y_scaler, device, A_list=None):
    preds, trues = predict_loader(model, loader, device, A_list)
    preds_inv = inverse_transform_3d(y_scaler, preds)
    trues_inv = inverse_transform_3d(y_scaler, trues)

    overall = compute_metrics(trues_inv, preds_inv)
    per_horizon = {f"H{h + 1}": compute_metrics(trues_inv[:, h, :], preds_inv[:, h, :])
                   for h in range(preds_inv.shape[1])}
    return overall, per_horizon, preds_inv, trues_inv

def _autocast(enabled):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=enabled)

def _grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)

def train_model(model, train_loader, val_loader, device,
                epochs=80, lr=1e-3, weight_decay=1e-4, patience=12,
                A_list=None, verbose=True, grad_clip=5.0, amp=False):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = model.to(device)
    scaler = _grad_scaler(amp)

    best_val, best_state, bad = float("inf"), None, 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(amp):
                loss = criterion(_predict(model, x, A_list, teacher=y), y)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            tot += loss.item()
            n += 1
        train_losses.append(tot / max(n, 1))

        model.eval()
        tot, n = 0.0, 0
        with torch.inference_mode():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with _autocast(amp):
                    loss = criterion(_predict(model, x, A_list), y)
                tot += loss.item()
                n += 1
        val_losses.append(tot / max(n, 1))

        if verbose:
            print(f"Epoch [{epoch:03d}/{epochs}] | Train MSE: {train_losses[-1]:.6f} "
                  f"| Val MSE: {val_losses[-1]:.6f}", flush=True)

        if val_losses[-1] < best_val:
            best_val, bad = val_losses[-1], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}. Best val MSE = {best_val:.6f}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"train": train_losses, "val": val_losses,
                   "best_val": best_val,
                   "best_epoch": int(np.argmin(val_losses)) + 1}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_run(out_dir, name, cfg, overall, per_horizon, history,
             preds=None, trues=None):
    os.makedirs(out_dir, exist_ok=True)
    row = {"model": name}
    row.update(overall)
    row.update({k: cfg.get(k) for k in ("seed", "fusion", "graphs") if k in cfg})
    pd.DataFrame([row]).to_csv(os.path.join(out_dir, f"{name}_overall.csv"), index=False)

    pd.DataFrame([{"Horizon": h, **m} for h, m in per_horizon.items()]).to_csv(
        os.path.join(out_dir, f"{name}_horizon.csv"), index=False)
    pd.DataFrame({"Epoch": np.arange(1, len(history["train"]) + 1),
                  "Train_MSE": history["train"],
                  "Val_MSE": history["val"]}).to_csv(
        os.path.join(out_dir, f"{name}_loss.csv"), index=False)
    with open(os.path.join(out_dir, f"{name}_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    if preds is not None:
        np.savez_compressed(os.path.join(out_dir, f"{name}_pred.npz"),
                            preds=preds.astype(np.float16),
                            trues=trues.astype(np.float16))
    return row
