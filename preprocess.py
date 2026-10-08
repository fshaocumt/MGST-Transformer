"""Build the multi-graph traffic datasets from raw PeMS files.
Loads and cleans flow/speed data, builds node features, constructs the four
graphs (distance, flow correlation, speed correlation, regional clustering),
splits into train/val/test and standardizes (fit on training set only).
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch
from torch.utils.data import Dataset, DataLoader

torch.set_default_device("cpu")

FLOW_PATH = r"/root/autodl-fs/加州流量24年.csv"
SPEED_PATH = r"/root/autodl-fs/加州速度24年.csv"
SENSOR_PATH = r"/root/autodl-fs/加州24年传感器_补充后.csv"

SAVE_DIR = r"/root/autodl-fs/preprocess_result24年"

INPUT_LEN = 12
PRED_LEN = 6
TEST_RATIO = 0.20
VAL_RATIO = 0.15

TOP_K = 8
SIGMA_KM = 2.0
SPARSE_REGION = False

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
        x = self.X[idx: idx + self.input_len]  # [Tin, N, F]
        y = self.y[idx + self.input_len: idx + self.input_len + self.pred_len]  # [Tout, N]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def add_time_features(index):
    ts = pd.DatetimeIndex(index)
    hour = ts.hour.values
    minute = ts.minute.values
    dow = ts.dayofweek.values
    is_weekend = (dow >= 5).astype(np.float32)

    minutes_of_day = hour * 60 + minute
    is_weekday = (dow < 5)

    morning_peak = ((minutes_of_day >= 7 * 60) & (minutes_of_day < 9 * 60) & is_weekday).astype(np.float32)
    evening_peak = ((minutes_of_day >= 16 * 60) & (minutes_of_day < 18 * 60) & is_weekday).astype(np.float32)

    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)

    feat = np.stack([
        hour_sin,
        hour_cos,
        dow_sin,
        dow_cos,
        is_weekend,
        morning_peak,
        evening_peak
    ], axis=-1)

    return feat.astype(np.float32)

def haversine_distance_matrix(coords):
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0
    return R * c

def load_and_clean_data():
    print("=" * 50 + " 开始读取数据 " + "=" * 50)

    df_flow = pd.read_csv(FLOW_PATH)
    df_flow["timestamp"] = pd.to_datetime(df_flow["timestamp"], errors="coerce")
    df_flow = df_flow.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    flow_sensors = [int(c) for c in df_flow.columns if str(c).isdigit()]
    df_flow.columns = [int(c) if str(c).isdigit() else c for c in df_flow.columns]
    df_flow = df_flow[flow_sensors]
    print(f"原始流量传感器数: {len(flow_sensors)}")

    df_speed = pd.read_csv(SPEED_PATH)
    df_speed["Timestamp"] = pd.to_datetime(df_speed["Timestamp"], errors="coerce")
    df_speed = df_speed.dropna(subset=["Timestamp"]).set_index("Timestamp").sort_index()
    speed_sensors = [int(c) for c in df_speed.columns if str(c).isdigit()]
    df_speed.columns = [int(c) if str(c).isdigit() else c for c in df_speed.columns]
    df_speed = df_speed[speed_sensors]
    print(f"原始速度传感器数: {len(speed_sensors)}")

    df_sensor = pd.read_csv(SENSOR_PATH)
    df_sensor = df_sensor.rename(columns={
        "ID": "sensor_id",
        "Latitude": "lat",
        "Longitude": "lon",
        "Lanes": "lanes"
    })
    df_sensor = df_sensor[["sensor_id", "lat", "lon", "lanes"]].dropna()
    df_sensor["sensor_id"] = df_sensor["sensor_id"].astype(int)
    sensor_meta = df_sensor.set_index("sensor_id")
    print(f"有效传感器元数据数: {len(sensor_meta)}")

    common_sensors = sorted(list(set(flow_sensors) & set(speed_sensors) & set(sensor_meta.index)))
    if len(common_sensors) == 0:
        raise ValueError("没有找到流量、速度、元数据都存在的共同传感器，请检查列名和ID。")
    print(f"流量/速度/元数据共传感器数: {len(common_sensors)}")

    df_flow = df_flow[common_sensors].copy()
    df_speed = df_speed[common_sensors].copy()
    sensor_meta = sensor_meta.reindex(common_sensors)

    lanes = sensor_meta["lanes"].values.reshape(1, -1)
    lanes = np.where(lanes <= 0, 1.0, lanes)
    df_flow = df_flow * 12.0 / lanes
    print("流量单位转换完成：5分钟车辆数 -> 每小时每车道车辆数")

    df_flow = df_flow.ffill().bfill()
    df_speed = df_speed.ffill().bfill()

    common_ts = df_flow.index.intersection(df_speed.index)
    df_flow = df_flow.reindex(common_ts)
    df_speed = df_speed.reindex(common_ts)
    print(f"时间戳交集数: {len(common_ts)}")

    return df_flow, df_speed, sensor_meta, common_sensors, common_ts

def build_features(df_flow, df_speed, common_ts):
    print("=" * 50 + " 构建时空特征 " + "=" * 50)

    flow = df_flow.values.astype(np.float32)
    speed = df_speed.values.astype(np.float32)
    time_feat = add_time_features(common_ts)  # [T, 7]

    T, N = flow.shape
    time_feat_expand = np.repeat(time_feat[:, None, :], N, axis=1)  # [T, N, 7]

    X = np.concatenate([
        flow[..., None],        # [T, N, 1]
        speed[..., None],       # [T, N, 1]
        time_feat_expand        # [T, N, 7]
    ], axis=-1)                 # [T, N, 9]

    y = flow.copy()
    print(f"特征构建完成：X.shape={X.shape}, y.shape={y.shape}")
    return X, y

def split_series(X, y, common_ts):
    print("=" * 50 + " 划分数据集 " + "=" * 50)

    T = X.shape[0]
    test_size = int(T * TEST_RATIO)
    val_size = int(T * VAL_RATIO)
    train_size = T - val_size - test_size

    X_train = X[:train_size]
    y_train = y[:train_size]
    ts_train = common_ts[:train_size]

    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    ts_val = common_ts[train_size:train_size + val_size]

    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    ts_test = common_ts[train_size + val_size:]

    print(f"训练集: {len(ts_train)} | 验证集: {len(ts_val)} | 测试集: {len(ts_test)}")
    return (X_train, y_train, ts_train,
            X_val, y_val, ts_val,
            X_test, y_test, ts_test)

def standardize_data(X_train, X_val, X_test, y_train, y_val, y_test):
    print("=" * 50 + " 标准化 " + "=" * 50)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    y_scaler.fit(y_train.reshape(-1, 1))

    def transform_X(X):
        return x_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)

    def transform_y(y):
        return y_scaler.transform(y.reshape(-1, 1)).reshape(y.shape).astype(np.float32)

    X_train_s = transform_X(X_train)
    X_val_s = transform_X(X_val)
    X_test_s = transform_X(X_test)

    y_train_s = transform_y(y_train)
    y_val_s = transform_y(y_val)
    y_test_s = transform_y(y_test)

    return X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, x_scaler, y_scaler

def _row_normalize(A):
    row_sum = A.sum(axis=1, keepdims=True) + 1e-8
    return A / row_sum

def _topk_sparse(A, top_k):
    N = A.shape[0]
    if top_k is None or top_k >= N:
        return A

    A_sparse = np.zeros_like(A)
    for i in range(N):
        idx = np.argsort(-A[i])[:top_k + 1]
        A_sparse[i, idx] = A[i, idx]
    return A_sparse

def build_multi_graphs(df_flow_train, df_speed_train, sensor_meta_train):
    print("=" * 50 + " 构建多图 " + "=" * 50)

    coords = sensor_meta_train[["lat", "lon"]].values.astype(np.float32)
    N = coords.shape[0]

    dist_km = haversine_distance_matrix(coords)
    A_dist = np.exp(-(dist_km ** 2) / (SIGMA_KM ** 2 + 1e-8))
    np.fill_diagonal(A_dist, 1.0)

    flow_corr = np.corrcoef(df_flow_train.values.T)
    flow_corr = np.nan_to_num(np.abs(flow_corr), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(flow_corr, 1.0)

    speed_corr = np.corrcoef(df_speed_train.values.T)
    speed_corr = np.nan_to_num(np.abs(speed_corr), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(speed_corr, 1.0)

    flow_mean = df_flow_train.mean(axis=0).values
    flow_std = df_flow_train.std(axis=0).values
    speed_mean = df_speed_train.mean(axis=0).values
    speed_std = df_speed_train.std(axis=0).values

    sensor_feat = np.column_stack([
        flow_mean,
        flow_std,
        speed_mean,
        speed_std,
        coords[:, 0],
        coords[:, 1]
    ])

    feat_scaler = StandardScaler()
    sensor_feat_scaled = feat_scaler.fit_transform(sensor_feat)

    n_clusters = max(2, min(12, N // 8 if N >= 8 else 2))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(sensor_feat_scaled)

    A_region = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if labels[i] == labels[j]:
                A_region[i, j] = 1.0
    np.fill_diagonal(A_region, 1.0)

    A_list = []
    for name, A in [
        ("distance", A_dist),
        ("flow_corr", flow_corr),
        ("speed_corr", speed_corr),
        ("region", A_region),
    ]:
        if SPARSE_REGION or name != "region":
            A = _topk_sparse(A, TOP_K)
        A = _row_normalize(A)
        A_list.append(A.astype(np.float32))
        print(f"{name} 图完成：shape={A.shape}")

    return A_list

def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=64, num_workers=4):
    train_ds = TrafficWindowDataset(X_train, y_train, INPUT_LEN, PRED_LEN)
    val_ds = TrafficWindowDataset(X_val, y_val, INPUT_LEN, PRED_LEN)
    test_ds = TrafficWindowDataset(X_test, y_test, INPUT_LEN, PRED_LEN)

    use_pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_pin_memory,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=persistent
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent
    )

    print(f"DataLoader完成：train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    return train_loader, val_loader, test_loader

def main_preprocess():
    df_flow, df_speed, sensor_meta, common_sensors, common_ts = load_and_clean_data()
    
    X, y = build_features(df_flow, df_speed, common_ts)
    
    (X_train, y_train, ts_train,
     X_val, y_val, ts_val,
     X_test, y_test, ts_test) = split_series(X, y, common_ts)
    
    X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, x_scaler, y_scaler = standardize_data(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    train_flow_df = df_flow.loc[ts_train, common_sensors]
    train_speed_df = df_speed.loc[ts_train, common_sensors]
    sensor_meta_train = sensor_meta.reindex(common_sensors)
    A_list = build_multi_graphs(train_flow_df, train_speed_df, sensor_meta_train)

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s,
        batch_size=64,
        num_workers=4
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    np.savez(
        os.path.join(SAVE_DIR, "graph_mats24年.npz"),
        A_distance=A_list[0],
        A_flow_corr=A_list[1],
        A_speed_corr=A_list[2],
        A_region=A_list[3],
    )

    np.savez_compressed(
        os.path.join(SAVE_DIR, "dataset24年.npz"),
        X_train=X_train_s,
        X_val=X_val_s,
        X_test=X_test_s,
        y_train=y_train_s,
        y_val=y_val_s,
        y_test=y_test_s
    )

    with open(os.path.join(SAVE_DIR, "scalers24年.pkl"), "wb") as f:
        pickle.dump(
            {"x_scaler": x_scaler, "y_scaler": y_scaler},
            f
        )

    pd.Series(common_sensors).to_csv(
        os.path.join(SAVE_DIR, "common_sensors24年.csv"),
        index=False,
        header=["sensor_id"]
    )

    pd.Series(common_ts).to_csv(
        os.path.join(SAVE_DIR, "all_timestamps24年.csv"),
        index=False,
        header=["timestamp"]
    )
    pd.Series(ts_train).to_csv(
        os.path.join(SAVE_DIR, "train_timestamps24年.csv"),
        index=False,
        header=["timestamp"]
    )
    pd.Series(ts_val).to_csv(
        os.path.join(SAVE_DIR, "val_timestamps24年.csv"),
        index=False,
        header=["timestamp"]
    )
    pd.Series(ts_test).to_csv(
        os.path.join(SAVE_DIR, "test_timestamps24年.csv"),
        index=False,
        header=["timestamp"]
    )

    print(f"预处理结果保存至：{SAVE_DIR}")

    return {
        "df_flow": df_flow,
        "df_speed": df_speed,
        "sensor_meta": sensor_meta,
        "common_sensors": common_sensors,
        "common_ts": common_ts,
        "ts_train": ts_train,
        "ts_val": ts_val,
        "ts_test": ts_test,
        "X_train": X_train_s,
        "X_val": X_val_s,
        "X_test": X_test_s,
        "y_train": y_train_s,
        "y_val": y_val_s,
        "y_test": y_test_s,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "A_list": A_list,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader
    }

if __name__ == "__main__":
    preprocess_data = main_preprocess()

