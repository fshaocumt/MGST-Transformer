"""Temporal baselines: per-node LSTM / GRU / TCN / Transformer, STID, and the
parameter-free naive predictor.
"""

"""Temporal baselines -- corrected version.

Why the old numbers were not credible
-------------------------------------
The previous implementation encoded a whole network snapshot with

    SequenceEncoder:  Linear(num_nodes * in_dim, 64)    272 * 9 -> 64
    head:             Linear(64, pred_len * num_nodes)  64 -> 6 * 272

so every LSTM / GRU / TCN / Transformer had to squeeze 2448 numbers into a
64-dimensional vector and then expand that single vector into 1632 outputs.
Such a bottleneck cannot represent node-specific dynamics; the models collapse
onto something close to the training mean and lose to one-step persistence
(RMSE 211-219 vs. 124.9 for Naive2 on the 2018 set).

The standard formulation used below is node-independent sequence modelling with
shared parameters:

    (B, T, N, F)  ->  permute -> (B*N, T, F) -> RNN/TCN/Transformer -> head
                                            -> (B*N, theta) -> (B, theta, N)

which is exactly how sequence baselines are implemented in the traffic
forecasting literature (e.g. STID, BasicTS).  `use_node_emb` optionally adds a
learnable identity embedding per sensor (STID style).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import PositionalEncoding

class PerNodeTemporal(nn.Module):

    def __init__(self, num_nodes, in_dim, pred_len=6, hidden=128, num_layers=2,
                 dropout=0.1, kind="lstm", nhead=4, ff=256, use_node_emb=False,
                 emb_dim=16):
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        self.hidden = hidden
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.dropout = dropout
        self.kind = kind

        self.node_emb_dim = emb_dim if use_node_emb else 0
        if use_node_emb:
            self.node_emb = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.02)

        in_width = in_dim + self.node_emb_dim

        if kind in ("lstm", "gru"):
            rnn_cls = nn.LSTM if kind == "lstm" else nn.GRU
            self.rnn = rnn_cls(input_size=in_width, hidden_size=hidden,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        elif kind == "tcn":
            self.tcn = nn.Sequential(*[
                _TemporalBlock(in_width if i == 0 else hidden, hidden,
                               kernel_size=3, dilation=2 ** i, dropout=dropout)
                for i in range(num_layers)
            ])
        elif kind == "transformer":
            self.input_proj = nn.Linear(in_width, hidden)
            self.pos_enc = PositionalEncoding(hidden)
            layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead,
                                               dim_feedforward=ff, dropout=dropout,
                                               batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        else:
            raise ValueError(f"unknown kind: {kind}")

        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, pred_len),
        )

    def _encode(self, x):
        B, T, N, Fdim = x.shape
        h = x.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, Fdim)
        if self.node_emb_dim:
            emb = self.node_emb.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)
            emb = emb.reshape(B * N, T, -1)
            h = torch.cat([h, emb], dim=-1)

        if self.kind in ("lstm", "gru"):
            out, state = self.rnn(h)
            rep = state[0][-1] if self.kind == "lstm" else state[-1]
        elif self.kind == "tcn":
            h = h.transpose(1, 2)                    # (B*N, width, T)
            rep = self.tcn(h)[:, :, -1]              # last observed step
        else:
            h = self.pos_enc(self.input_proj(h))
            rep = self.encoder(h)[:, -1, :]
        return rep

    def forward(self, x):
        B = x.shape[0]
        rep = self._encode(x)                        # (B*N, hidden)
        y = self.head(rep)                           # (B*N, theta)
        return y.reshape(B, self.num_nodes, self.pred_len).permute(0, 2, 1).contiguous()

class _TemporalBlock(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.dropout(out)
        out = F.relu(self.conv2(out))
        out = self.dropout(out)
        return F.relu(self.norm(out + self.downsample(x)))

class STID(nn.Module):
    def __init__(self, num_nodes, pred_len=6, input_len=12, in_dim=1,
                 hidden=128, num_layers=4, spatial_dim=64, temporal_dim=64,
                 time_feat_dim=4, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        self.input_len = input_len
        self.time_feat_dim = time_feat_dim
        self.spatial_emb = nn.Parameter(torch.randn(num_nodes, spatial_dim) * 0.02)
        self.temporal_emb = nn.Parameter(torch.randn(input_len, temporal_dim) * 0.02)

        wide = input_len * (1 + temporal_dim) + spatial_dim + time_feat_dim
        layers = []
        prev = wide
        for _ in range(num_layers):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        layers.append(nn.Linear(prev, pred_len))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        B, T, N, Fdim = x.shape
        xt = x.permute(0, 2, 1, 3).contiguous()               # (B,N,T,F)
        M = B * N
        obs = xt[..., 0].reshape(M, T, 1)                    # flow channel
        te = self.temporal_emb[-T:].unsqueeze(0).expand(M, T, -1)
        flow = torch.cat([obs, te], dim=-1).reshape(M, T * (1 + te.size(-1)))
        se = self.spatial_emb.unsqueeze(0).expand(B, N, -1).reshape(M, -1)
        parts = [flow, se]
        if self.time_feat_dim > 0:
            if Fdim >= 6:
                tfeat = xt[..., 2:6].mean(dim=2)             # hour/dow sin-cos
            else:
                tfeat = xt[..., : self.time_feat_dim].mean(dim=2)
            parts.append(tfeat.reshape(M, -1))
        h = torch.cat(parts, dim=-1)
        y = self.mlp(h)                                      # (M, theta)
        return y.reshape(B, N, self.pred_len).permute(0, 2, 1).contiguous()

def naive_predict(X, y, input_len=12, pred_len=6, mode="last"):
    """Rebuild the test windows and return predictions in scaled space.

    mode='last'  Naive2 persistence (repeat the most recent observation)
    mode='mean'  historical average of the look-back window
    """
    import numpy as np

    T = X.shape[0]
    idx = np.arange(T - input_len - pred_len + 1)
    if mode == "last":
        base = X[idx + input_len - 1][:, :, 0:1]             # (W,N,1) flow channel
        return np.repeat(base, pred_len, axis=1)             # (W,theta,N)
    base = X[:, :, 0]
    win = np.stack([base[i: i + input_len] for i in idx], 0)  # (W,phi,N)
    return np.repeat(win.mean(1)[:, None, :], pred_len, axis=1)

def naive_metrics(data, y_scaler, mode="last"):
    from .common import compute_metrics, inverse_transform_3d
    X_test, y_test = data["X_test"], data["y_test"]
    input_len, pred_len = data["input_len"], data["pred_len"]
    pred = naive_predict(X_test, y_test, input_len, pred_len, mode)
    trues = np.stack([y_test[i + input_len: i + input_len + pred_len]
                      for i in range(len(X_test) - input_len - pred_len + 1)], 0)
    p = inverse_transform_3d(y_scaler, pred)
    t = inverse_transform_3d(y_scaler, trues)
    overall = compute_metrics(t, p)
    per_horizon = {f"H{h + 1}": compute_metrics(t[:, h, :], p[:, h, :])
                   for h in range(pred_len)}
    return overall, per_horizon, p, t
