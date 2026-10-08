"""Transformer baseline for multivariate time-series forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool, device=device),
                diagonal=1
            )

    @property
    def mask(self):
        return self._mask

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, Q, mask=None):
        K, V = Q, Q
        Q = self.query(Q)
        K = self.key(K)
        V = self.value(V)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(K.size(-1), dtype=torch.float32, device=Q.device)
        )

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = SelfAttention(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x, mask=None):
        x2 = self.self_attn(x, mask=mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Transformer_TS(nn.Module):
    def __init__(self, input_len, output_len, num_nodes, d_model=64, nhead=2, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_nodes = num_nodes
        self.d_model = d_model

        self.embedding = nn.Linear(num_nodes, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.fc = nn.Sequential(
            nn.Linear(d_model * input_len, 256),
            nn.GELU(),
            nn.Linear(256, output_len * num_nodes)
        )

    def _normalize_input(self, x):
        if x.dim() == 3:
            if x.shape[1] == self.input_len and x.shape[2] == self.num_nodes:
                return x
            if x.shape[1] == self.num_nodes and x.shape[2] == self.input_len:
                return x.transpose(1, 2).contiguous()
            raise ValueError(f"3维输入形状不对：{tuple(x.shape)}")

        if x.dim() != 4:
            raise ValueError(f"只接受3维或4维输入，但收到：{tuple(x.shape)}")

        shape = list(x.shape)

        try:
            t_axis = shape.index(self.input_len)
        except ValueError:
            raise ValueError(f"找不到时间维 input_len={self.input_len}，输入形状：{tuple(x.shape)}")

        try:
            n_axis = shape.index(self.num_nodes)
        except ValueError:
            raise ValueError(f"找不到节点维 num_nodes={self.num_nodes}，输入形状：{tuple(x.shape)}")

        extra_axes = [ax for ax in range(4) if ax not in (0, t_axis, n_axis)]
        if len(extra_axes) > 1:
            raise ValueError(f"多余维度太多，无法安全处理：{tuple(x.shape)}")

        for ax in sorted(extra_axes, reverse=True):
            x = x.select(dim=ax, index=0)

        if x.dim() == 3:
            if x.shape[1] == self.input_len and x.shape[2] == self.num_nodes:
                return x
            if x.shape[1] == self.num_nodes and x.shape[2] == self.input_len:
                return x.transpose(1, 2).contiguous()

        raise ValueError(f"4维输入处理后仍无法整理成 (B, T, N)：{tuple(x.shape)}")

    def forward(self, x):
        x = self._normalize_input(x)

        B, T, N = x.shape
        if T != self.input_len or N != self.num_nodes:
            raise ValueError(f"输入整理后维度不匹配：当前 {tuple(x.shape)}，期望 (B, {self.input_len}, {self.num_nodes})")

        x = x.float().contiguous()

        x = self.embedding(x)  # (B, T, N) -> (B, T, d_model)
        x = self.pos_encoding(x)

        mask = TriangularCausalMask(B, T, device=x.device).mask
        for layer in self.encoder_layers:
            x = layer(x, mask=mask)

        x = x.reshape(B, -1)   # (B, T*d_model)
        if x.shape[1] != self.input_len * self.d_model:
            raise RuntimeError(f"展开后维度异常：当前 {x.shape[1]}，期望 {self.input_len * self.d_model}")

        x = self.fc(x)
        x = x.reshape(B, self.output_len, N)
        return x
