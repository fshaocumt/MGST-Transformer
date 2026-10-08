"""MGST-Transformer model definition.
Multi-graph convolution branches (distance / flow correlation / speed
correlation / regional) fused by a node-wise adaptive gate, followed by a
per-node Transformer encoder-decoder for multi-step forecasting.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]

class MultiGraphConv(nn.Module):
    """One multi-graph spatial layer.

    fusion_mode:
        'global'  static scalar per graph        (the old behaviour, ablation)
        'sample'  one weight vector per (batch element, look-back step)
        'node'    one weight vector per node, per step   <-- default
    """

    def __init__(self, in_dim, out_dim, num_graphs=4, dropout=0.1,
                 fusion_mode="node", gate_dim=32, ctx_dim=0, tau=1.0,
                 collect_alpha=False):
        super().__init__()
        self.num_graphs = num_graphs
        self.fusion_mode = fusion_mode
        self.ctx_dim = ctx_dim
        self.tau = tau
        self.collect_alpha = collect_alpha

        self.proj = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_graphs)])

        if fusion_mode == "global":
            self.graph_logits = nn.Parameter(torch.zeros(num_graphs))
        else:
            self.register_parameter("graph_logits", None)

        if fusion_mode == "sample":
            self.pool_proj = nn.Linear(out_dim, out_dim)
            self.gate_mlp = nn.Sequential(
                nn.Linear(in_dim + num_graphs * out_dim + ctx_dim, gate_dim),
                nn.GELU(),
                nn.Linear(gate_dim, num_graphs),
            )
        if fusion_mode == "node":
            self.W_h = nn.Linear(in_dim, gate_dim, bias=False)
            self.W_z = nn.Linear(out_dim, gate_dim, bias=False)
            if ctx_dim > 0:
                self.W_e = nn.Linear(ctx_dim, gate_dim, bias=False)
            else:
                self.register_parameter("W_e", None)
            self.b = nn.Parameter(torch.zeros(gate_dim))
            self.w = nn.Linear(gate_dim, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

        self.last_alpha = None

    # Node-wise adaptive gate (Eq. 7): a separate softmax over the K graphs is
    # computed for every node i and time step t, so alpha_{k,i,t} varies with
    # the current node state instead of being a static per-graph scalar.
    def _gate_node(self, x_bt, Z, ctx):
        base = self.W_h(x_bt)                                   # (M,N,g)
        if self.W_e is not None and ctx is not None:
            base = base + self.W_e(ctx).unsqueeze(1)            # (M,N,g)
        score = self.W_z(Z) + (base + self.b).unsqueeze(1)      # (M,K,N,g)
        score = self.w(torch.tanh(score)).squeeze(-1)           # (M,K,N)
        return torch.softmax(score / self.tau, dim=1)           # softmax over K

    # Sample-wise gate: one weight vector per (batch, time-step) sample.
    def _gate_sample(self, x_bt, Z, ctx):
        h_bar = x_bt.mean(dim=1)                                # (M,d_in)
        z_bar = Z.mean(dim=2)                                   # (M,K,d)
        parts = [h_bar, z_bar.flatten(1)]
        if self.ctx_dim > 0:
            parts.append(ctx if ctx is not None
                         else torch.zeros_like(z_bar[:, 0, :self.ctx_dim]))
        logits = self.gate_mlp(torch.cat(parts, dim=-1))        # (M,K)
        return torch.softmax(logits / self.tau, dim=-1)

    def forward(self, x, A_list, ctx=None):
        B, T, N, Din = x.shape
        M = B * T
        x_bt = x.reshape(M, N, Din)

        A2 = torch.cat(A_list, dim=0)                           # (K*N, N)
        agg = torch.matmul(A2, x_bt).view(M, self.num_graphs, N, Din)

        outs = []
        for k in range(self.num_graphs):
            outs.append(F.relu(self.proj[k](agg[:, k])))        # (M,N,d)
        Z = torch.stack(outs, dim=1)                            # (M,K,N,d)

        if ctx is not None:
            ctx = ctx.reshape(M, -1)                            # (M,d_t)

        if self.fusion_mode == "global":
            # One learnable scalar weight per graph (static across nodes and time).
            alpha = torch.softmax(self.graph_logits, dim=0)     # (K,)
            fused = torch.einsum("k,mknd->mnd", alpha, Z)
            if self.collect_alpha:
                self.last_alpha = alpha.detach().expand(M, N, self.num_graphs).permute(0, 2, 1)
        elif self.fusion_mode == "sample":
            alpha = self._gate_sample(x_bt, Z, ctx)             # (M,K)
            fused = torch.einsum("mk,mknd->mnd", alpha, Z)
            if self.collect_alpha:
                self.last_alpha = alpha.detach().unsqueeze(2).expand(-1, -1, N)
        else:
            alpha = self._gate_node(x_bt, Z, ctx)               # (M,K,N)
            fused = torch.einsum("mkn,mknd->mnd", alpha, Z)
            if self.collect_alpha:
                self.last_alpha = alpha.detach()

        fused = self.dropout(fused)
        out = self.norm(fused + self.residual(x_bt))
        return out.reshape(B, T, N, -1)

class TemporalSeq2SeqTransformer(nn.Module):
    def __init__(self, d_model, pred_len, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.pos_enc = PositionalEncoding(d_model)

        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=dim_feedforward,
                                         dropout=dropout, batch_first=True)
        dec = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=dim_feedforward,
                                         dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(dec, num_layers=num_layers)

        self.future_queries = nn.Parameter(torch.randn(pred_len, d_model) * 0.02)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        B, T, N, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, D)
        memory = self.encoder(self.pos_enc(x))

        tgt = self.pos_enc(self.future_queries.unsqueeze(0).expand(B * N, -1, -1))
        dec = self.decoder(tgt=tgt, memory=memory)
        y = self.out_proj(dec).squeeze(-1)
        return y.reshape(B, N, self.pred_len).permute(0, 2, 1).contiguous()

class MGSTTransformer(nn.Module):
    def __init__(self, in_dim, num_nodes, d_model=64, gnn_layers=2, nhead=4,
                 num_layers=2, dim_feedforward=128, dropout=0.1, pred_len=6,
                 num_graphs=4, use_spatial=True, fusion_mode="node",
                 gate_dim=32, tau=1.0, n_time_feat=7, ctx_dim=16,
                 collect_alpha=False):
        super().__init__()
        self.n_time_feat = n_time_feat
        self.ctx_dim = ctx_dim
        self.pred_len = pred_len
        self.use_spatial = use_spatial
        self.input_proj = nn.Linear(in_dim, d_model)

        self.use_time_ctx = (in_dim > n_time_feat) and (ctx_dim > 0)
        if self.use_time_ctx:
            self.time_proj = nn.Linear(n_time_feat, ctx_dim)

        if use_spatial:
            self.spatial_layers = nn.ModuleList([
                MultiGraphConv(d_model, d_model, num_graphs=num_graphs,
                               dropout=dropout, fusion_mode=fusion_mode,
                               gate_dim=gate_dim, ctx_dim=ctx_dim if self.use_time_ctx else 0,
                               tau=tau, collect_alpha=collect_alpha)
                for _ in range(gnn_layers)
            ])
        else:
            self.spatial_layers = nn.ModuleList()

        self.temporal_predictor = TemporalSeq2SeqTransformer(
            d_model=d_model, pred_len=pred_len, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout)

    def _temporal_context(self, x):
        tfeat = x[..., -self.n_time_feat:].mean(dim=2)          # (B,T,n_time_feat)
        return torch.tanh(self.time_proj(tfeat))                # (B,T,ctx_dim)

    def forward(self, x, A_list=None):
        h = self.input_proj(x)                                  # (B,T,N,d)
        if self.use_time_ctx:
            ctx = self._temporal_context(x)
        else:
            ctx = None
        if self.use_spatial and A_list is not None and len(A_list) > 0:
            for layer in self.spatial_layers:
                h = layer(h, A_list, ctx)
        # Temporal module: reshape (B,T,N,d) -> (B*N,T,d) so that each sensor is
        # modelled as one sequence; all cross-node interaction already happens
        # in the graph convolution above.
        return self.temporal_predictor(h)

    @torch.no_grad()
    def collect_fusion_weights(self, loader, device, A_list):
        """Return alpha (W,K,N) over the test set and the matching timestamps index.

        Used to produce the evidence figures requested by the reviewers:
        mean alpha per hour of day, dispersion across sensors, etc.
        """
        import numpy as np  # lazy: keeps the forward path torch-only

        self.eval()
        old = [l.collect_alpha for l in self.spatial_layers]
        for l in self.spatial_layers:
            l.collect_alpha = True

        alphas, hours = [], []
        for x, _ in loader:
            x = x.to(device)
            _ = self.forward(x, A_list)
            a = self.spatial_layers[-1].last_alpha
            if a is None:  # global mode -> broadcast
                continue
            if a.dim() == 2:  # sample mode (M,K)
                a = a.unsqueeze(-1)
            B, T, N, _ = x.shape
            a = a.reshape(B, T, a.shape[1], -1)                 # (B,T,K,N)
            alphas.append(a[:, -1].detach().cpu().numpy())      # last look-back step
            hours.append(x[:, -1, 0, 2:4].detach().cpu().numpy())  # hour sin/cos
        for l, o in zip(self.spatial_layers, old):
            l.collect_alpha = o
        if not alphas:
            return None, None
        return np.concatenate(alphas, 0), np.concatenate(hours, 0)
