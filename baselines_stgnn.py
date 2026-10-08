"""Spatio-temporal graph baselines: STGCN, Graph WaveNet, DCRNN, ASTGCN and
STAEformer, together with the shared graph-convolution helpers.
"""

"""Spatio-temporal GNN baselines.

Every model below follows the equations of the cited paper.  The official
repositories are given so that the exact code can be swapped in if required --
the interface used here (x: [B, T, N, F] -> y: [B, theta, N], everything
standardised with the same train-only scaler) is identical to the official
implementations after their own preprocessing step.

  STGCN          Yu, Yin & Zhu, IJCAI 2018      10.24963/ijcai.2018/505
                 https://github.com/veritasyin/STGCN_IJCAI-18
  Graph WaveNet  Wu et al., IJCAI 2019          10.24963/ijcai.2019/264
                 https://github.com/nnzhan/Graph-WaveNet
  DCRNN          Li et al., ICLR 2018           arXiv:1707.01926
                 https://github.com/liyaguang/DCRNN
  ASTGCN         Guo et al., AAAI 2019          10.1609/aaai.v33i01.3301922
                 https://github.com/guoshnBJTU/ASTGCN
  STAEformer     Liu et al., CIKM 2023          10.1145/3583780.3615160
                 https://github.com/XDZhelheim/STAEformer

All of them receive the *distance graph* of the target dataset, top-k
sparsified and row-normalised exactly as for the proposed model; the standalone
adaptive adjacency of Graph WaveNet (E1 x E2) is learned on top of it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def sym_normalise(A, eps=1e-8):
    A = (A + A.transpose(0, 1)) / 2.0
    d = A.sum(1)
    inv = torch.pow(d + eps, -0.5)
    return inv.unsqueeze(1) * A * inv.unsqueeze(0)

def add_self_loops(A):
    return A + torch.eye(A.size(0), dtype=A.dtype, device=A.device)

def random_walk_norm(A, eps=1e-8):
    A = add_self_loops(A)
    return A / (A.sum(1, keepdim=True) + eps)

def scaled_laplacian(A):
    n = A.size(0)
    A = add_self_loops(A)
    d = torch.pow(A.sum(1), -0.5)
    L = torch.eye(n, dtype=A.dtype, device=A.device) - d.unsqueeze(1) * A * d.unsqueeze(0)
    lam = torch.linalg.eigvalsh(L).max()
    return 2.0 * L / lam - torch.eye(n, dtype=A.dtype, device=A.device)

def chebyshev_polynomials(L, K):
    n = L.size(0)
    out = [torch.eye(n, dtype=L.dtype, device=L.device), L]
    for _ in range(2, K):
        out.append(2.0 * torch.matmul(L, out[-1]) - out[-2])
    return out[:K]

def register_polynomials(module, polys):
    for i, p in enumerate(polys):
        module.register_buffer(f"cheb_{i}", p)
    return [getattr(module, f"cheb_{i}") for i in range(len(polys))]

def _nconv(x, A):
    return torch.einsum("ncvt,vw->ncwt", x, A).contiguous()

class TimeCollapseHead(nn.Module):

    def __init__(self, in_ch, pred_len, n_time):
        super().__init__()
        self.linear = nn.Linear(in_ch * n_time, pred_len)

    def forward(self, x):
        B, C, N, T = x.shape
        z = x.permute(0, 2, 1, 3).contiguous().reshape(B, N, C * T)
        return self.linear(z).permute(0, 2, 1).contiguous()      # (B,theta,N)

class TemporalGLU(nn.Module):

    def __init__(self, Kt, c_in, c_out):
        super().__init__()
        self.Kt = Kt
        self.pad = nn.ZeroPad2d((0, 0, Kt - 1, 0))          # pad the front of time
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(Kt, 1))
        self.res = None if c_in == c_out else nn.Conv2d(c_in, c_out, kernel_size=(1, 1))

    def forward(self, x):
        xp = self.pad(x)
        x_in = xp[:, :, self.Kt - 1:, :]
        P, Q = torch.split(self.conv(xp), self.conv.out_channels // 2, dim=1)
        r = x_in if self.res is None else self.res(x_in)
        return (P + r[:, : P.size(1), :, :]) * torch.sigmoid(Q)

class ChebGraphConv(nn.Module):

    def __init__(self, c_in, c_out, K):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(K, c_in, c_out))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(c_out))

    def forward(self, x, T_k):
        xt = x.permute(0, 1, 3, 2)                                   # (B,c,T,N)
        terms = torch.stack([torch.matmul(xt, m) for m in T_k], dim=2)  # (B,c,K,T,N)
        terms = terms.permute(0, 1, 2, 4, 3)                         # (B,c,K,N,T)
        out = torch.einsum("bcknt,kcd->bdnt", terms,
                           self.weight) + self.bias.view(1, -1, 1, 1)
        return out

class STConvBlock(nn.Module):
    def __init__(self, Kt, Ks, c_in, c_mid, c_out, T_k, dropout=0.1, n_time=12):
        super().__init__()
        self.tconv1 = TemporalGLU(Kt, c_in, c_mid)
        self.gconv = ChebGraphConv(c_mid, c_mid, Ks)
        self.T_k = T_k
        self.tconv2 = TemporalGLU(Kt, c_mid, c_out)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(n_time)

    def forward(self, x):
        x = self.tconv1(x)
        x = F.relu(self.gconv(x, self.T_k))
        x = self.dropout(x)
        x = self.tconv2(x)
        return self.dropout(self.layernorm(x))

class STGCN(nn.Module):

    def __init__(self, adj, pred_len=6, input_len=12, Ks=3, Kt=3,
                 channels=(32, 32, 64, 64), dropout=0.1):
        super().__init__()
        A = sym_normalise(torch.as_tensor(adj, dtype=torch.float32))
        self.register_buffer("A", A)
        self.T_k = register_polynomials(self, chebyshev_polynomials(scaled_laplacian(self.A), Ks))

        layer1, layer2, hidden1, hidden2 = channels
        self.blocks = nn.ModuleList([
            STConvBlock(Kt, Ks, 1, layer1, hidden1, self.T_k, dropout, input_len),
            STConvBlock(Kt, Ks, hidden1, layer2, hidden2, self.T_k, dropout, input_len),
        ])
        self.out_tconv = TemporalGLU(Kt, hidden2, hidden1)
        self.head = TimeCollapseHead(hidden1, pred_len, input_len)

    def forward(self, x):
        x = x[..., :1].permute(0, 3, 2, 1).contiguous()          # (B,1,N,T)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.out_tconv(x))

class MixProp(nn.Module):
    def __init__(self, c_in, c_out, order=2):
        super().__init__()
        self.order = order
        self.mlp = nn.Conv2d(c_in * order, c_out, kernel_size=(1, 1))

    def forward(self, x, supports):
        h, out = x, [x]
        for _ in range(1, self.order):
            h = sum(_nconv(h, s) for s in supports) / len(supports)
            out.append(h)
        return self.mlp(torch.cat(out, dim=1))

class DilatedInception(nn.Module):

    def __init__(self, c_in, c_out, kernel_set=(2, 3, 6, 7), dilation=1):
        super().__init__()
        self.kernel_set = tuple(kernel_set)
        self.dilation = dilation
        self.convs = nn.ModuleList([
            nn.Conv2d(c_in, c_out, kernel_size=(1, k), padding=(0, 0), dilation=(1, dilation))
            for k in self.kernel_set
        ])

    def forward(self, x):
        outs = []
        for conv, k in zip(self.convs, self.kernel_set):
            pad = (k - 1) * self.dilation
            xi = F.pad(x, (pad, 0, 0, 0)) if pad > 0 else x
            outs.append(conv(xi))
        return sum(outs) / len(outs)

class GWNetLayer(nn.Module):
    def __init__(self, kernel_set, dilation, residual_ch, conv_ch, skip_ch, order=2):
        super().__init__()
        self.conv_ch = conv_ch
        self.tconv = DilatedInception(residual_ch, 2 * conv_ch, kernel_set, dilation)
        self.gconv = MixProp(conv_ch, residual_ch, order)
        self.skip = nn.Conv2d(conv_ch, skip_ch, kernel_size=(1, 1))

    def forward(self, x, supports):
        residual = x
        filt, gate = torch.split(self.tconv(x), self.conv_ch, dim=1)
        x = torch.tanh(filt) * torch.sigmoid(gate)
        skip = self.skip(x)
        x = self.gconv(x, supports)
        return x + residual[:, : x.size(1), :, :], skip

class GraphWaveNet(nn.Module):
    """Wu et al., IJCAI 2019 -- gated dilated TCN + mix-hop graph propagation
    + learnable adaptive adjacency softmax(ReLU(E1 E2))."""

    def __init__(self, adj, pred_len=6, input_len=12, residual_ch=32, conv_ch=16,
                 skip_ch=32, end_ch=128, layers=4, kernel_set=(2, 3, 6, 7),
                 dropout=0.3, embed_dim=10, order=2):
        super().__init__()
        n = adj.shape[0]
        A = sym_normalise(torch.as_tensor(adj, dtype=torch.float32))
        self.register_buffer("A", A)

        self.nodevec1 = nn.Parameter(torch.randn(n, embed_dim) * 0.02)
        self.nodevec2 = nn.Parameter(torch.randn(embed_dim, n) * 0.02)

        self.start_conv = nn.Conv2d(1, residual_ch, kernel_size=(1, 1))
        self.layers = nn.ModuleList([
            GWNetLayer(kernel_set, 2 ** i, residual_ch, conv_ch, skip_ch, order)
            for i in range(layers)
        ])
        self.end_conv1 = nn.Conv2d(skip_ch, end_ch, kernel_size=(1, 1))
        self.head = TimeCollapseHead(end_ch, pred_len, input_len)
        self.dropout = nn.Dropout(dropout)

    def supports(self):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        return [self.A, adp]

    def forward(self, x):
        x = x[..., :1].permute(0, 3, 2, 1).contiguous()          # (B,1,N,T)
        x = self.start_conv(x)
        skip = None
        for layer in self.layers:
            x, s = layer(x, self.supports())
            skip = s if skip is None else skip + s
        return self.head(self.dropout(F.relu(self.end_conv1(F.relu(skip)))))

class DCGRUCell(nn.Module):
    """Diffusion-convolutional GRU cell.

    The three gates share one diffusion step over [X, H] (as in the official
    implementation) and then use separate linear heads; the candidate gate is
    conditioned on the gated state r * H, exactly as in the paper.
    """

    def __init__(self, input_dim, hidden_dim, supports):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.supports = supports
        self.gate = nn.Linear(input_dim + hidden_dim, 2 * hidden_dim)
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def diffuse(self, v):
        out = torch.zeros_like(v)
        for s in self.supports:
            out = out + torch.einsum("ij,bjd->bid", s, v)
        return out / len(self.supports)

    def forward(self, x, state):
        r_u = torch.sigmoid(self.gate(self.diffuse(torch.cat([x, state], dim=-1))))
        r, u = torch.split(r_u, self.hidden_dim, dim=-1)
        c = torch.tanh(self.candidate(self.diffuse(torch.cat([x, r * state], dim=-1))))
        return u * state + (1.0 - u) * c

class DCRNN(nn.Module):

    def __init__(self, adj, pred_len=6, hidden_dim=64, max_diffusion_step=2,
                 num_layers=1):
        super().__init__()
        A = torch.as_tensor(adj, dtype=torch.float32)
        fwd = [torch.matrix_power(random_walk_norm(A), k) for k in range(max_diffusion_step + 1)]
        bwd = [torch.matrix_power(random_walk_norm(A.transpose(0, 1)), k)
               for k in range(max_diffusion_step + 1)]
        self.supports = fwd + bwd
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        self.enc_cells = nn.ModuleList([
            DCGRUCell(1 if i == 0 else hidden_dim, hidden_dim, self.supports)
            for i in range(num_layers)
        ])
        self.dec_cell = DCGRUCell(1, hidden_dim, self.supports)
        self.out = nn.Linear(hidden_dim, 1)
        self.accepts_teacher = True

    def forward(self, x, y_true=None):
        x_seq = x[..., :1]                                        # (B,T,N,1)
        B, T, N, _ = x_seq.shape
        states = [torch.zeros(B, N, self.hidden_dim, dtype=x.dtype, device=x.device)
                  for _ in self.enc_cells]
        for t in range(T):
            inp = x_seq[:, t]
            for i, cell in enumerate(self.enc_cells):
                states[i] = cell(inp, states[i])
                inp = states[i]
        state = states[-1]

        dec_in = x_seq[:, -1]
        preds = []
        for tau in range(self.pred_len):
            state = self.dec_cell(dec_in, state)
            y_t = self.out(state)
            preds.append(y_t)
            if self.training and y_true is not None and torch.rand((), device=x.device) < 0.5:
                dec_in = y_true[:, tau].unsqueeze(-1)             # scheduled sampling
            else:
                dec_in = y_t
        return torch.stack(preds, dim=1).squeeze(-1)              # (B,theta,N)

class SpatialAttention(nn.Module):

    def __init__(self, c_in, hidden=64):
        super().__init__()
        self.W1 = nn.Linear(c_in, hidden)
        self.W2 = nn.Linear(hidden, hidden)
        self.W3 = nn.Linear(c_in, hidden)
        self.scale = hidden ** 0.5

    def forward(self, x):
        v = x.permute(0, 3, 2, 1)                                 # (B,T,N,c)
        lhs = self.W2(torch.tanh(self.W1(v)))                     # (B,T,N,h)
        rhs = self.W3(v).transpose(2, 3)                          # (B,T,h,N)
        e = torch.matmul(lhs, rhs) / self.scale
        return F.softmax(F.leaky_relu(e), dim=-1).mean(1)         # (B,N,N)

class TemporalAttention(nn.Module):

    def __init__(self, c_in, n_time, hidden=64):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_time, hidden), nn.Tanh(),
                                  nn.Linear(hidden, n_time))

    def forward(self, x):
        h = x.mean(dim=2)                                         # (B,N,T)
        e = torch.softmax(self.proj(h), dim=-1).unsqueeze(2)      # (B,N,1,T)
        return x + e * x

class ASTGCN(nn.Module):
    """Guo et al., AAAI 2019 -- recent component: temporal attention,
    spatial attention adjusted adjacency, Chebyshev graph convolution and a
    gated temporal convolution."""

    def __init__(self, adj, pred_len=6, input_len=12, K=3, channels=64, dropout=0.1):
        super().__init__()
        A = torch.as_tensor(adj, dtype=torch.float32)
        self.register_buffer("A", sym_normalise(A))
        self.T_k = register_polynomials(self, chebyshev_polynomials(scaled_laplacian(self.A), K))

        self.tatt = TemporalAttention(1, input_len)
        self.satt = SpatialAttention(1)
        self.gconv = ChebGraphConv(1, channels, K)
        self.tconv = TemporalGLU(3, channels, channels)
        self.out_tconv = TemporalGLU(3, channels, channels)
        self.head = TimeCollapseHead(channels, pred_len, input_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x[..., :1].permute(0, 3, 2, 1).contiguous()           # (B,1,N,T)
        x = self.tatt(x)
        adj = torch.matmul(self.satt(x).mean(0), self.A)          # (N,N)
        T_k = [adj] + [torch.matmul(adj, t) for t in self.T_k[1:]]
        x = F.relu(self.gconv(x, T_k))
        x = self.tconv(self.dropout(x))
        return self.head(self.out_tconv(x))

class STAEformer(nn.Module):
    """Liu et al., CIKM 2023 -- vanilla Transformer with adaptive spatio-temporal
    embeddings.  Temporal attention runs over the look-back axis with all nodes
    treated as batch elements; spatial attention runs over the node axis.
    Simplification: no FiLM gating of the embeddings."""

    def __init__(self, num_nodes, pred_len=6, input_len=12, d_model=64, nhead=4,
                 num_layers=3, ff=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.data_emb = nn.Linear(1, d_model)
        self.tod_emb = nn.Linear(4, d_model)                     # hour/dow sin-cos
        self.dow_emb = nn.Parameter(torch.randn(num_nodes, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=ff, dropout=dropout,
                                           batch_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.spatial = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x):
        B, T, N, Fdim = x.shape
        h = self.data_emb(x[..., :1])                            # (B,T,N,d)
        if Fdim >= 6:
            h = h + self.tod_emb(x[..., 2:6])
        h = h + self.dow_emb.unsqueeze(0).unsqueeze(0)

        zt = h.reshape(B * N, T, self.d_model)
        zt = self.temporal(zt).reshape(B, T, N, self.d_model)

        zs = zt.reshape(B * T, N, self.d_model)
        zs = self.spatial(zs).reshape(B, T, N, self.d_model)

        return self.head(zs[:, -1]).permute(0, 2, 1).contiguous()  # (B,theta,N)
