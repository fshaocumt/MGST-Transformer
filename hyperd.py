"""HyperD baseline: hybrid periodic-pattern decomposition with a dual-view loss.
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
    
    def forward(self, Q):
        K, V = Q, Q
        Q = self.query(Q)
        K = self.key(K)
        V = self.value(V)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(K.size(-1), dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

class stfe(nn.Module):
    def __init__(self, num_nodes, seq_len, pred_len, embed_size, hidden_size, fc_hidden_size):
        super(stfe, self).__init__()
        self.scale = 0.02
        self.feature_size = num_nodes
        self.seq_length = seq_len
        self.sparsity_threshold = 0.01
        self.embeddings = nn.Parameter(torch.randn(1, embed_size))
        
        self.spatial_r1 = nn.Parameter(self.scale * torch.randn(embed_size, hidden_size))
        self.spatial_i1 = nn.Parameter(self.scale * torch.randn(embed_size, hidden_size))
        self.spatial_rb1 = nn.Parameter(self.scale * torch.randn(hidden_size))
        self.spatial_ib1 = nn.Parameter(self.scale * torch.randn(hidden_size))
        self.spatial_r2 = nn.Parameter(self.scale * torch.randn(hidden_size, embed_size))
        self.spatial_i2 = nn.Parameter(self.scale * torch.randn(hidden_size, embed_size))
        self.spatial_rb2 = nn.Parameter(self.scale * torch.randn(embed_size))
        self.spatial_ib2 = nn.Parameter(self.scale * torch.randn(embed_size))
        
        self.temporal_r1 = nn.Parameter(self.scale * torch.randn(embed_size, hidden_size))
        self.temporal_i1 = nn.Parameter(self.scale * torch.randn(embed_size, hidden_size))
        self.temporal_rb1 = nn.Parameter(self.scale * torch.randn(hidden_size))
        self.temporal_ib1 = nn.Parameter(self.scale * torch.randn(hidden_size))
        self.temporal_r2 = nn.Parameter(self.scale * torch.randn(hidden_size, embed_size))
        self.temporal_i2 = nn.Parameter(self.scale * torch.randn(hidden_size, embed_size))
        self.temporal_rb2 = nn.Parameter(self.scale * torch.randn(embed_size))
        self.temporal_ib2 = nn.Parameter(self.scale * torch.randn(embed_size))
        
        self.fc = nn.Sequential(
            nn.Linear(seq_len * embed_size, fc_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(fc_hidden_size, pred_len)
        )
    
    def tokenEmb(self, x):
        x = x.unsqueeze(3)
        y = self.embeddings
        return x * y
    
    def C_MLP_s(self, x):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.C_MLP(x, self.spatial_r1, self.spatial_i1, self.spatial_r2, self.spatial_i2, 
                       self.spatial_rb1, self.spatial_rb2, self.spatial_ib1, self.spatial_ib2)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        return x
    
    def C_MLP_t(self, x):
        x = x.transpose(1, 2)
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.C_MLP(x, self.temporal_r1, self.temporal_i1, self.temporal_r2, self.temporal_i2,
                       self.temporal_rb1, self.temporal_rb2, self.temporal_ib1, self.temporal_ib2)
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
        x = x.transpose(1, 2)
        return x
    
    def C_MLP(self, x, r1, i1, r2, i2, rb1, rb2, ib1, ib2):
        o1_real = F.relu(
            torch.einsum('bijd,df->bijf', x.real, r1) - 
            torch.einsum('bijd,df->bijf', x.imag, i1) + rb1
        )
        o1_imag = F.relu(
            torch.einsum('bijd,df->bijf', x.imag, r1) + 
            torch.einsum('bijd,df->bijf', x.real, i1) + ib1
        )
        o2_real = F.relu(
            torch.einsum('bijf,fd->bijd', o1_real, r2) - 
            torch.einsum('bijf,fd->bijd', o1_imag, i2) + rb2
        )
        o2_imag = F.relu(
            torch.einsum('bijf,fd->bijd', o1_imag, r2) + 
            torch.einsum('bijf,fd->bijd', o1_real, i2) + ib2
        )
        y = torch.stack([o2_real, o2_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        return torch.view_as_complex(y)
    
    def forward(self, x):
        B, T, N = x.shape
        x = self.tokenEmb(x)
        bias = x
        x = self.C_MLP_s(x)
        x = self.C_MLP_t(x)
        x = x + bias
        x = self.fc(x.transpose(1, 2).reshape(B, N, -1)).permute(0, 2, 1)
        return x

def dual_view_loss(y, periodic, residual, F_low):
    y_fft = torch.fft.rfft(y, dim=1)
    low_fft = torch.zeros_like(y_fft)
    low_fft[:, :F_low] = y_fft[:, :F_low]
    high_fft = torch.zeros_like(y_fft)
    high_fft[:, F_low:] = y_fft[:, F_low:]
    low_time = torch.fft.irfft(low_fft, n=y.size(1), dim=1)
    high_time = torch.fft.irfft(high_fft, n=y.size(1), dim=1)
    loss_low = F.mse_loss(periodic, low_time)
    loss_high = F.mse_loss(residual, high_time)
    return loss_low, loss_high

class Hybrid_Periodic_Pattern(nn.Module):
    def __init__(self, period_len, num_nodes, adj, init_npy_path=None):
        super(Hybrid_Periodic_Pattern, self).__init__()
        self.period_len = period_len
        self.num_nodes = num_nodes
        self.A = adj
        self.linear1 = nn.Linear(num_nodes, num_nodes)
        self.linear2 = nn.Linear(2*num_nodes, num_nodes)
        self.attention_t = SelfAttention(num_nodes)
        self.attention_s = SelfAttention(period_len)
        
        if init_npy_path and os.path.exists(init_npy_path):
            init_data = np.load(init_npy_path)
            assert init_data.shape == (self.period_len, self.num_nodes)
            self.data = nn.Parameter(torch.from_numpy(init_data).float(), requires_grad=True)
        else:
            self.data = nn.Parameter(torch.randn(self.period_len, self.num_nodes), requires_grad=True)
    
    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.period_len
        data = self.data
        data = torch.einsum('ln,nv->lv', (data, self.A.to(data.device)))
        data = self.linear1(data)
        data = F.relu(data)
        data_t = self.attention_t(data)
        data_s = self.attention_s(data.transpose(0, 1)).transpose(0, 1)
        data = self.linear2(torch.cat([data_t, data_s], dim=-1))
        return data[gather_index.long()]

class HyperD(nn.Module):
    def __init__(self, seq_len, pred_len, num_nodes, adj, 
                 time_of_day_size=288, day_of_week_size=7,
                 embed_size=64, hidden_size=128, fc_hidden_size=128,
                 alpha=2, F_low=1, init_path_daily=None, init_path_weekly=None):
        super(HyperD, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.adj = adj
        self.alpha = alpha
        self.F_low = F_low
        self.daily_len = time_of_day_size
        self.weekly_len = day_of_week_size * self.daily_len
        
        self.daily_emb = Hybrid_Periodic_Pattern(
            period_len=self.daily_len, 
            num_nodes=self.num_nodes, 
            adj=self.adj,
            init_npy_path=init_path_daily
        )
        self.weekly_emb = Hybrid_Periodic_Pattern(
            period_len=self.weekly_len, 
            num_nodes=self.num_nodes, 
            adj=self.adj,
            init_npy_path=init_path_weekly
        )
        
        self.stfe = stfe(self.num_nodes, self.seq_len, self.pred_len, embed_size, hidden_size, fc_hidden_size)
    
    def forward(self, history_x, history_t, history_t_last):
        x = history_x  # (B, seq_len, N)
        
        index_daily = (history_t_last[:, 0] * self.daily_len).long()  # (B,)
        index_weekly = (history_t_last[:, 0] * self.daily_len + history_t_last[:, 1] * self.weekly_len).long()  # (B,)
        
        S_D_in = self.daily_emb(index_daily, self.seq_len)
        S_W_in = self.weekly_emb(index_weekly, self.seq_len)
        S_in = S_D_in + S_W_in
        
        residual_in = x - S_in
        residual_out = self.stfe(residual_in)
        
        S_D_out = self.daily_emb((index_daily + self.seq_len) % self.daily_len, self.pred_len)
        S_W_out = self.weekly_emb((index_weekly + self.seq_len) % self.weekly_len, self.pred_len)
        S_out = S_D_out + S_W_out
        
        y = residual_out + S_out
        
        if self.training:
            loss_low_y, loss_high_y = dual_view_loss(y, S_out, residual_out, self.F_low)
            loss = (loss_low_y + loss_high_y) * self.alpha
            return y, loss
        else:
            return y
