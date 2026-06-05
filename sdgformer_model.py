import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

# ===================== 1. 基础模块 =====================
class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
    def forward(self, x):
        return self.token_embed(x)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()

class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)
    def forward(self, lap_mx):
        return self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ===================== 2. ST Encoder（简化版） =====================
class STEncoder(nn.Module):
    def __init__(self, dim, num_nodes, input_window):
        super().__init__()
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, 10))
        self.Linear3 = nn.Linear(dim, dim//2)
        self.Linear2 = nn.Linear(dim, dim//2)
        self.Linear1 = nn.Linear(dim, dim//2)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)
        self.t_q_conv = nn.Conv2d(dim, dim//4, kernel_size=1)
        self.t_k_conv = nn.Conv2d(dim, dim//4, kernel_size=1)
        self.scale = (dim//4) ** -0.5
        self.att1 = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.att2 = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.attn1 = nn.Parameter(torch.tensor([0.6]))
        self.attn2 = nn.Parameter(torch.tensor([0.4]))
        self.attn3 = nn.Parameter(torch.tensor([0.4]))
        self.threshold_param = nn.Parameter(torch.rand(1))
    
    def create_adaptive_high_freq_mask(self, x_fft):
        B, j, _, _ = x_fft.shape
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        flat_energy = energy.view(B, j, -1)
        median_energy = flat_energy.median(dim=2, keepdim=True)[0]
        median_energy = median_energy.view(B, j, 1)
        normalized_energy = energy / (median_energy + 1e-6)
        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        return adaptive_mask.unsqueeze(-1)
    
    def forward(self, x):
        B, T, N, D = x.shape
        
        # 空间注意力
        Ad = F.softmax((x @ x.permute(0, 1, 3, 2))[-1, T//2, :, :], dim=-1)
        Aad = F.softmax(F.relu(torch.matmul(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1) + Ad
        x1 = x.reshape(-1, N, D)
        mask1 = torch.zeros(N, N, device=x.device)
        mask2 = torch.zeros(N, N, device=x.device)
        mask3 = torch.zeros(N, N, device=x.device)
        index = torch.topk(Aad, k=int(N * 0.6), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, Aad, torch.full_like(Aad, 0))
        index = torch.topk(Aad, k=int(N * 0.5), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, Aad, torch.full_like(Aad, 0))
        index = torch.topk(Aad, k=int(N * 0.4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, Aad, torch.full_like(Aad, 0))
        out1 = self.Linear1((attn1 @ x1).reshape(B, T, N, D))
        out2 = self.Linear2((attn2 @ x1).reshape(B, T, N, D))
        out3 = self.Linear3((attn3 @ x1).reshape(B, T, N, D))
        ZS = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3
        
        # 时间注意力
        input = x.permute(0, 2, 1, 3)
        f = torch.fft.rfft(input, dim=2, norm='ortho')
        freq_mask = self.create_adaptive_high_freq_mask(f)
        low_pass = f * freq_mask
        high_pass = 1 - freq_mask * f
        low_pass = torch.fft.irfft(low_pass, n=T, dim=2, norm='ortho')
        high_pass = torch.fft.irfft(high_pass, n=T, dim=2, norm='ortho')
        high_pass = high_pass.permute(0, 2, 1, 3)
        low_pass = low_pass.permute(0, 2, 1, 3)
        x_l = self.att1(low_pass).permute(0, 3, 1, 2) + x.permute(0, 3, 1, 2)
        x_h = self.att2(high_pass).permute(0, 3, 1, 2) + x.permute(0, 3, 1, 2)
        t_ql = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_kl = self.t_k_conv(x_l).permute(0, 3, 2, 1)
        t_attnl = (t_ql @ t_kl.transpose(-2, -1)) * self.scale
        t_attnl = F.softmax(t_attnl, dim=-1)
        t_l = (t_attnl @ t_kl).transpose(2, 3).reshape(B, N, T, -1).transpose(1, 2)
        ZT = torch.cat((t_l, t_l), dim=-1)  # 简化
        
        Z = torch.cat((ZS, ZT), dim=-1)
        x = self.proj(Z)
        x = self.proj_drop(x)
        return x

class STEncoderBlock(nn.Module):
    def __init__(self, dim, num_nodes, input_window, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.st_attn = STEncoder(dim, num_nodes, input_window)
        self.drop_path = nn.Identity()  # 简化
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
    
    def forward(self, x):
        x = x + self.drop_path(self.st_attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ===================== 3. SDGFormer主模型 =====================
class SDGFormer(nn.Module):
    def __init__(self, input_window, output_window, num_nodes, feature_dim=1, 
                 embed_dim=64, enc_depth=3, lape_dim=8):
        super().__init__()
        self.input_window = input_window
        self.output_window = output_window
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        
        # 嵌入层
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)
        self.spatial_embedding = LaplacianPE(lape_dim, embed_dim)
        
        # 编码器
        self.encoder_blocks = nn.ModuleList([
            STEncoderBlock(embed_dim, num_nodes, input_window)
            for _ in range(enc_depth)
        ])
        
        # 跳跃连接
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(embed_dim, 256, kernel_size=1)
            for _ in range(enc_depth)
        ])
        
        # 输出层
        self.end_conv1 = nn.Conv2d(input_window, output_window, kernel_size=1)
        self.end_conv2 = nn.Conv2d(256, 1, kernel_size=1)
    
    def forward(self, x, lap_mx=None):
        """
        x: (B, input_window, N, feature_dim)
        lap_mx: (N, lape_dim) 拉普拉斯位置编码
        """
        B, T, N, D = x.shape
        
        # 生成默认拉普拉斯编码
        if lap_mx is None:
            lap_mx = torch.randn(N, 8, device=x.device)
        
        # 嵌入
        x = self.value_embedding(x)
        x += self.position_encoding(x)
        x += self.spatial_embedding(lap_mx)
        
        # 编码
        skip = 0
        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x)
            skip += self.skip_convs[i](x.permute(0, 3, 2, 1))
        
        # 输出
        skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))
        return skip.permute(0, 3, 2, 1).squeeze(-1)