"""Shared graph convolution building blocks (GCN, GAT) and the BaseGCN / BaseGAT
sequence wrappers used by the graph baselines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(n_feat, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_out)
        self.dropout = dropout
    
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        return x

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a_l = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a_r = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        a_l = torch.matmul(h, self.a_l)
        a_r = torch.matmul(h, self.a_r)
        e = self.leakyrelu(a_l.permute(0, 2, 1) + a_r)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return torch.relu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_out, dropout, alpha=0.1, nheads=2):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(n_hid * nheads, n_out, dropout=dropout, alpha=alpha, concat=False)
    
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.out_att(x, adj)

class BaseGCN(nn.Module):
    def __init__(self, input_len, output_len, num_nodes, adj, in_channels=1, hidden_channels=64, dropout=0.2):
        super(BaseGCN, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_nodes = num_nodes
        self.adj = adj
        self.gcn = GCN(n_feat=input_len, n_hid=hidden_channels, n_out=output_len, dropout=dropout)
    
    def forward(self, x):
        """
        x: (B, input_len, N)
        return: (B, output_len, N)
        """
        B, T, N = x.shape
        adj = self.adj.to(x.device)
        x = x.permute(0, 2, 1)  # (B, N, input_len)
        x = self.gcn(x, adj)  # (B, N, output_len)
        x = x.permute(0, 2, 1)  # (B, output_len, N)
        return x

class BaseGAT(nn.Module):
    def __init__(self, input_len, output_len, num_nodes, adj, in_channels=1, hidden_channels=64, dropout=0.2, nheads=2):
        super(BaseGAT, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_nodes = num_nodes
        self.adj = adj
        self.gat = GAT(n_feat=input_len, n_hid=hidden_channels, n_out=output_len, dropout=dropout, nheads=nheads)
    
    def forward(self, x):
        """
        x: (B, input_len, N)
        return: (B, output_len, N)
        """
        B, T, N = x.shape
        adj = self.adj.to(x.device)
        x = x.permute(0, 2, 1)  # (B, N, input_len)
        x = self.gat(x, adj)  # (B, N, output_len)
        x = x.permute(0, 2, 1)  # (B, output_len, N)
        return x
