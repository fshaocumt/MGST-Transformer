"""STGNN baseline: graph convolution layers with a DenseNet-style temporal block.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate
    
    def forward(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class STGNN(nn.Module):
    def __init__(self, input_len, output_len, num_nodes, adj, 
                 in_channels=1, hidden_channels=32, growth_rate=12, 
                 num_layers=4, dropout=0.2):
        super(STGNN, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_nodes = num_nodes
        self.adj = adj

        self.gcn = GCN(input_len, hidden_channels, hidden_channels, dropout)

        self.conv_in = nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1)
        self.dense_block = _DenseBlock(
            num_layers=num_layers,
            num_input_features=hidden_channels,
            bn_size=4,
            growth_rate=growth_rate,
            drop_rate=dropout
        )
        
        num_features = hidden_channels + num_layers * growth_rate
        self.conv_out = nn.Conv2d(num_features, output_len, kernel_size=1)
        self.fc = nn.Linear(num_nodes, num_nodes)
    
    def forward(self, x):
        B, T, N = x.shape
        adj = self.adj.to(x.device)

        x_gcn = x.permute(0, 2, 1)
        x_gcn = self.gcn(x_gcn, adj)

        x_t = x.unsqueeze(1)
        x_t = self.conv_in(x_t)
        x_t = self.dense_block(x_t)

        x_out = self.conv_out(x_t)
        x_out = x_out.mean(dim=2)
        x_out = self.fc(x_out)
        return x_out
