# coding: utf-8
import math
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric as tg
import scipy.sparse as sp
import numpy as np

# GRU_GCN class
class GRU_GCN(torch.nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    method_name: str
    egcn_type: str

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRU_GCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # self.line1 = nn.Linear(input_dim,hidden_dim)
        # self.line2 = nn.Linear(input_dim,input_dim)
        
        self.GRU_node_layer = GRU_cell(input_dim, hidden_dim)
        self.GRU_edge_layer = GRU_cell(input_dim,hidden_dim)
        self.line1 = nn.Linear(hidden_dim,output_dim)
        self.GCN_init_weights = Parameter(torch.FloatTensor(hidden_dim, output_dim))
        self.reset_param(self.GCN_init_weights)
        # self.GCN_layer = GCNConv(hidden_dim,output_dim)

        # self.lin = nn.Linear(output_dim, input_dim, bias=False)
        
    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)
        
    def forward(self, A_list, Nodes_list):
        GCN_weights = self.GCN_init_weights
        # node = self.line1(Nodes_list)
        # edge = self.line2(A_list)
        node = self.GRU_node_layer(Nodes_list)
        edge = self.GRU_edge_layer(A_list)
        edge = self.line1(edge)
        
        node_emb = F.rrelu(edge.matmul(node.matmul(GCN_weights)))
        # edge = sp.coo_matrix(edge.detach().cpu().numpy(), dtype=np.float32)
        # # 
        # edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(edge)
        # emb = self.GCN_layer(node,edge_index.to(node.device), edge_weight.to(node.device))
        return node_emb

    def get_loss(self, sup_list, feat_list, gnd_tnr):
        # graphs = feed_dict["graphs"]
        # run gnn
        # graphs = graphs[:-1]
        # gnd = torch_geometric.utils.to_scipy_sparse_matrix(graphs[-1].edge_index).todense() 
        # final_emb = self.forward(graphs)
        final_emb = self.forward(sup_list, feat_list)

        self.graph_loss = 0
        loss = torch.norm((final_emb - gnd_tnr), p='fro')**2
        self.graph_loss += loss
        return self.graph_loss  

class GRU_cell(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(GRU_cell, self).__init__()
        self.hidden_size = hidden_size  # 隐层大小
        self.num_layers = 1  # gru层数
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为1
        self.gru = nn.GRU(feature_size, hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0] # 获取批次大小
        
        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            # print(x.shape)
            # h_0 = x 
        else:
            h_0 = hidden
            
        # GRU运算
        output, h_0 = self.gru(x, h_0)
        
        # 获取GRU输出的维度信息
        batch_size, timestep, hidden_size = output.shape  
            
        # 将output变成 batch_size * timestep, hidden_dim
        # output = output.reshape(-1, hidden_size)
        
        # 全连接层
        # output = self.fc(output)  # 形状为batch_size * timestep, 1
        
        # 转换维度，用于输出
        # output = output.reshape(timestep, batch_size, -1)
        output = output.transpose(0,1)
        
        # 我们只需要返回最后一个时间片的数据即可
        return output[-1]
    