import torch
import torch.nn as nn

from torch_scatter import scatter
from torch_geometric.nn import MessagePassing, global_mean_pool

from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax as sparse_softmax
from torch_geometric.utils import remove_self_loops, add_self_loops


class GATv2Layer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, head_num=8, head_dim=8, aggr='add'):
        super().__init__(aggr=aggr)

        self.head_num, self.head_dim = head_num, head_dim

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        self.lin_l = nn.Linear(emb_dim, head_num * head_dim)
        self.lin_r = nn.Linear(emb_dim, head_num * head_dim)
        self.lin_edge_attr = nn.Linear(edge_dim, emb_dim)
        self.attention_layer = nn.Parameter(torch.empty(1, head_num, head_dim), requires_grad=True)
        
        self.upd_h, self.upd_agg = nn.Linear(emb_dim, emb_dim), nn.Linear(emb_dim, emb_dim)

        self.mlp_upd = nn.ModuleList()
        upd_layers = [nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()]
        for layer in upd_layers:
            self.mlp_upd.append(layer)

        self.mlp_msg = nn.ModuleList()
        msg_layers = [nn.Linear(emb_dim + 1, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()]
        for layer in msg_layers:
            self.mlp_msg.append(layer)
        

    def forward(self, x, pos, edge_index, edge_attr, size=None):
        out = self.propagate(edge_index, x=x, size=size, pos=pos)
    
        return out

    def message(self, x_j, x_i, pos_j, pos_i, index, ptr, size_i):
        distance = torch.sum((pos_j - pos_i) ** 2, dim=-1).unsqueeze(-1)
        distance_heads = distance.unsqueeze(1).repeat(1, self.head_num, 1)

        x_i = self.lin_l(x_i).view(-1, self.head_dim, self.head_num)
        x_j = self.lin_r(x_j).view(-1, self.head_dim, self.head_num)
        x = x_i + x_j
        
        # x = torch.concat([x, distance_heads], dim=-1)
        
        x = F.leaky_relu(x, negative_slope=0.2)
        
        alpha = (x * self.attention_layer).sum(dim=-1)
        
        alpha = sparse_softmax(alpha, index, ptr, size_i).unsqueeze(-1)
        
        msg = (x_j * alpha).view(-1, self.head_dim * self.head_num)

        msg = torch.concat([msg, distance], dim=-1)

        for layer in self.mlp_msg:
            msg = layer(msg)

        return msg
    

    def aggregate(self, inputs, index):
        aggr_msg = scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

        return aggr_msg

    def update(self, aggr_out, x):
        upd_out = self.upd_h(x) + self.upd_agg(aggr_out)
        
        for layer in self.mlp_upd: 
            upd_out = layer(upd_out)

        return upd_out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')