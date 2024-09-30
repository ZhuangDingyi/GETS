import dgl.nn as dglnn

import torch.nn as nn
import torch.nn.functional as F


def load_gnn(conf):
    if conf.gnn["type"] == "gcn":
        return GCN(
            conf.gnn["in_dim"],
            conf.gnn["hid_dim"],
            conf.gnn["out_dim"],
            conf.gnn["num_layer"],
            conf.gnn["dropout"],
            conf.gnn["norm"]
        )
        
    elif conf.gnn["type"] == "gat":
        return GAT(
            conf.gnn["in_dim"],
            conf.gnn["hid_dim"],
            conf.gnn["out_dim"],
            conf.gnn["num_layer"],
            conf.gnn["dropout"],
            conf.gnn["norm"]
        )
    elif conf.gnn["type"] == "gin":
        return GIN(
            conf.gnn["in_dim"],
            conf.gnn["hid_dim"],
            conf.gnn["out_dim"],
            conf.gnn["num_layer"],
            conf.gnn["dropout"],
            conf.gnn["norm"]
        )
    else:
        raise NotImplementedError

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layer, dropout, norm):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = norm
        if norm:
            self.norms = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size)
        )
        if norm:
            self.norms.append(nn.BatchNorm1d(hid_size))
        for _ in range(num_layer - 2):
            self.layers.append(
                dglnn.GraphConv(hid_size, hid_size)
            )
            if norm:
                self.norms.append(nn.BatchNorm1d(hid_size))
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < len(self.layers) - 1:
                if self.norm:
                    h = self.norms[i](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h
    
class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layer, dropout, norm):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = norm
        if norm:
            self.norms = nn.ModuleList()
        num_heads = 2 
        
        # First GAT layer
        self.layers.append(
            dglnn.GATConv(in_size, hid_size, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout)
        )
        if norm:
            self.norms.append(nn.BatchNorm1d(hid_size * num_heads))
        
        # # Middle GAT layers
        # for _ in range(num_layer - 2):
        #     self.layers.append(
        #         dglnn.GATConv(hid_size * num_heads, hid_size, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout)
        #     )
        #     if norm:
        #         self.norms.append(nn.BatchNorm1d(hid_size * num_heads))
        
        # Output GAT layer
        self.final_project = nn.Linear(hid_size* num_heads , out_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < len(self.layers) - 1:
                if self.norm:
                    h = self.norms[i](h)
                h = F.relu(h)
                h = self.dropout(h)
        h = self.final_project(h.view(h.size(0), -1))
        return h
    
class GIN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layer, dropout, norm):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = norm
        if norm:
            self.norms = nn.ModuleList()
        # two-layer GIN
        self.layers.append(
            dglnn.GINConv(
                nn.Sequential(
                    nn.Linear(in_size, hid_size)
                ),
                'mean'
            )
        )
        if norm:
            self.norms.append(nn.BatchNorm1d(hid_size))
        for _ in range(num_layer - 2):
            self.layers.append(
                dglnn.GINConv(
                    nn.Sequential(
                        nn.Linear(hid_size, hid_size)
                    ),
                    'mean'
                )
            )
            if norm:
                self.norms.append(nn.BatchNorm1d(hid_size))
        self.layers.append(
            dglnn.GINConv(
                nn.Sequential(
                    nn.Linear(hid_size, out_size)
                ),
                'mean'
            )
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < len(self.layers) - 1:
                if self.norm:
                    h = self.norms[i](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h