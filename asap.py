import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (ASAPooling, GraphConv, global_mean_pool,
                                JumpingKnowledge)

from drelu import DyReLUA, DyReLUB, DyReLUC
from edgerelu import EdgeReluV2

def relu_None(x, edge_index):
    return F.relu(x)

class ASAP(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.8, dropout=0, kind="None"):
        super(ASAP, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, hidden, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        if kind == "None":
           self.relus = []
        else:
            self.relus = torch.nn.ModuleList()
        for i in range(num_layers):
            if kind == "A":
                self.relus.append(DyReLUA(hidden))
            elif kind == "B":
                self.relus.append(DyReLUB(hidden))
            elif kind == "C":
                self.relus.append(DyReLUC(hidden))
            elif kind == "D":
                self.relus.append(EdgeReluV2(hidden))
            elif kind == "None":
                self.relus.append(relu_None)

        self.convs.extend([
            GraphConv(hidden, hidden, aggr='mean')
            for i in range(num_layers - 1)
        ])
        self.pools.extend([
            ASAPooling(hidden, ratio, dropout=dropout)
            for i in range((num_layers) // 2)
        ])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = None
        x = self.relus[0](self.conv1(x, edge_index), edge_index)
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_weight)
            x = self.relus[i+1](x, edge_index)
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, edge_weight, batch, _ = pool(
                    x=x, edge_index=edge_index, edge_weight=edge_weight,
                    batch=batch)
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
