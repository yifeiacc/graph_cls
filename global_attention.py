import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, GlobalAttention

from drelu import DyReLUA, DyReLUB, DyReLUC
from edgerelu import EdgeReluV2
class xReLU(torch.nn.Module):
    def __init__(self, kind):
        super(xReLU, self).__init__()
        self.kind = kind
        self.PReLU = torch.nn.PReLU()

    def forward(self, x, edge_index):
        if self.kind == "ReLU":
            return F.relu(x)
        elif self.kind == "PReLU":
            return self.PReLU(x)
        elif self.kind == "ELU":
            return F.elu(x, alpha=1)
        elif self.kind == "LReLU":
            return F.leaky_relu(x, negative_slope=0.01)
        else:
            return x

class GlobalAttentionNet(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, kind="None"):
        super(GlobalAttentionNet, self).__init__()
        self.conv1 = SAGEConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()

        self.kind = kind
        self.relus = torch.nn.ModuleList()
        for i in range(num_layers):
            if self.kind == "ReLU":
                self.relus.append(xReLU("ReLU"))
            elif self.kind == "ELU":
                self.relus.append(xReLU("ELU"))
            elif self.kind == "PReLU":
                self.relus.append(xReLU("PReLU"))
            elif self.kind == "LReLU":
                self.relus.append(xReLU("LReLU"))
            elif self.kind == "GraphReLUNode":
                self.relus.append(DyReLUC(hidden))
            elif self.kind == "GraphReLUEdge":
                self.relus.append(EdgeReluV2(hidden))

        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.att = GlobalAttention(Linear(hidden, 1))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.att.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relus[0](self.conv1(x, edge_index), edge_index)
        for conv, relu in zip(self.convs, self.relus[1:]):
            x = relu(conv(x, edge_index), edge_index)
        x = self.att(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
