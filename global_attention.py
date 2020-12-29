import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, GlobalAttention

from drelu import DyReLUA, DyReLUB, DyReLUC
from edgerelu import EdgeReluV2
def relu_None(x, edge_index):
    return F.relu(x)

class GlobalAttentionNet(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, kind="None"):
        super(GlobalAttentionNet, self).__init__()
        self.conv1 = SAGEConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()

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
