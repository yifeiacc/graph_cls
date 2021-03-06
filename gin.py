import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool, JumpingKnowledge
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


class GIN0(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, kind="None"):
        super(GIN0, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(dataset.num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=False)
        self.convs = torch.nn.ModuleList()
        self.relus = torch.nn.ModuleList()
        self.kind = kind
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
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=False))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relus[0](self.conv1(x, edge_index), edge_index)
        for conv, relu in zip(self.convs, self.relus[1:]):
            x = relu(conv(x, edge_index), edge_index)

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GIN0WithJK(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, mode='cat'):
        super(GIN0WithJK, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(dataset.num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=False)
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=False))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        x = self.jump(xs)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GIN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, kind="None"):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(dataset.num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=True)
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
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=True))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relus[0](self.conv1(x, edge_index), edge_index)
        for conv, relu in zip(self.convs, self.relus[1:]):
            x = relu(conv(x, edge_index), edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GINWithJK(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, mode='cat'):
        super(GINWithJK, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(dataset.num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=True))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        x = self.jump(xs)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
