from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch
import torch.nn.functional as F


class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.fc1 = GCNConv(channels, channels // reduction)
#         self.fc1 = GraphConv(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2 * k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor(
            [1.] * k + [0.5] * k).float())
        self.register_buffer('init_v', torch.Tensor(
            [1.] + [0.] * (2 * k - 1)).float())

    def get_relu_coefs(self, x, edge_index):
        theta = x
        theta = self.fc1(theta, edge_index)
        theta = theta.mean(dim=0)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x, edge_index):
        raise NotImplementedError


class DyReLUA(DyReLU):
    def __init__(self, channels, reduction=4, k=2):
        super(DyReLUA, self).__init__(channels, reduction, k)
        self.fc2 = nn.Linear(channels // reduction, 2 * k)

    def forward(self, x, edge_index):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x, edge_index)
        relu_coefs = theta.view(-1, 2 * self.k) * self.lambdas + self.init_v
        x_perm = x.t().unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        result = torch.max(output, dim=-1)[0]
        return result.t()


class DyReLUB(DyReLU):
    def __init__(self, channels, reduction=4, k=2):
        super(DyReLUB, self).__init__(channels, reduction, k)
        self.fc2 = nn.Linear(channels // reduction, 2 * k * channels)

    def forward(self, x, edge_index):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x, edge_index)
        relu_coefs = theta.view(-1, self.channels, 2 *
                                self.k) * self.lambdas + self.init_v
        x = x.unsqueeze(-1)
        x_perm = x.permute(2, 0, 1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
        result = torch.max(output, dim=-1)[0].permute(1, 2, 0).squeeze()
        return result


class DyReLUC(DyReLU):
    def __init__(self, channels, reduction=4, k=2):
        super(DyReLUC, self).__init__(channels, reduction, k)
        self.fc2 = nn.Linear(channels // reduction, 2 * k * channels)
#         self.pos = GraphConv(channels, 1)
        self.pos = GCNConv(channels, 1)

    def pos_coefs(self, x, edge_index):
        x = self.pos(x, edge_index)
        x = x.squeeze()
        x = x / 10
        x_norm = F.softmax(x).view(-1, 1)
        x_norm = x_norm * (x.shape[0] / 3)
        return torch.min(x_norm, torch.ones_like(x_norm))

    def forward(self, x, edge_index):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x, edge_index)
        relu_coefs = theta.view(-1, self.channels, 2 *
                                self.k)
        pos_norm_coefs = self.pos_coefs(x, edge_index).view(-1, 1, 1)
        relu_coefs = relu_coefs * pos_norm_coefs * self.lambdas + self.init_v

        x = x.unsqueeze(-1)
        x_perm = x.permute(2, 0, 1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
        result = torch.max(output, dim=-1)[0].permute(1, 2, 0).squeeze()
        return result


class DyReLUE(DyReLU):
    def __init__(self, channels, reduction=4, k=2):
        super(DyReLUE, self).__init__(channels, reduction, k)
        self.fc2 = nn.Linear(channels // reduction, 2 * k * channels)
#         self.pos = GraphConv(channels, 1)
        self.pos = GCNConv(channels, 1)

    def pos_coefs(self, x, edge_index):
        x = self.pos(x, edge_index)
        x = x.squeeze()
        x = x / 10
        x_norm = F.softmax(x).view(-1, 1)
        x_norm = x_norm * (x.shape[0] / 3)
        return torch.min(x_norm, torch.ones_like(x_norm))

    def forward(self, x, edge_index):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x, edge_index)
        relu_coefs = theta.view(-1, self.channels, 2 *
                                self.k)
        pos_norm_coefs = self.pos_coefs(x, edge_index).view(-1, 1, 1)
        relu_coefs = relu_coefs * pos_norm_coefs * self.lambdas + self.init_v

        x = x.unsqueeze(-1)
        x_perm = x.permute(2, 0, 1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
        result = torch.max(output, dim=-1)[0].permute(1, 2, 0).squeeze()
        return result
