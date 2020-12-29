from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool, JumpingKnowledge
from drelu import DyReLUA, DyReLUB, DyReLUC
from edgerelu import EdgeReluV2
from torch_geometric.utils import dense_to_sparse

def relu_None(x, edge_index):
    return F.relu(x)

class Block(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, mode='cat', kind="None"):
        super(Block, self).__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, out_channels)
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin = Linear(hidden_channels + out_channels, out_channels)
        else:
            self.lin = Linear(out_channels, out_channels)
        
        if kind == "None":
           self.relus = []
        else:
            self.relus = torch.nn.ModuleList()
       
        if kind == "A":
            self.relus.append(DyReLUA(hidden_channels))
            self.relus.append(DyReLUA(out_channels))
        elif kind == "B":
            self.relus.append(DyReLUB(hidden_channels))
            self.relus.append(DyReLUB(out_channels))
        elif kind == "C":
            self.relus.append(DyReLUC(hidden_channels))
            self.relus.append(DyReLUC(out_channels))
        elif kind == "D":
            self.relus.append(EdgeReluV2(hidden_channels))
            self.relus.append(EdgeReluV2(out_channels))
        elif kind == "None":
            self.relus.append(relu_None)
            self.relus.append(relu_None)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        edge_index, _ = dense_to_sparse(adj)
        x1 = self.relus[0](self.conv1(x, adj, mask, add_loop), edge_index)
        x2 = self.relus[1](self.conv2(x1, adj, mask, add_loop), edge_index)
        return self.lin(self.jump([x1, x2]))


class DiffPool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.25, kind="None"):
        super(DiffPool, self).__init__()

        num_nodes = ceil(ratio * dataset[0].num_nodes)
        self.embed_block1 = Block(dataset.num_features, hidden, hidden, kind=kind)
        self.pool_block1 = Block(dataset.num_features, hidden, num_nodes, kind=kind)

        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()
        for i in range((num_layers // 2) - 1):
            num_nodes = ceil(ratio * num_nodes)
            self.embed_blocks.append(Block(hidden, hidden, hidden, kind=kind))
            self.pool_blocks.append(Block(hidden, hidden, num_nodes, kind=kind))

        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear((len(self.embed_blocks) + 1) * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.embed_block1.reset_parameters()
        self.pool_block1.reset_parameters()
        for embed_block, pool_block in zip(self.embed_blocks,
                                           self.pool_blocks):
            embed_block.reset_parameters()
            pool_block.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, adj, mask = data.x, data.adj, data.mask

        s = self.pool_block1(x, adj, mask, add_loop=True)
        x = F.relu(self.embed_block1(x, adj, mask, add_loop=True))
        xs = [x.mean(dim=1)]
        x, adj, _, _ = dense_diff_pool(x, adj, s, mask)

        for i, (embed_block, pool_block) in enumerate(
                zip(self.embed_blocks, self.pool_blocks)):
            s = pool_block(x, adj)
            x = F.relu(embed_block(x, adj))
            xs.append(x.mean(dim=1))
            if i < len(self.embed_blocks) - 1:
                x, adj, _, _ = dense_diff_pool(x, adj, s)

        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__