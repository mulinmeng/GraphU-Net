import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_geometric.nn import global_sort_pool
from torch_geometric.nn import TopKPooling, GCNConv, GINConv, GraphConv
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)

import utils


class Conv1dReadout(nn.Module):
    def __init__(
        self,
        input_dim,
        k,
        conv1d_channels=[16, 32],
        conv1d_kws=[0, 5]
    ):
        super(Conv1dReadout, self).__init__()
        self.k = k

        conv1d_kws[0] = input_dim
        self.input_dim = input_dim
        
        self.conv1d_p1 = nn.Conv1d(
            1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.pool = nn.MaxPool1d(2, 2)
        self.conv1d_p2 = nn.Conv1d(
            conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        
        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        
    def forward(self, x, batch):
        batch_size = batch.max() + 1
        re = global_sort_pool(x, batch, self.k)
        # re shape bs * (k * out_dim) 2910
        re = re.unsqueeze(2).transpose(2, 1)
        conv1 = F.relu(self.conv1d_p1(re))
        conv1 = self.pool(conv1)
        conv2 = F.relu(self.conv1d_p2(conv1))
        to_dense = conv2.view(batch_size, -1)
        out = F.relu(to_dense)
        return out


class DownNet(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, ks, dropout=0.5):
        super(DownNet, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.pool_ratios = ks
        self.depth = len(ks)
        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, self.hid_channels, improved=True))

        for i in range(len(ks)):
            self.pools.append(TopKPooling(self.hid_channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(self.hid_channels, self.hid_channels, improved=True))
        #in_channels = self.hid_channels
        
        self.convs = torch.nn.ModuleList()
        # for i in range(len(ks)):
        #     self.convs.append(GCNConv(in_channels, self.hid_channels, improved=True))
        self.convs.append(GCNConv(self.hid_channels, out_channels, improved=True))
        if dropout:
            self.drop = torch.nn.Dropout(p=0.5)
        else:
            self.drop = torch.nn.Dropout(p=0.)
        self.act = F.relu
        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))
        edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)
        x = self.drop(x)

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))

            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](x, edge_index, edge_weight, batch)
            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

        # for i in range(self.depth):
        #     x = self.convs[i](x, edge_index, edge_weight)
        #     x = self.act(x) if i < self.depth - 1 else x
        #     x = self.drop(x) if i < self.depth - 1 else x
        x = self.convs[-1](x, edge_index, edge_weight)
        return x, batch

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight


class GraphUNet(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, ks,
        sum_res=True, dropout=0.5
    ):
        super(GraphUNet, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.pool_ratios = ks
        self.depth = len(ks)
        self.act = F.relu
        self.sum_res=sum_res
        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, self.hid_channels))
        self.bn_layers.append(nn.BatchNorm1d(self.hid_channels))

        for i in range(len(ks)):
            self.pools.append(TopKPooling(self.hid_channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(self.hid_channels, self.hid_channels))
            self.bn_layers.append(nn.BatchNorm1d(self.hid_channels))
        
        in_channels = self.hid_channels if sum_res else 2 * self.hid_channels
        self.up_convs = torch.nn.ModuleList()
        for i in range(len(ks)):
            self.up_convs.append(GCNConv(in_channels, self.hid_channels))
            self.bn_layers.append(nn.BatchNorm1d(self.hid_channels))
        self.up_convs.append(GCNConv(2 * self.hid_channels, out_channels))
        if dropout:
            self.drop = torch.nn.Dropout(p=0.3)
        else:
            self.drop = torch.nn.Dropout(p=0.)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        #print("Step 1:", x)
        #print(x)
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))
        #edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
        #print("edge_weight:", edge_weight)
        #print("edge_index:", edge_index)
        x = self.down_convs[0](x, edge_index, edge_weight)
        #x = self.bn_layers[0](x)
        x = self.act(x)
        x = self.drop(x)
        #print("Step 2:", x)
        #print("init x shape", x.shape)
        org_X = x

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            #edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
            #print(i, x)
            #print("**************")

            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](x, edge_index, edge_weight, batch)
            #print("[ITEST] pool:", i, x)
            #print(self.down_convs[i])
            #print("edge_index:", edge_index)
            #print("edge_weight:", edge_weight)
            x = self.down_convs[i](x, edge_index, edge_weight)
            #x = self.bn_layers[i](x)
            x = self.act(x)
            #print("[INFO] i:", i, x)
            #print("x shape:", x.shape)
            # x = self.drop(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]
        #print("Step 3:", x)
        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            #x = self.bn_layers[i + self.depth + 1](x)
            x = self.act(x) if i < self.depth - 1 else x
            x = self.drop(x) if i < self.depth - 1 else x
        x = torch.cat([x, org_X], 1)
        #print("Step 4:", x)
        x = self.up_convs[-1](x, edge_index, edge_weight)
        #print("[INFO] Test")
        #print(x)
        #exit(0)
        return x

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, with_dropout=False):
        super(MLPClassifier, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.with_dropout = with_dropout
        utils.weights_init(self)

    def forward(self, x):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)

        logits = self.h2_weights(h1)
        logits = F.log_softmax(logits, dim=1)
        return logits


class DownClassifier(nn.Module):
    def __init__(self, node_feat, nn_hid, nn_out, k, hidden, num_class):
        super(DownClassifier, self).__init__()
        self.feature_extractor = DownNet(
            in_channels=node_feat,
            hid_channels=nn_hid,
            out_channels=nn_out,
            ks=[0.9, 0.7, 0.6, 0.5],
            dropout=0.3
        )
        self.readout = Conv1dReadout(input_dim=nn_out, k=k)
        self.mlp = MLPClassifier(
            input_size=self.readout.dense_dim, hidden_size=hidden,
            num_class=num_class, with_dropout=True
        )
    
    def forward(self, x, edge_index, batch):
        gfeat, batch = self.feature_extractor(x, edge_index, batch)
        #print(gfeat.shape)
        vfeat = self.readout(gfeat, batch)
        return self.mlp(vfeat)


class Classifier(nn.Module):
    def __init__(self, node_feat, nn_hid, nn_out, k, hidden, num_class):
        super(Classifier, self).__init__()
        self.feature_extractor = GraphUNet(
            in_channels=node_feat,
            hid_channels=nn_hid,
            out_channels=nn_out,
            ks=[0.9, 0.7, 0.6, 0.5],
            sum_res=True,
            dropout=0.3
        )
        self.readout = Conv1dReadout(input_dim=nn_out, k=k)
        self.mlp = MLPClassifier(
            input_size=self.readout.dense_dim, hidden_size=hidden,
            num_class=num_class, with_dropout=True
        )
    
    def forward(self, x, edge_index, batch):
        gfeat = self.feature_extractor(x, edge_index, batch)
        vfeat = self.readout(gfeat, batch)
        return self.mlp(vfeat)


class GraphUNets(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, ks,
        sum_res=True, dropout=0.5
    ):
        super(GraphUNets, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.pool_ratios = ks
        self.depth = len(ks)
        self.act = F.relu
        self.sum_res=sum_res
        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        self.down_convs.append(GraphConv(in_channels, self.hid_channels))
        self.bn_layers.append(nn.BatchNorm1d(self.hid_channels))

        for i in range(len(ks)):
            self.pools.append(TopKPooling(self.hid_channels, self.pool_ratios[i]))
            self.down_convs.append(GraphConv(self.hid_channels, self.hid_channels))
            self.bn_layers.append(nn.BatchNorm1d(self.hid_channels))
        
        in_channels = self.hid_channels if sum_res else 2 * self.hid_channels
        self.up_convs = torch.nn.ModuleList()
        for i in range(len(ks)):
            self.up_convs.append(GraphConv(in_channels, self.hid_channels))
            self.bn_layers.append(nn.BatchNorm1d(self.hid_channels))
        self.up_convs.append(GraphConv(2 * self.hid_channels, out_channels))
        if dropout:
            self.drop = torch.nn.Dropout(p=0.5)
        else:
            self.drop = torch.nn.Dropout(p=0.)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        #print("Step 1:", x)
        #print(x)
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))
        #edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
        #print("edge_weight:", edge_weight)
        #print("edge_index:", edge_index)
        x = self.down_convs[0](x, edge_index, edge_weight)
        #x = self.bn_layers[0](x)
        x = self.act(x)
        x = self.drop(x)
        #print("Step 2:", x)
        #print("init x shape", x.shape)
        org_X = x

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            #edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
            #print(i, x)
            #print("**************")

            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](x, edge_index, edge_weight, batch)
            #print("[ITEST] pool:", i, x)
            #print(self.down_convs[i])
            #print("edge_index:", edge_index)
            #print("edge_weight:", edge_weight)
            x = self.down_convs[i](x, edge_index, edge_weight)
            #x = self.bn_layers[i](x)
            x = self.act(x)
            #print("[INFO] i:", i, x)
            #print("x shape:", x.shape)
            #x = self.drop(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]
        #print("Step 3:", x)
        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            #x = self.bn_layers[i + self.depth + 1](x)
            x = self.act(x) if i < self.depth - 1 else x
            x = self.drop(x) if i < self.depth - 1 else x
        x = torch.cat([x, org_X], 1)
        #print("Step 4:", x)
        x = self.up_convs[-1](x, edge_index, edge_weight)
        #print("[INFO] Test")
        #print(x)
        #exit(0)
        return x

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight
