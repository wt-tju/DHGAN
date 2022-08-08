import torch
from torch import nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from layers.sparse_encoder import SparseFeatureEncoder

class Graphsage(nn.Module):
    def __init__(self, node_nums, nodeid_emb_dim, sparse_maxs, sparse_emb_dim, hidden_dim, output_dim, n_layers, negs_nums, dropout, etypes, use_nodeid_emb):
        super(Graphsage, self).__init__()
        self.negs_nums = negs_nums
        self.etypes = etypes
        if use_nodeid_emb:
            sparse_maxs.append(node_nums)
            sparse_emb_dim.append(nodeid_emb_dim)
        self.sparse_feats_embedding = SparseFeatureEncoder(sparse_maxs, sparse_emb_dim)
        self.layers = nn.ModuleDict({mode: nn.ModuleList() for mode in etypes})
        self.fc_input = nn.ModuleDict({mode: nn.Linear(sum(sparse_emb_dim), hidden_dim) for mode in etypes})
        for mode in etypes:
            self.layers[mode].append(SAGESimCorConv(dropout=dropout, mode=mode))
            for i in range(1, n_layers):
                self.layers[mode].append(SAGESimCorConv(dropout=dropout, mode=mode))
        self.fc_output = nn.ModuleDict({mode: nn.Linear(hidden_dim, output_dim) for mode in etypes})
        self.reset_parameters()


    def reset_parameters(self):
        pass

    def forward(self, blocks, x):
        x = self.sparse_feats_embedding(x)
        x = torch.cat(x, dim=1)
        emb = {mode: x.clone().detach() for mode in self.etypes}
        for mode in self.etypes:
            emb[mode] = self.fc_input[mode](emb[mode])
            for l, (layer, block) in enumerate(zip(self.layers[mode], blocks)):
                emb[mode] = layer(block, emb[mode])
                if l != len(self.layers[mode]) - 1:
                    emb[mode] = F.relu(emb[mode])
                else:
                    emb[mode] = self.fc_output[mode](emb[mode])
        return emb



class SAGESimCorConv(nn.Module):
    def __init__(self, dropout, mode):
        super(SAGESimCorConv, self).__init__()
        self.mode = mode
        self.feat_drop = nn.Dropout(dropout)

    def forward(self, graph, feat):
        with graph.local_scope():
            feat_src = feat_dst = self.feat_drop(feat)
            if graph.is_block:
                feat_dst = feat_src[:graph.number_of_dst_nodes()]
            h_self = feat_dst

            # Message Passing
            graph.srcdata['h'] = feat_src
            graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'), etype=self.mode)
            h_neigh = graph.dstdata['neigh']

            rst = h_self + h_neigh
            return rst

