import torch
from torch import nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from layers.sparse_encoder import SparseFeatureEncoder

class GraphsageSimCor(nn.Module):
    def __init__(self, node_nums, nodeid_emb_dim, sparse_maxs, sparse_emb_dim, hidden_dim, output_dim, n_layers, negs_nums, dropout, etypes, use_nodeid_emb):
        super(GraphsageSimCor, self).__init__()
        self.negs_nums = negs_nums
        self.etypes = etypes
        if use_nodeid_emb:
            sparse_maxs.append(node_nums)
            sparse_emb_dim.append(nodeid_emb_dim)
        self.sparse_feats_embedding = nn.ModuleDict({mode: SparseFeatureEncoder(sparse_maxs, sparse_emb_dim) for mode in etypes})
        self.layers = nn.ModuleDict({mode: nn.ModuleList() for mode in etypes})
        self.fc_input = nn.ModuleDict({mode: nn.Linear(sum(sparse_emb_dim), hidden_dim) for mode in etypes})
        for mode in etypes:
            self.layers[mode].append(SAGESimCorConv(dropout=dropout, mode=mode))
            for i in range(1, n_layers):
                self.layers[mode].append(SAGESimCorConv(dropout=dropout, mode=mode))
        self.fc_output = nn.ModuleDict({mode: nn.Linear(hidden_dim, output_dim) for mode in etypes})
        self.semantic_integration = SemanticIntegration(output_dim, output_dim)
        self.reset_parameters()


    def reset_parameters(self):
        pass

    def forward(self, blocks, x):
        emb = {mode: x.clone().detach() for mode in self.etypes}
        for mode in self.etypes:
            emb[mode] = self.sparse_feats_embedding[mode](emb[mode])
            emb[mode] = torch.cat(emb[mode], dim=1)
            emb[mode] = self.fc_input[mode](emb[mode])
            for l, (layer, block) in enumerate(zip(self.layers[mode], blocks)):
                emb[mode] = layer(block, emb[mode])
                if l != len(self.layers[mode]) - 1:
                    emb[mode] = F.relu(emb[mode])
                else:
                    emb[mode] = self.fc_output[mode](emb[mode])
        emb = self.semantic_integration(emb)
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



class SemanticIntegration(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(SemanticIntegration, self).__init__()
        self.sim2cor = nn.Linear(in_feats, out_feats, bias=False)
        self.cor2sim = nn.Linear(in_feats, out_feats, bias=False)
        self.a1 = nn.Parameter(torch.zeros(size=(1,)))
        self.a2 = nn.Parameter(torch.zeros(size=(1,)))
        self.b2 = nn.Parameter(torch.zeros(size=(1,)))

    def forward(self, emb):
        z1sim = (1 - self.a1) * emb['sim'] + self.a1 * self.cor2sim(emb['cor'])
        z1cor = (1 - self.a1) * emb['cor'] + self.a1 * self.sim2cor(emb['sim'])
        z2sim = (1 - self.a2 - self.b2) * emb['sim'] + self.a2 * self.cor2sim(emb['cor']) + self.b2 * self.cor2sim(z1cor)
        z2cor = (1 - self.a2 - self.b2) * emb['cor'] + self.a2 * self.sim2cor(emb['sim']) + self.b2 * self.sim2cor(z1sim)
        return {'sim': z2sim, 'cor': z2cor}



