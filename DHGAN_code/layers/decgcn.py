import torch
from torch import nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from layers.sparse_encoder import SparseFeatureEncoder


class DecGCN(nn.Module):
    def __init__(self, node_nums, nodeid_emb_dim, sparse_maxs, sparse_emb_dim, hidden_dim, output_dim, n_layers, negs_nums, dropout, etypes, fanout, use_nodeid_emb):
        super(DecGCN, self).__init__()
        self.negs_nums = negs_nums
        self.etypes = etypes
        self.fanout = fanout
        if use_nodeid_emb:
            sparse_maxs.append(node_nums)
            sparse_emb_dim.append(nodeid_emb_dim)
        self.sparse_feats_embedding = nn.ModuleDict({mode: SparseFeatureEncoder(sparse_maxs, sparse_emb_dim) for mode in etypes})
        self.layers = nn.ModuleDict({mode: nn.ModuleList() for mode in etypes})
        self.fc_input = nn.ModuleDict({mode: nn.Linear(sum(sparse_emb_dim), hidden_dim) for mode in etypes})
        for mode in etypes:
            self.layers[mode].append(CoattentiveAggregation(dropout=dropout, mode=mode))
            for i in range(1, n_layers):
                self.layers[mode].append(CoattentiveAggregation(dropout=dropout, mode=mode))
        self.fc_output = nn.ModuleDict({mode: nn.Linear(hidden_dim, output_dim) for mode in etypes})
        self.semantic_integration = SemanticIntegration(output_dim, output_dim)
        self.reset_parameters()


    def reset_parameters(self):
        pass

    def forward(self, blocks, x):
        emb = {mode: x.clone().detach() for mode in self.etypes}
        in_edges_id = self._get_in_edges(blocks)
        for mode in self.etypes:
            emb[mode] = self.sparse_feats_embedding[mode](emb[mode])
            emb[mode] = torch.cat(emb[mode], dim=1)
            emb[mode] = self.fc_input[mode](emb[mode])
            for l, (layer, block) in enumerate(zip(self.layers[mode], blocks)):
                emb[mode] = layer(block, emb[mode], in_edges_id[l])
                if l != len(self.layers[mode]) - 1:
                    emb[mode] = F.relu(emb[mode])
                else:
                    emb[mode] = self.fc_output[mode](emb[mode])
        emb = self.semantic_integration(emb)
        return emb

    def _get_in_edges(self, blocks):
        in_edge = [None for l in range(len(blocks))]
        for l, block in enumerate(blocks):
            g = block
            max_neighs = self.fanout[l]
            in_edges_layer = {k: torch.zeros((len(g.dstdata['_ID']), max_neighs), dtype=torch.int64, device=block.device) for k in g.etypes}
            for node in range(len(g.dstdata['_ID'])):
                for etype in g.etypes:
                    src, dst = g.in_edges(int(node), etype=etype)
                    if src.shape[0] == 0:
                        src = torch.tensor([int(node)], dtype=torch.int64)
                    src = src.repeat(max_neighs)[:max_neighs]
                    in_edges_layer[etype][node] = src
            in_edge[l] = in_edges_layer.copy()
        return in_edge


class CoattentiveAggregation(nn.Module):
    def __init__(self, dropout, mode):
        super(CoattentiveAggregation, self).__init__()
        self.mode = mode
        self.reset_parameters()
        self.feat_drop = nn.Dropout(dropout)
        self.mean_pooling = nn.AvgPool1d(kernel_size=3, stride=None, padding=0)

    def reset_parameters(self):
        pass

    def forward(self, graph, feat, in_edges_id):
        feat_src = feat_dst = self.feat_drop(feat)
        if graph.is_block:
            feat_dst = feat_dst[:graph.number_of_dst_nodes()]
        h_self = feat_dst
        h_neigh = self._pre_aggregation(graph, feat_src, in_edges_id=in_edges_id)

        rst = h_self + h_neigh
        return rst

    def _pre_aggregation(self, g, feat_src, in_edges_id):
        if self.mode == 'cor':
            D = feat_src[in_edges_id['sim']]
            Q = feat_src[in_edges_id['cor']]
        elif self.mode == 'sim':
            D = feat_src[in_edges_id['cor']]
            Q = feat_src[in_edges_id['sim']]
        neigh_hidden = self._coattention(D, Q)
        neigh_hidden = self.mean_pooling(neigh_hidden)
        neigh_hidden = torch.mean(neigh_hidden, dim=1)
        return neigh_hidden

    def _coattention(self, D, Q):
        L = torch.matmul(D, torch.transpose(Q, dim0=1, dim1=2))
        AC = F.softmax(L, dim=-1)
        AS = F.softmax(torch.transpose(L, dim0=1, dim1=2), dim=-1)
        CQ = torch.matmul(AC, Q)
        CD = torch.matmul(AS, torch.cat([D, CQ], dim=-1))
        return torch.cat([Q, CD], dim=-1)


class SemanticIntegration(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(SemanticIntegration, self).__init__()
        self.sim2cor = nn.Linear(in_feats, out_feats, bias=False)
        self.cor2sim = nn.Linear(in_feats, out_feats, bias=False)
        self.a1 = nn.Parameter(torch.zeros(size=(1,)))
        self.a2 = nn.Parameter(torch.zeros(size=(1,)))
        self.b2 = nn.Parameter(torch.zeros(size=(1,)))
        # self.a1 = 0.5
        # self.a2 = 0.33
        # self.b2 = 0.33

    def forward(self, emb):
        z1sim = (1 - self.a1) * emb['sim'] + self.a1 * self.cor2sim(emb['cor'])
        z1cor = (1 - self.a1) * emb['cor'] + self.a1 * self.sim2cor(emb['sim'])
        z2sim = (1 - self.a2 - self.b2) * emb['sim'] + self.a2 * self.cor2sim(emb['cor']) + self.b2 * self.cor2sim(z1cor)
        z2cor = (1 - self.a2 - self.b2) * emb['cor'] + self.a2 * self.sim2cor(emb['sim']) + self.b2 * self.sim2cor(z1sim)
        return {'sim': z2sim, 'cor': z2cor}


