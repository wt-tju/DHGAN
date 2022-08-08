import torch
from torch import nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from layers.sparse_encoder import SparseFeatureEncoder

class DecGCN_fast(nn.Module):
    def __init__(self, node_nums, nodeid_emb_dim, sparse_maxs, sparse_emb_dim, hidden_dim, output_dim, n_layers, negs_nums, dropout, etypes, use_nodeid_emb):
        super(DecGCN_fast, self).__init__()
        self.negs_nums = negs_nums
        self.etypes = etypes
        if use_nodeid_emb:
            sparse_maxs.append(node_nums)
            sparse_emb_dim.append(nodeid_emb_dim)
        self.sparse_feats_embedding = {mode: SparseFeatureEncoder(sparse_maxs, sparse_emb_dim) for mode in etypes}
        self.layers = {mode: nn.ModuleList() for mode in etypes}
        self.fc_input = {mode: nn.Linear(sum(sparse_emb_dim), hidden_dim) for mode in etypes}
        for mode in etypes:
            self.layers[mode].append(CoattentiveAggregation(dropout=dropout, mode=mode))
            for i in range(1, n_layers):
                self.layers[mode].append(CoattentiveAggregation(dropout=dropout, mode=mode))
        self.fc_output = {mode: nn.Linear(hidden_dim, output_dim) for mode in etypes}
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



class CoattentiveAggregation(nn.Module):
    def __init__(self, dropout, mode):
        super(CoattentiveAggregation, self).__init__()
        self.mode = mode
        self.reset_parameters()
        self.feat_drop = nn.Dropout(dropout)
        self.mean_pooling = nn.AvgPool1d(kernel_size=3, stride=None, padding=0)

    def reset_parameters(self):
        pass

    def forward(self, graph, feat):
        feat_src = feat_dst = self.feat_drop(feat)
        if graph.is_block:
            feat_dst = feat_dst[:graph.number_of_dst_nodes()]
        h_self = feat_dst
        max_neighs = torch.max(graph.in_degrees(torch.arange(graph.number_of_dst_nodes(), dtype=torch.int32), self.mode))
        h_neigh = self._pre_aggregation(graph, feat_src, max_neighs=max_neighs)

        rst = h_self + h_neigh
        return rst


    def _pre_aggregation(self, g, feat_src, max_neighs):
        neigh_feat = {k: torch.zeros((len(g.dstdata['_ID']), max_neighs, feat_src.shape[1]),
                                     device=feat_src.device) for k in g.etypes}
        for node in range(len(g.dstdata['_ID'])):
            for etype in g.etypes:
                src, dst = g.in_edges(int(node), etype=etype)
                nidx = torch.tensor(src, dtype=torch.long)
                repeat_num = max_neighs // nidx.shape[0] + (1 if max_neighs % nidx.shape[0] else 0)
                neigh_feat[etype][node] = feat_src[nidx].repeat(repeat_num, 1)[:max_neighs]
        if self.mode == 'cor':
            D = neigh_feat['sim']
            Q = neigh_feat['cor']
        elif self.mode == 'sim':
            D = neigh_feat['cor']
            Q = neigh_feat['sim']
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
        # self.a1 = nn.Parameter(torch.zeros(size=(1,)))
        # self.a2 = nn.Parameter(torch.zeros(size=(1,)))
        # self.b2 = nn.Parameter(torch.zeros(size=(1,)))
        self.a1 = 0.5
        self.a2 = 0.33
        self.b2 = 0.33

    def forward(self, emb):
        z1sim = (1 - self.a1) * emb['sim'] + self.a1 * self.cor2sim(emb['cor'])
        z1cor = (1 - self.a1) * emb['cor'] + self.a1 * self.sim2cor(emb['sim'])
        z2sim = (1 - self.a2 - self.b2) * emb['sim'] + self.a2 * self.cor2sim(emb['cor']) + self.b2 * self.cor2sim(z1cor)
        z2cor = (1 - self.a2 - self.b2) * emb['cor'] + self.a2 * self.sim2cor(emb['sim']) + self.b2 * self.sim2cor(z1sim)
        return {'sim': z2sim, 'cor': z2cor}


