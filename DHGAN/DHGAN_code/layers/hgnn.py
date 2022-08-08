import torch
from torch import nn
import torch.nn.functional as F
import dgl
import numpy as np
from dgl.nn.functional import edge_softmax
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from layers.sparse_encoder import SparseFeatureEncoder
from hyperbolic.PoincareManifold import PoincareManifold

class HGNN(nn.Module):
    def __init__(self, node_nums, nodeid_emb_dim, sparse_maxs, sparse_emb_dim, hidden_dim, output_dim, n_layers, negs_nums, dropout, etypes, use_nodeid_emb):
        super(HGNN, self).__init__()
        self.negs_nums = negs_nums
        self.etypes = etypes
        if use_nodeid_emb:
            sparse_maxs.append(node_nums)
            sparse_emb_dim.append(nodeid_emb_dim)
        self.sparse_feats_embedding = SparseFeatureEncoder(sparse_maxs, sparse_emb_dim)
        self.fc_input = nn.ModuleDict({mode: nn.Linear(sum(sparse_emb_dim), hidden_dim) for mode in etypes})
        self.poincare = PoincareManifold(0, 0)
        self.layers = nn.ModuleDict({mode: nn.ModuleList() for mode in etypes})
        for mode in etypes:
            self.layers[mode].append(HATLayer(dropout=dropout, hidden_dim=hidden_dim, mode=mode, poincare=self.poincare))
            for i in range(1, n_layers):
                self.layers[mode].append(HATLayer(dropout=dropout, hidden_dim=hidden_dim, mode=mode, poincare=self.poincare))
        self.fc_output = nn.ModuleDict({mode: nn.Linear(hidden_dim, output_dim) for mode in etypes})
        self.reset_parameters()


    def reset_parameters(self):
        pass

    def forward(self, blocks, x):
        x = self.sparse_feats_embedding(x)
        x = torch.cat(x, dim=1)  ## [190335, 38]
        emb = {mode: x.clone().detach() for mode in self.etypes}
        blocks_homo = {mode: [] for mode in self.etypes}
        ## homo graph
        for block in blocks:
            for mode in self.etypes:
                edges_num = block.num_edges(mode)
                src, dst = block.find_edges(np.arange(edges_num), etype=mode)
                g_homo = dgl.graph((src, dst), num_nodes=block.num_nodes('item'), device=block.device)
                g_homo.add_edges(dst, dst)
                block_homo = dgl.to_block(g_homo, dst_nodes=np.arange(block.dstdata['_ID'].shape[0]))
                blocks_homo[mode].append(block_homo)
        for mode in self.etypes:
            emb[mode] = self.fc_input[mode](emb[mode])
            emb[mode] = self.poincare.exp_map_zero(emb[mode])
            for l, (layer, block) in enumerate(zip(self.layers[mode], blocks_homo[mode])):
                emb_index = torch.tensor(block.srcdata['_ID'], dtype=torch.long)
                emb_tmp = layer(block, emb[mode][emb_index])
                emb[mode] = emb_tmp
                if l != len(self.layers[mode]) - 1:
                    emb[mode] = F.relu(emb[mode])
                else:
                    emb[mode] = self.poincare.exp_map_zero(self.fc_output[mode](self.poincare.log_map_zero(emb[mode])))
#         print(emb['cor'])
#         print(emb['sim'])
        return emb



class HATLayer(nn.Module):
    def __init__(self, dropout, hidden_dim, mode, poincare):
        super(HATLayer, self).__init__()
        self.mode = mode
        self.feat_drop = nn.Dropout(dropout)
        self.poincare = poincare

    def forward(self, graph, feat):

        feat_src = feat_dst = self.feat_drop(feat)
        if graph.is_block:
            feat_dst = feat_src[:graph.number_of_dst_nodes()]
        graph.srcdata.update({'ft_src': feat_src})
        graph.dstdata.update({'ft_dst': feat_dst})
        # message passing
        graph.update_all(message_func, reduce_func)
        rst = graph.dstdata['h']
        return rst

def message_func(edges):
    # scores = torch.matmul(edges.src['h'].unsqueeze(dim=-2), edges.dst['h'].unsqueeze(dim=-1))
#     from hyperbolic.PoincareManifold import PoincareManifold
#     poincare = PoincareManifold(0, 0)
#     src_emb, dis_emb = edges.src.pop('ft_src'), edges.dst.pop('ft_dst')
#     e = -poincare.distance(src_emb, dis_emb)
#     return {'e': e}
    return {'z':edges.src['ft_src']}

def reduce_func(nodes):
    from hyperbolic.PoincareManifold import PoincareManifold
    poincare = PoincareManifold(0, 0)
    h = poincare.exp_map_zero(torch.mean(poincare.log_map_zero(nodes.mailbox['z']), dim = 1))
    #h = torch.sum(nodes.mailbox['e'] * nodes.mailbox['z'], dim = 1)
    return {'h':h}