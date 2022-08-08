import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv
from layers.sparse_encoder import SparseFeatureEncoder


class HAN(nn.Module):
    def __init__(self, node_nums, nodeid_emb_dim, sparse_maxs, sparse_emb_dim, hidden_dim, output_dim, n_layers, negs_nums, dropout, etypes, use_nodeid_emb):
        super(HAN, self).__init__()
        self.negs_nums = negs_nums
        self.etypes = etypes
        if use_nodeid_emb:
            sparse_maxs.append(node_nums)
            sparse_emb_dim.append(nodeid_emb_dim)
        self.sparse_feats_embedding = SparseFeatureEncoder(sparse_maxs, sparse_emb_dim)
        self.fc_input = nn.ModuleDict({mode: nn.Linear(sum(sparse_emb_dim), hidden_dim) for mode in etypes})
        self.layers = nn.ModuleDict({mode: nn.ModuleList() for mode in etypes})
        for mode in etypes:
            self.layers[mode].append(HANLayer(len(etypes), hidden_dim, hidden_dim, dropout=dropout))
            for i in range(1, n_layers):
                self.layers[mode].append(HANLayer(len(etypes), hidden_dim, hidden_dim, dropout=dropout))
        self.fc_output = nn.ModuleDict({mode: nn.Linear(hidden_dim, output_dim) for mode in etypes})
        self.reset_parameters()


    def reset_parameters(self):
        pass

    def forward(self, blocks, x):
        num_src = blocks[0].number_of_src_nodes()
        num_dst = blocks[1].number_of_dst_nodes()

        x = self.sparse_feats_embedding(x)
        x = torch.cat(x, dim=1)
        emb = {mode: x.clone().detach() for mode in self.etypes}
        gs_homo = []
        for block in blocks:
            g_homo = []
            for mode in self.etypes:
                edges_num = block.num_edges(mode)
                src, dst = block.find_edges(np.arange(edges_num), etype=mode)
                g = dgl.graph((src, dst), num_nodes=num_src, device=block.device)
                g = dgl.add_self_loop(g)
                g_homo.append(g)
            gs_homo.append(g_homo)
        for mode in self.etypes:
            emb[mode] = self.fc_input[mode](emb[mode])
            for l, (layer, block) in enumerate(zip(self.layers[mode], gs_homo)):
                emb[mode] = layer(block, emb[mode])
                if l != len(self.layers[mode]) - 1:
                    emb[mode] = F.relu(emb[mode])
                else:
                    emb[mode] = emb[mode][:num_dst]
                    emb[mode] = self.fc_output[mode](emb[mode])
        return emb

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)


class HANLayer(nn.Module):
    def __init__(self, num_meta_paths, in_size, out_size, dropout, layer_num_heads=1):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)




