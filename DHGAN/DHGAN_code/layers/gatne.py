import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import dgl
import dgl.function as fn
from layers.sparse_encoder import SparseFeatureEncoder


class GATNE(nn.Module):
    def __init__(self, node_nums, nodeid_emb_dim, sparse_maxs, sparse_emb_dim, hidden_dim, output_dim, n_layers, negs_nums, dropout, etypes, use_nodeid_emb):
        super(GATNE, self).__init__()
        self.negs_nums = negs_nums
        self.etypes = etypes
        if use_nodeid_emb:
            sparse_maxs.append(node_nums)
            sparse_emb_dim.append(nodeid_emb_dim)
        self.sparse_feats_embedding = SparseFeatureEncoder(sparse_maxs, sparse_emb_dim)
        self.fc_input = nn.Linear(sum(sparse_emb_dim), hidden_dim)
        self.layer = GATNELayer(etypes, hidden_dim)
        self.fc_output = nn.ModuleDict({mode: nn.Linear(hidden_dim, output_dim) for mode in etypes})
        self.reset_parameters()


    def reset_parameters(self):
        pass

    def forward(self, blocks, x):
        x = self.sparse_feats_embedding(x)
        x = torch.cat(x, dim=1)
        x = self.fc_input(x)
        emb = self.layer(blocks[0], x)
        for mode in self.etypes:
            emb[mode] = self.fc_output[mode](emb[mode])
        return emb


class GATNELayer(nn.Module):
    def __init__(
        self,
        edge_types,
        embedding_size,
        dim_a=20,
    ):
        super(GATNELayer, self).__init__()
        self.embedding_size = embedding_size
        self.edge_types = edge_types
        self.edge_type_count = len(edge_types)
        self.dim_a = dim_a

        self.trans_weights = Parameter(
            torch.FloatTensor(self.edge_type_count, embedding_size, embedding_size)
        )
        self.trans_weights_s1 = Parameter(
            torch.FloatTensor(self.edge_type_count, embedding_size, dim_a)
        )
        self.trans_weights_s2 = Parameter(torch.FloatTensor(self.edge_type_count, dim_a, 1))

        self.reset_parameters()

    def reset_parameters(self):
        # self.node_embeddings.data.uniform_(-1.0, 1.0)
        # self.node_type_embeddings.data.uniform_(-1.0, 1.0)
        self.trans_weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s1.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s2.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    # embs: [batch_size, embedding_size]
    def forward(self, block, feat):
        batch_size = block.number_of_dst_nodes()
        node_type_embed = []
        with block.local_scope():
            feat_src = feat
            feat_dst = feat_src[:block.number_of_dst_nodes()]
            for etype in self.edge_types:
                block.srcdata[etype] = feat_src
                block.dstdata[etype] = feat_dst
                block.update_all(
                    fn.copy_u(etype, "m"), fn.sum("m", etype), etype=etype
                )
                node_type_embed.append(block.dstdata[etype])
            #print(node_type_embed[0].shape)
            node_type_embed = torch.stack(node_type_embed, 1)
            #print(node_type_embed.shape)
            tmp_node_type_embed = node_type_embed.unsqueeze(2).view(
                -1, 1, self.embedding_size
            )
            #print(tmp_node_type_embed.shape)
            trans_w = (
                self.trans_weights.unsqueeze(0)
                .repeat(batch_size, 1, 1, 1)
                .view(-1, self.embedding_size, self.embedding_size)
            )
            #print(trans_w.shape)
            trans_w_s1 = (
                self.trans_weights_s1.unsqueeze(0)
                .repeat(batch_size, 1, 1, 1)
                .view(-1, self.embedding_size, self.dim_a)
            )
            #print(trans_w_s1.shape)
            trans_w_s2 = (
                self.trans_weights_s2.unsqueeze(0)
                .repeat(batch_size, 1, 1, 1)
                .view(-1, self.dim_a, 1)
            )
            #print(trans_w_s2.shape)

            attention = (
                F.softmax(
                    torch.matmul(
                        torch.tanh(torch.matmul(tmp_node_type_embed, trans_w_s1)),
                        trans_w_s2,
                    )
                    .squeeze(2)
                    .view(-1, self.edge_type_count),
                    dim=1,
                )
                .unsqueeze(1)
                .repeat(1, self.edge_type_count, 1)
            )
            #print(attention.shape)
            node_type_embed = torch.matmul(attention, node_type_embed).view(
                -1, 1, self.embedding_size
            )
            node_embed = feat_dst.unsqueeze(1).repeat(
                1, self.edge_type_count, 1
            ) + torch.matmul(node_type_embed, trans_w).view(
                -1, self.edge_type_count, self.embedding_size
            )
            #print(node_embed.shape)
            last_node_embed = F.normalize(node_embed, dim=2)

            res_emb = {}
            for i, etype in enumerate(self.edge_types):
                res_emb[etype] = last_node_embed[:, i]
            # return last_node_embed  # [batch_size, edge_type_count, embedding_size]
            return res_emb
