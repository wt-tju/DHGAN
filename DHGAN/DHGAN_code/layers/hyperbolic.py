import torch
from torch import nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from layers.sparse_encoder import SparseFeatureEncoder
from hyperbolic.PoincareManifold import PoincareManifold


class HyperbolicGCN(nn.Module):
    def __init__(self, node_nums, nodeid_emb_dim, sparse_maxs, sparse_emb_dim, hidden_dim, output_dim, n_layers, negs_nums, dropout, etypes, fanout, use_nodeid_emb):
        super(HyperbolicGCN, self).__init__()
        self.negs_nums = negs_nums
        self.etypes = etypes
        self.fanout = fanout
        self.Ws_s = nn.Parameter(torch.zeros(size=(1,))) 
        self.Ws_c = nn.Parameter(torch.zeros(size=(1,)))
        self.Wc_c = nn.Parameter(torch.zeros(size=(1,)))
        self.Wc_s = nn.Parameter(torch.zeros(size=(1,)))
        if use_nodeid_emb:
            sparse_maxs.append(node_nums)  
            sparse_emb_dim.append(nodeid_emb_dim) 
        self.poincare = PoincareManifold(0, 0) 
        self.sparse_feats_embedding = nn.ModuleDict({mode: SparseFeatureEncoder(sparse_maxs, sparse_emb_dim) for mode in etypes})
        self.layers = nn.ModuleDict({mode: nn.ModuleList() for mode in etypes})
        for mode in etypes:
            self.layers[mode].append(H_CoattentiveAggregation(model_dims=sum(sparse_emb_dim), poincare=self.poincare, dropout=dropout, mode=mode))
            for i in range(1, n_layers):
                self.layers[mode].append(H_CoattentiveAggregation(model_dims=sum(sparse_emb_dim), poincare=self.poincare, dropout=dropout, mode=mode))
        self.fc_output = {mode: nn.Linear(hidden_dim, output_dim) for mode in etypes}  
        self.fc_input = nn.ModuleDict({mode: nn.Linear(sum(sparse_emb_dim), hidden_dim) for mode in etypes})
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

            emb[mode] = self.poincare.exp_map_zero(emb[mode])

            for l, (layer, block) in enumerate(zip(self.layers[mode], blocks)):
                emb[mode] = layer(block, emb[mode], in_edges_id[l])
                if l != len(self.layers[mode]) - 1:
                    emb[mode] = F.relu(emb[mode])
        emb[self.etypes[0]] = self.poincare.exp_map_zero(self.fc_output[self.etypes[0]](self.poincare.log_map_zero(emb[self.etypes[0]])))
        emb[self.etypes[1]] = self.poincare.exp_map_zero(self.fc_output[self.etypes[1]](self.poincare.log_map_zero(emb[self.etypes[1]])))
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



class H_CoattentiveAggregation(nn.Module):
    def __init__(self, model_dims, poincare, dropout, mode):
        super(H_CoattentiveAggregation, self).__init__()
        self.mode = mode
        self.poincare = poincare
        self.feat_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(model_dims, model_dims)
        self.attention = Attention(64)
        self.reset_parameters()
        self.q = nn.Parameter(torch.zeros(size=(1,)))
        self.qq = nn.Parameter(torch.zeros(size=(1,)))
        self.q1 = nn.Parameter(torch.zeros(size=(1,)))
        self.q2 = nn.Parameter(torch.zeros(size=(1,)))

    def reset_parameters(self):
        pass

    def forward(self, graph, feat, in_edges_id):
        feat_src = feat_dst = self.feat_drop(feat) 
        if graph.is_block:
            feat_dst = feat_dst[:graph.number_of_dst_nodes()]
        h_self = feat_dst  
        h = h_self.unsqueeze(1)  
        h_neigh = self._pre_aggregation(graph, feat_src, in_edges_id)  
        atten = -self.poincare.distance(h, h_neigh)  
        n_atten = F.softmax(atten, dim=1)

        neigh_T = self.poincare.mob_scalar_multi(h_neigh, n_atten)
        neigh_TE = self.poincare.log_map_zero(neigh_T).sum(dim=1, keepdim=True)
        neigh_TE = self.poincare.exp_map_zero(neigh_TE).squeeze(1)
        neigh_TE = self.poincare.mob_add(self.poincare.mob_scalar_multi(h_self, self.q), neigh_TE)
        
        if self.mode in ['cor','sim']:
            h_neigh_dir = self._aggre(graph, feat_src, in_edges_id)
            atten1 = -self.poincare.distance(h, h_neigh_dir)  
            n_atten1 = F.softmax(atten1, dim=1)

            neigh_T_dir = self.poincare.mob_scalar_multi(h_neigh_dir, n_atten1)
            neigh_TE_dir = self.poincare.log_map_zero(neigh_T_dir).sum(dim=1, keepdim=True)
            neigh_TE_dir = self.poincare.exp_map_zero(neigh_TE_dir).squeeze(1)
            neigh_TE_dir = self.poincare.mob_add(self.poincare.mob_scalar_multi(h_self, self.qq), neigh_TE_dir)

            rst = torch.stack([self.poincare.log_map_zero(neigh_TE), self.poincare.log_map_zero(neigh_TE_dir)], dim=1)
            rst, att = self.attention(rst)
            return self.poincare.exp_map_zero(rst)

    def _pre_aggregation(self, g, feat_src, in_edges_id):
        if self.mode == 'cor':
            D = feat_src[in_edges_id['sim']]  
            Q = feat_src[in_edges_id['cor']]  
            
        elif self.mode == 'sim':
            D = feat_src[in_edges_id['cor']]
            Q = feat_src[in_edges_id['sim']]
        neigh_hidden = self._coattention(D, Q) 
        return neigh_hidden

        
    def _aggre(self, g, feat_src, in_edges_id):
        if self.mode == 'cor':
            D = feat_src[in_edges_id['sim']]  
            Q = feat_src[in_edges_id['cor']] 
        elif self.mode == 'sim':
            D = feat_src[in_edges_id['cor']]
            Q = feat_src[in_edges_id['sim']]
        return Q


    def _coattention(self, D, Q):
        for i in range(len(D[1, :, 1])):  
           
            Zk_s = D[:, i:i+1, :].clone().detach()
            L = -self.poincare.distance(Zk_s, Q)
            G = F.softmax(L, dim=1)
            neigh_E = self.poincare.mob_scalar_multi(Q, G)
            neigh_h1 = self.poincare.log_map_zero(neigh_E).sum(dim=1, keepdim=True)
            neigh_h_1 = self.poincare.exp_map_zero(neigh_h1)
            D[:, i:i + 1, :] = self.poincare.mob_add(self.poincare.mob_scalar_multi(Zk_s, self.q1), neigh_h_1)
            
        return D
    
    def _coattention_sim(self, D, Q):
        for i in range(len(D[1, :, 1])):  
           
            Zk_s = D[:, i:i+1, :].clone().detach()
            L = -self.poincare.distance(Zk_s, Q)
            G = F.softmax(L, dim=1)
            neigh_E = self.poincare.mob_scalar_multi(Q, G)
            neigh_h1 = self.poincare.log_map_zero(neigh_E).sum(dim=1, keepdim=True)
            neigh_h_1 = self.poincare.exp_map_zero(neigh_h1)
            D[:, i:i + 1, :] = self.poincare.mob_add(self.poincare.mob_scalar_multi(Zk_s, self.q1), neigh_h_1)
            
            
        res = torch.zeros_like(Q)
        for i in range(len(Q[1, :, 1])):
            Zj_s = Q[:, i:i+1, :].clone().detach()
            L2 = -self.poincare.distance(Zj_s, D)
            G2 = F.softmax(L2, dim=1)
            neigh_E2 = self.poincare.mob_scalar_multi(D, G2)
            neigh_h2 = self.poincare.log_map_zero(neigh_E2).sum(dim=1, keepdim=True)
            neigh_h_2 = self.poincare.exp_map_zero(neigh_h2)
            res[:, i:i + 1, :] = self.poincare.mob_add(self.poincare.mob_scalar_multi(Zk_s, self.q2), neigh_h_2)
            
        return res


    
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta