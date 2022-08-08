import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


def mrr(logits, negs_logits):
    mrr_all = torch.cat((negs_logits, logits), dim=-1)
    mrr_size = mrr_all.shape[-1]
    _, indices_of_ranks = mrr_all.topk(mrr_size)
    _, ranks = (-indices_of_ranks).topk(mrr_size)
    _mrr = (ranks[:, -1] + 1).float().reciprocal().mean()
    return _mrr


class DECGCNLoss(nn.Module):
    def __init__(self, neg_nums, edge_types):
        super(DECGCNLoss, self).__init__()
        self.neg_nums = neg_nums
        self.edge_types = edge_types

    def forward(self, block_outputs, pos_graph, neg_graph):
        loss = []
        _mrr = []
        for etype in self.edge_types:
            with pos_graph.local_scope():
                pos_graph.ndata['h'] = block_outputs[etype]
                pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype) 
                pos_score = pos_graph.edata['score']
            with neg_graph.local_scope():
                neg_graph.ndata['h'] = block_outputs[etype]
                neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
                neg_score = neg_graph.edata['score']


            if pos_score[list(pos_score.keys())[0]].shape[0] == 0:
                continue
            pos_logits = pos_score[list(pos_score.keys())[0]]
            neg_logits = neg_score[list(neg_score.keys())[0]]
            print(pos_logits)
            print(neg_logits)
            _mrr.append(mrr(pos_logits, neg_logits.view(-1, self.neg_nums)))

            pos_loss = F.logsigmoid(pos_logits).squeeze(dim=1)
            negs_loss = F.logsigmoid(-neg_logits).squeeze(dim=1)
            loss.append(-(pos_loss.mean() + negs_loss.mean()))
            if etype == 'cor':
                print("Cor Loss "+str(loss) + "       mrr " + str(_mrr))
        return sum(loss) / len(loss), sum(_mrr) / len(_mrr)


class HyperbolicLoss(nn.Module):
    def __init__(self, neg_nums, edge_types):
        super(HyperbolicLoss, self).__init__()
        self.neg_nums = neg_nums
        self.edge_types = edge_types

    def forward(self, block_outputs, pos_graph, neg_graph):
        loss = []
        _mrr = []
        etype='cor'
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs[etype]
            pos_graph.apply_edges(hyperbolic_dis)
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs[etype]
            neg_graph.apply_edges(hyperbolic_dis)
            neg_score = neg_graph.edata['score']
        #if pos_score[list(pos_score.keys())[0]].shape[0] == 0:
        
        pos_logits = pos_score
        neg_logits = neg_score
        _mrr.append(mrr(-pos_logits, -neg_logits.view(-1, self.neg_nums)))
        pos_loss = pos_logits.squeeze(dim=1)
        neg_loss = neg_logits.squeeze(dim=1)
        #print(pos_loss)
        #print(neg_loss)
        pos_loss = torch.log(1-torch.sigmoid(pos_loss)+1e-5)
        neg_loss = torch.log(torch.sigmoid(neg_loss)+1e-5)
        #pos_loss = F.logsigmoid(-pos_logits).squeeze(dim=1)
        #negs_loss = F.logsigmoid(neg_logits).squeeze(dim=1)
#         print(pos_loss)
#         print(neg_loss)
        loss_mean = -pos_loss.mean() - neg_loss.mean()
        loss.append(loss_mean)
        return sum(loss) / len(loss), sum(_mrr) / len(_mrr)

class HATLoss(nn.Module):
    def __init__(self, neg_nums, edge_types):
        super(HATLoss, self).__init__()
        self.neg_nums = neg_nums
        self.edge_types = edge_types

    def forward(self, block_outputs, pos_graph, neg_graph):
        loss = []
        _mrr = []
        for etype in self.edge_types:
            if etype == 'sim':
                continue
            print(pos_graph)
            print(neg_graph)
            with pos_graph.local_scope():
                pos_graph.ndata['h'] = block_outputs[etype]
                pos_graph.apply_edges(hyperbolic_dis, etype=etype)
                pos_score = pos_graph.edata['score']
            with neg_graph.local_scope():
                neg_graph.ndata['h'] = block_outputs[etype]
                neg_graph.apply_edges(hyperbolic_dis, etype=etype)
                neg_score = neg_graph.edata['score']
            if pos_score[list(pos_score.keys())[0]].shape[0] == 0:
                continue
            pos_logits = pos_score[list(pos_score.keys())[0]]
            neg_logits = neg_score[list(neg_score.keys())[0]]
            _mrr.append(mrr(-pos_logits, -neg_logits.view(-1, self.neg_nums)))

            pos_loss = pos_logits.squeeze(dim=1)
            negs_loss = neg_logits.squeeze(dim=1)
            print(pos_loss)
            print(negs_loss)
            # print(pos_loss.mean())
            # print(1)
            # print(negs_loss.mean())
#             loss_mean = pos_loss.mean() + negs_loss.mean() + 150
#             loss_mean[loss_mean<0] = 0
#             loss.append(loss_mean)
            pos_loss = torch.log(1-torch.sigmoid(pos_logits))
            neg_loss = torch.log(torch.sigmoid(neg_logits))
            loss_mean = - pos_loss.mean() - neg_loss.mean()
            loss.append(loss_mean)
            # loss.append(-(pos_loss.mean() + negs_loss.mean()))
        return sum(loss) / len(loss), sum(_mrr) / len(_mrr)

def hyperbolic_dis(edges):
    # scores = torch.matmul(edges.src['h'].unsqueeze(dim=-2), edges.dst['h'].unsqueeze(dim=-1))
    from hyperbolic.PoincareManifold import PoincareManifold
    poincare = PoincareManifold(0, 0)
    src_emb, dis_emb = edges.src.pop('h'), edges.dst.pop('h')
    scores = poincare.distance(src_emb, dis_emb) ** 2
    return {'score': scores}

def hyperbolic_inner(edges):
    # scores = torch.matmul(edges.src['h'].unsqueeze(dim=-2), edges.dst['h'].unsqueeze(dim=-1))
    from hyperbolic.PoincareManifold import PoincareManifold
    poincare = PoincareManifold(0, 0)
    src_emb, dis_emb = edges.src.pop('h'), edges.dst.pop('h')
    scores = poincare.mob_matrix_multi(src_emb.unsqueeze(dim=-2), dis_emb.unsqueeze(dim=-2)).squeeze(dim=2)
    return {'score': scores}




# class CrossEntropyLoss(nn.Module):
#     def __init__(self, neg_nums):
#         super(CrossEntropyLoss, self).__init__()
#         self.neg_nums = neg_nums
#
#     def forward(self, block_outputs, pos_graph, neg_graph):
#         with pos_graph.local_scope():
#             pos_graph.ndata['h'] = block_outputs
#             pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
#             pos_score = pos_graph.edata['score']
#         with neg_graph.local_scope():
#             neg_graph.ndata['h'] = block_outputs
#             neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
#             neg_score = neg_graph.edata['score']
#         pos_logits = pos_score.view(1, -1)
#         neg_logits = neg_score.view(self.neg_nums, -1)
#
#         pos_loss = F.logsigmoid(-pos_logits).sum(dim=0)
#         negs_loss = F.logsigmoid(neg_logits).sum(dim=0)
#         loss = -(pos_loss + negs_loss).mean()
#         return loss

